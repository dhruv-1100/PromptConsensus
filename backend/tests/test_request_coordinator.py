"""
Tests for pipeline request deduplication: identical in-flight runs rejoin,
completed runs are reused briefly, and a joiner is never stuck forever.
"""
import threading
import time

import pytest

import request_coordinator as rc
from request_coordinator import build_request_key, run_deduplicated


@pytest.fixture(autouse=True)
def clean_coordinator_state():
    rc._INFLIGHT.clear()
    rc._COMPLETED.clear()
    original_timeout = rc._JOIN_TIMEOUT_SECONDS
    yield
    rc._JOIN_TIMEOUT_SECONDS = original_timeout
    rc._INFLIGHT.clear()
    rc._COMPLETED.clear()


class TestRequestKey:
    def test_same_request_same_key(self):
        assert build_request_key(" summarise this ", "Healthcare", True) == build_request_key(
            "summarise this", "healthcare", True
        )

    def test_domain_and_demo_mode_change_the_key(self):
        base = build_request_key("q", "general", True)
        assert base != build_request_key("q", "legal", True)
        assert base != build_request_key("q", "general", False)

    def test_engine_changes_the_key(self):
        # Without this, a langgraph request could be served the asyncio run's
        # cached result, making the two engines impossible to compare.
        assert build_request_key("q", "general", True, "asyncio") != build_request_key(
            "q", "general", True, "langgraph"
        )

    def test_engine_defaults_to_asyncio(self):
        assert build_request_key("q", "general", True) == build_request_key(
            "q", "general", True, "asyncio"
        )

    def test_blank_domain_is_treated_as_general(self):
        assert build_request_key("q", "", True) == build_request_key("q", "general", True)


class TestDeduplication:
    def test_first_call_runs_the_worker(self):
        result, source = run_deduplicated("k", lambda: "value")
        assert (result, source) == ("value", "new")

    def test_repeat_call_reuses_the_completed_result(self):
        run_deduplicated("k", lambda: "first")
        result, source = run_deduplicated("k", lambda: pytest.fail("should not re-run"))
        assert (result, source) == ("first", "cached")

    def test_expired_cache_entry_runs_again(self):
        run_deduplicated("k", lambda: "first")
        rc._COMPLETED["k"] = (time.time() - rc._TTL_SECONDS - 1, "first")
        result, source = run_deduplicated("k", lambda: "second")
        assert (result, source) == ("second", "new")

    def test_concurrent_identical_requests_run_the_worker_once(self):
        calls = []
        release = threading.Event()
        sources = {}

        def worker():
            calls.append(1)
            release.wait(5)
            return "shared"

        def leader():
            sources["leader"] = run_deduplicated("k", worker)[1]

        def joiner():
            sources["joiner"] = run_deduplicated("k", worker)[1]

        t_leader = threading.Thread(target=leader)
        t_leader.start()
        while "k" not in rc._INFLIGHT:  # wait until the leader has registered
            time.sleep(0.01)
        t_joiner = threading.Thread(target=joiner)
        t_joiner.start()
        release.set()
        t_leader.join(5)
        t_joiner.join(5)

        assert len(calls) == 1
        assert sources == {"leader": "new", "joiner": "joined"}

    def test_worker_failure_propagates_to_joiners(self):
        release = threading.Event()
        errors = {}

        def failing_worker():
            release.wait(5)
            raise RuntimeError("model call failed")

        def leader():
            try:
                run_deduplicated("k", failing_worker)
            except RuntimeError as exc:
                errors["leader"] = str(exc)

        def joiner():
            try:
                run_deduplicated("k", failing_worker)
            except RuntimeError as exc:
                errors["joiner"] = str(exc)

        t_leader = threading.Thread(target=leader)
        t_leader.start()
        while "k" not in rc._INFLIGHT:
            time.sleep(0.01)
        t_joiner = threading.Thread(target=joiner)
        t_joiner.start()
        release.set()
        t_leader.join(5)
        t_joiner.join(5)

        assert errors["leader"] == "model call failed"
        assert errors["joiner"] == "model call failed"
        assert "k" not in rc._INFLIGHT  # failed key is not left behind

    def test_a_failed_run_is_not_cached(self):
        with pytest.raises(RuntimeError):
            run_deduplicated("k", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result, source = run_deduplicated("k", lambda: "retry works")
        assert (result, source) == ("retry works", "new")

    def test_a_killed_leader_still_releases_joiners(self):
        # BaseException (a cancelled or killed worker thread) must not leave the
        # key signalled-never.
        release = threading.Event()
        outcome = {}

        def leader():
            def worker():
                release.wait(5)
                raise KeyboardInterrupt("killed")

            try:
                run_deduplicated("k", worker)
            except BaseException as exc:
                outcome["leader"] = type(exc).__name__

        def joiner():
            try:
                run_deduplicated("k", lambda: "unused")
            except RuntimeError as exc:
                outcome["joiner"] = str(exc)

        t_leader = threading.Thread(target=leader)
        t_leader.start()
        while "k" not in rc._INFLIGHT:
            time.sleep(0.01)
        t_joiner = threading.Thread(target=joiner)
        t_joiner.start()
        release.set()
        t_leader.join(5)
        t_joiner.join(5)

        assert outcome["leader"] == "KeyboardInterrupt"
        assert not t_joiner.is_alive()

    def test_joiner_gives_up_instead_of_hanging_forever(self):
        # Regression: the wait had no timeout, so a leader that died without
        # signalling blocked every joiner indefinitely.
        rc._JOIN_TIMEOUT_SECONDS = 0.2
        rc._INFLIGHT["k"] = rc.InflightRequest()  # leader that never signals

        started = time.time()
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            run_deduplicated("k", lambda: "unused")
        assert time.time() - started < 3
