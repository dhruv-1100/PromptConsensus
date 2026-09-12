"""
Tests for the local JSON persistence layer: concurrent appends must not lose
entries, and an interrupted write must not truncate existing data.
"""
import json
import os
import threading

import pytest

from json_store import append_json_list, read_json_list, write_json_list


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "entries.json")


class TestRead:
    def test_missing_file_reads_as_empty(self, store):
        assert read_json_list(store) == []

    def test_corrupt_file_reads_as_empty(self, store):
        with open(store, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        assert read_json_list(store) == []

    def test_non_list_payload_reads_as_empty(self, store):
        with open(store, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        assert read_json_list(store) == []

    def test_a_corrupt_file_can_still_be_appended_to(self, store):
        with open(store, "w", encoding="utf-8") as f:
            f.write("truncated")
        append_json_list(store, {"id": 1})
        assert read_json_list(store) == [{"id": 1}]


class TestAppend:
    def test_returns_the_stored_entry(self, store):
        entry = {"id": "abc"}
        assert append_json_list(store, entry) is entry

    def test_appends_in_order(self, store):
        for i in range(5):
            append_json_list(store, {"i": i})
        assert [e["i"] for e in read_json_list(store)] == [0, 1, 2, 3, 4]

    def test_creates_missing_directories(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "log.json")
        append_json_list(nested, {"i": 1})
        assert read_json_list(nested) == [{"i": 1}]

    def test_max_entries_keeps_the_most_recent(self, store):
        for i in range(10):
            append_json_list(store, {"i": i}, max_entries=3)
        assert [e["i"] for e in read_json_list(store)] == [7, 8, 9]

    def test_concurrent_appends_do_not_lose_entries(self, store):
        # The regression this guards: an unsynchronised read-modify-write dropped
        # entries whenever two FastAPI worker threads submitted at the same time.
        writers = 40
        barrier = threading.Barrier(writers)

        def write(i):
            barrier.wait()  # maximise overlap
            append_json_list(store, {"i": i})

        threads = [threading.Thread(target=write, args=(i,)) for i in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stored = read_json_list(store)
        assert len(stored) == writers
        assert {e["i"] for e in stored} == set(range(writers))

    def test_concurrent_appends_respect_max_entries(self, store):
        threads = [
            threading.Thread(target=append_json_list, args=(store, {"i": i}), kwargs={"max_entries": 5})
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(read_json_list(store)) == 5


class TestAtomicWrite:
    def test_leaves_no_temp_files_behind(self, tmp_path):
        store = str(tmp_path / "entries.json")
        for i in range(3):
            append_json_list(store, {"i": i})
        leftovers = [name for name in os.listdir(tmp_path) if name.startswith(".tmp-")]
        assert leftovers == []

    def test_a_failed_write_preserves_the_previous_contents(self, tmp_path, monkeypatch):
        store = str(tmp_path / "entries.json")
        append_json_list(store, {"keep": "me"})

        # An unserialisable entry makes json.dump raise midway through the write.
        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            write_json_list(store, [{"broken": Unserialisable()}])

        assert read_json_list(store) == [{"keep": "me"}]
        assert [n for n in os.listdir(tmp_path) if n.startswith(".tmp-")] == []

    def test_written_json_is_ascii_safe_and_reloadable(self, tmp_path):
        store = str(tmp_path / "entries.json")
        append_json_list(store, {"text": "café — naïve"})
        with open(store, "r", encoding="ascii") as f:  # would raise if non-ascii leaked
            reloaded = json.load(f)
        assert reloaded == [{"text": "café — naïve"}]
