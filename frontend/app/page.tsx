"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import type { CouncilPhase } from "./CouncilScene";

const CouncilScene = dynamic(() => import("./CouncilScene"), { ssr: false });

/* ── Types ── */
interface PeerReview {
  reviewer: string;
  model: string;
  evaluation: string;
  parsed_ranking: string[];
}

interface AggregateRank {
  label: string;
  candidate: string;
  average_rank: number;
  votes: number;
  ranks: number[];
}

interface PipelineState {
  raw_query: string;
  intent: Record<string, any>;
  candidate_a: string;
  candidate_b: string;
  candidate_c: string;
  peer_reviews: PeerReview[];
  aggregate_rankings: AggregateRank[];
  label_map: Record<string, string>;
  chairman: { model: string; rationale: string };
  perspectives?: Record<string, string>;
  optimised_prompt: string;
}

interface DiffLine {
  type: "added" | "removed" | "unchanged";
  text: string;
}

interface SessionSummary {
  session_id: string;
  timestamp: string;
  domain?: string;
  compare_mode?: boolean;
  raw_query?: string;
  optimised_prompt?: string;
  final_prompt?: string;
  llm_response?: string;
  baseline_response?: string;
  text?: string;
  target_model?: string;
  chairman_model?: string;
  improvement?: number;
  control?: number;
  quality?: number;
  trust?: number;
  safety_report?: SafetyReport | null;
  safety_acknowledged?: boolean;
  intent?: Record<string, any>;
  candidate_a?: string;
  candidate_b?: string;
  candidate_c?: string;
  peer_reviews?: PeerReview[];
  aggregate_rankings?: AggregateRank[];
  label_map?: Record<string, string>;
  perspectives?: Record<string, string>;
  chairman?: { model?: string; rationale?: string };
}

interface SessionAnalytics {
  total_sessions: number;
  avg_quality: number;
  avg_trust: number;
  avg_improvement: number;
  avg_control: number;
  edit_rate: number;
  compare_mode_rate: number;
  top_domain: string | null;
  top_winner: string | null;
}

interface SafetyCheckItem {
  severity: "low" | "medium" | "high";
  title: string;
  detail: string;
  action: string;
}

interface SafetyReport {
  domain: string;
  sensitive_domain: boolean;
  risk_level: "none" | "low" | "medium" | "high";
  requires_acknowledgement: boolean;
  checks: SafetyCheckItem[];
  checked_prompt_length: number;
  raw_query_length: number;
}

type Stage = "input" | "processing" | "council" | "review" | "execute" | "feedback";

const STAGES: { key: Stage; label: string }[] = [
  { key: "input", label: "Input" },
  { key: "processing", label: "Processing" },
  { key: "council", label: "Council" },
  { key: "review", label: "Review" },
  { key: "execute", label: "Execute" },
  { key: "feedback", label: "Feedback" },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_QUERIES = [
  {
    label: "Clinical discharge summary",
    domain: "Healthcare",
    query: "Help me write a discharge summary for a diabetic patient who was admitted for hyperglycaemia",
  },
  {
    label: "Research abstract analysis",
    domain: "Research",
    query: "Summarise the key findings and methodology gaps in this machine learning paper about transformer architectures",
  },
  {
    label: "Lesson plan generation",
    domain: "Education",
    query: "Create a lesson plan for teaching photosynthesis to 9th graders",
  },
];

async function copyToClipboard(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { return false; }
}

function formatSessionTime(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function buildLineDiff(beforeText: string, afterText: string): DiffLine[] {
  const before = beforeText.split("\n");
  const after = afterText.split("\n");
  const dp = Array.from({ length: before.length + 1 }, () => Array(after.length + 1).fill(0));

  for (let i = before.length - 1; i >= 0; i -= 1) {
    for (let j = after.length - 1; j >= 0; j -= 1) {
      dp[i][j] = before[i] === after[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const diff: DiffLine[] = [];
  let i = 0;
  let j = 0;

  while (i < before.length && j < after.length) {
    if (before[i] === after[j]) {
      diff.push({ type: "unchanged", text: before[i] });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      diff.push({ type: "removed", text: before[i] });
      i += 1;
    } else {
      diff.push({ type: "added", text: after[j] });
      j += 1;
    }
  }

  while (i < before.length) {
    diff.push({ type: "removed", text: before[i] });
    i += 1;
  }

  while (j < after.length) {
    diff.push({ type: "added", text: after[j] });
    j += 1;
  }

  return diff;
}

function summarizeDiff(diff: DiffLine[]) {
  return diff.reduce((summary, line) => {
    if (line.type === "added") summary.added += 1;
    if (line.type === "removed") summary.removed += 1;
    return summary;
  }, { added: 0, removed: 0 });
}

function safeSnippet(text?: string, fallback = "No content recorded.") {
  const value = (text || "").trim();
  return value || fallback;
}

/* ── Header ── */
function Header({ theme, toggleTheme }: { theme: string; toggleTheme: () => void }) {
  return (
    <div className="brand-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <Image src="/logo.png" alt="ConsensusPrompt Logo" width={56} height={56} style={{ borderRadius: 8 }} priority />
        <div className="brand-text">
          <span className="brand-name">ConsensusPrompt</span>
          <span className="brand-sub">Multi-agent prompt optimisation</span>
        </div>
      </div>
      <label className="theme-switch" title="Toggle Light/Dark Mode">
        <input 
          type="checkbox" 
          checked={theme === "light"} 
          onChange={toggleTheme} 
        />
        <span className="theme-slider"></span>
      </label>
    </div>
  );
}

/* ── Stage Nav ── */
function StageNav({ current }: { current: Stage }) {
  const idx = STAGES.findIndex((s) => s.key === current);
  return (
    <div className="stage-nav">
      {STAGES.map((s, i) => (
        <React.Fragment key={s.key}>
          <span className={`stage-dot ${i < idx ? "completed" : i === idx ? "active" : "inactive"}`}>
            {i < idx ? "\u2713" : `0${i + 1}`} {s.label}
          </span>
          {i < STAGES.length - 1 && <span className="stage-chevron">{"\u203A"}</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

/* ── Stage 1 — Input ── */
function StageInput({
  onSubmit,
  recentSessions,
  analytics,
}: {
  onSubmit: (q: string, domain: string, model: string, demo: boolean, compare: boolean) => void;
  recentSessions: SessionSummary[];
  analytics: SessionAnalytics | null;
}) {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("General");
  const [model, setModel] = useState("gpt-4o");
  const [demo, setDemo] = useState(true);
  const [compareMode, setCompareMode] = useState(true);
  const [showHow, setShowHow] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(recentSessions[0]?.session_id || null);

  React.useEffect(() => {
    if (!recentSessions.length) {
      setSelectedSessionId(null);
      return;
    }
    setSelectedSessionId((current) => (
      current && recentSessions.some((session) => session.session_id === current)
        ? current
        : recentSessions[0].session_id
    ));
  }, [recentSessions]);

  const loadExample = (ex: typeof EXAMPLE_QUERIES[number]) => { setQuery(ex.query); setDomain(ex.domain); };
  const selectedSession = recentSessions.find((session) => session.session_id === selectedSessionId) || null;
  const selectedWinner = selectedSession?.aggregate_rankings?.[0];
  const selectedDiff = selectedSession
    ? buildLineDiff(selectedSession.optimised_prompt || "", selectedSession.final_prompt || "")
    : [];
  const selectedDiffSummary = summarizeDiff(selectedDiff);

  return (
    <>
      <h1 className="page-title">Enter your <span className="accent">prompt</span></h1>
      <p className="page-subtitle">
        Write what you need in plain language. The system rewrites it using three independent strategies,
        a council of models reviews and ranks them anonymously, then synthesises the strongest version
        for your review.
      </p>

      <button className="collapsible-header" onClick={() => setShowHow(!showHow)} style={{ marginBottom: showHow ? 0 : 20 }}>
        <span>How does this work?</span>
        <span style={{ transition: "transform 0.2s", transform: showHow ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
      </button>
      {showHow && (
        <div style={{ marginBottom: 28 }}>
          <div className="how-steps" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--terracotta)" }}>01</div>
              <div className="how-step-title">Write naturally</div>
              <div className="how-step-desc">Describe what you need in your own words. No prompt-engineering knowledge required.</div>
            </div>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--terracotta)" }}>02</div>
              <div className="how-step-title">Agents rewrite</div>
              <div className="how-step-desc">Three agents independently restructure your prompt using different strategies.</div>
            </div>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--terracotta)" }}>03</div>
              <div className="how-step-title">Council reviews</div>
              <div className="how-step-desc">Each model reviews the others anonymously, preventing bias. Rankings are aggregated and the best is synthesised.</div>
            </div>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--terracotta)" }}>04</div>
              <div className="how-step-title">You decide</div>
              <div className="how-step-desc">Compare original vs optimised side-by-side. Edit, approve, or reject before execution.</div>
            </div>
          </div>
          <div className="trust-banner">
            No prompt is dispatched without your explicit approval. You retain full control at every step.
          </div>
        </div>
      )}

      {demo && (
        <div className="info-banner" style={{ marginBottom: 24 }}>
          Running in demo mode — responses use fixture data. Toggle off and provide API keys for live inference.
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "3fr 1fr", gap: 28 }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <label className="field-label">Your Query</label>
            <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
              {query.length > 0 ? `${query.split(/\s+/).filter(Boolean).length} words` : ""}
            </span>
          </div>
          <textarea className="textarea-field" value={query} onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe what you need — the agents handle the rest." rows={6} />
        </div>
        <div>
          <label className="field-label">Domain</label>
          <select className="select-field" value={domain} onChange={(e) => setDomain(e.target.value)}>
            {["General", "Healthcare", "Education", "Legal", "Research", "Business", "Technology"].map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
          <div style={{ height: 14 }} />
          <label className="field-label">Target Model</label>
          <select className="select-field" value={model} onChange={(e) => setModel(e.target.value)}>
            {["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <div style={{ height: 14 }} />
          <label className="field-label">Mode</label>
          <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", fontSize: 13, color: "var(--text-secondary)" }}>
            <input type="checkbox" checked={demo} onChange={() => setDemo(!demo)} style={{ accentColor: "var(--terracotta)" }} />
            Demo mode
          </label>
          <div style={{ height: 14 }} />
          <label className="field-label">Study Tools</label>
          <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
            <input type="checkbox" checked={compareMode} onChange={() => setCompareMode(!compareMode)} style={{ accentColor: "var(--terracotta)" }} />
            Run direct baseline comparison
          </label>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 8, lineHeight: 1.5 }}>
            Sends the raw query and the approved optimised prompt to the same target model so you can compare both outputs side by side.
          </div>
        </div>
      </div>

      <div style={{ marginTop: 20 }}>
        <div className="section-label">Example Queries</div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {EXAMPLE_QUERIES.map((ex) => (
            <button key={ex.label} className="btn btn-secondary" style={{ fontSize: 12, padding: "7px 14px" }}
              onClick={() => loadExample(ex)}>{ex.label}</button>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 24, display: "flex", gap: 12 }}>
        <button className="btn btn-primary" disabled={!query.trim()}
          onClick={() => onSubmit(query.trim(), domain.toLowerCase(), model, demo, compareMode)}>
          Optimise prompt
        </button>
      </div>

      <div style={{ marginTop: 36 }}>
        <div className="section-label">Study Analytics</div>
        <div className="analytics-grid">
          <div className="analytics-card">
            <div className="analytics-value">{analytics?.total_sessions ?? 0}</div>
            <div className="analytics-label">Saved Sessions</div>
          </div>
          <div className="analytics-card">
            <div className="analytics-value">{analytics?.avg_quality?.toFixed(1) ?? "0.0"}</div>
            <div className="analytics-label">Avg Quality</div>
          </div>
          <div className="analytics-card">
            <div className="analytics-value">{analytics?.avg_trust?.toFixed(1) ?? "0.0"}</div>
            <div className="analytics-label">Avg Trust</div>
          </div>
          <div className="analytics-card">
            <div className="analytics-value">{analytics ? `${analytics.edit_rate}%` : "0%"}</div>
            <div className="analytics-label">Edit Rate</div>
          </div>
          <div className="analytics-card">
            <div className="analytics-value">{analytics ? `${analytics.compare_mode_rate}%` : "0%"}</div>
            <div className="analytics-label">Comparison Use</div>
          </div>
          <div className="analytics-card">
            <div className="analytics-value">{analytics?.top_winner || "-"}</div>
            <div className="analytics-label">Most Common Winner</div>
          </div>
        </div>
        <div className="analytics-footnote">
          Top domain: {analytics?.top_domain || "n/a"} • Avg improvement: {analytics?.avg_improvement?.toFixed(1) ?? "0.0"} • Avg control: {analytics?.avg_control?.toFixed(1) ?? "0.0"}
        </div>
      </div>

      <div style={{ marginTop: 36 }}>
        <div className="section-label">Recent Sessions</div>
        <div className="session-panel">
          <div className="session-panel-header">
            <div>
              <div className="session-panel-title">Study Runs</div>
              <div className="session-panel-subtitle">Recent saved sessions from this workspace.</div>
            </div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/json`} target="_blank" rel="noreferrer">
                Export JSON
              </a>
              <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/csv`} target="_blank" rel="noreferrer">
                Export CSV
              </a>
            </div>
          </div>
          {recentSessions.length === 0 ? (
            <div className="session-empty">No saved sessions yet. Complete a run and submit feedback to build your study log.</div>
          ) : (
            <div className="session-list">
              {recentSessions.slice(0, 5).map((session) => {
                const winner = session.aggregate_rankings?.[0];
                return (
                  <button
                    key={session.session_id}
                    type="button"
                    className={`session-row ${selectedSessionId === session.session_id ? "session-row-active" : ""}`}
                    onClick={() => setSelectedSessionId(session.session_id)}
                  >
                    <div className="session-main">
                      <div className="session-query">{session.raw_query || "Untitled session"}</div>
                      <div className="session-meta-line">
                        <span>{formatSessionTime(session.timestamp)}</span>
                        <span>{session.domain || "general"}</span>
                        <span>{session.compare_mode ? "comparison" : "single condition"}</span>
                        {winner && <span>winner {winner.candidate}</span>}
                      </div>
                    </div>
                    <div className="session-side">
                      <span className="session-score">Q {session.quality ?? "-"}</span>
                      <span className="session-score">T {session.trust ?? "-"}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {selectedSession && (
            <div className="session-detail-panel">
              <div className="session-detail-header">
                <div>
                  <div className="session-detail-kicker">Session Details</div>
                  <div className="session-detail-title">{selectedSession.raw_query || "Untitled session"}</div>
                  <div className="session-meta-line">
                    <span>{formatSessionTime(selectedSession.timestamp)}</span>
                    <span>{selectedSession.domain || "general"}</span>
                    <span>{selectedSession.compare_mode ? "comparison mode" : "single condition"}</span>
                    <span>{selectedSession.target_model || "unknown model"}</span>
                    {selectedSession.safety_report?.risk_level && selectedSession.safety_report.risk_level !== "none" && (
                      <span>
                        safety {selectedSession.safety_report.risk_level}
                        {selectedSession.safety_acknowledged ? " · acknowledged" : ""}
                      </span>
                    )}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => {
                      const restoredDomain = selectedSession.domain
                        ? selectedSession.domain.charAt(0).toUpperCase() + selectedSession.domain.slice(1)
                        : "General";
                      setQuery(selectedSession.raw_query || "");
                      setDomain(restoredDomain);
                      if (selectedSession.target_model) setModel(selectedSession.target_model);
                      setCompareMode(Boolean(selectedSession.compare_mode));
                    }}
                  >
                    Reuse query
                  </button>
                </div>
              </div>

              <div className="session-detail-grid">
                <div className="analytics-card">
                  <div className="analytics-value">{selectedSession.quality ?? "-"}</div>
                  <div className="analytics-label">Quality</div>
                </div>
                <div className="analytics-card">
                  <div className="analytics-value">{selectedSession.trust ?? "-"}</div>
                  <div className="analytics-label">Trust</div>
                </div>
                <div className="analytics-card">
                  <div className="analytics-value">{selectedSession.improvement ?? "-"}</div>
                  <div className="analytics-label">Improvement</div>
                </div>
                <div className="analytics-card">
                  <div className="analytics-value">{selectedSession.control ?? "-"}</div>
                  <div className="analytics-label">Control</div>
                </div>
              </div>

              {selectedSession.safety_report && selectedSession.safety_report.risk_level !== "none" && (
                <div className={`safety-panel risk-${selectedSession.safety_report.risk_level}`} style={{ marginTop: 18 }}>
                  <div className="safety-panel-header">
                    <div>
                      <div className="safety-panel-kicker">Saved Safety Record</div>
                      <div className="safety-panel-title">
                        {selectedSession.safety_report.risk_level === "high"
                          ? "This run triggered a higher-risk execution condition"
                          : "This run included safety checks before execution"}
                      </div>
                    </div>
                    <span className={`safety-badge risk-${selectedSession.safety_report.risk_level}`}>
                      {selectedSession.safety_report.domain} · {selectedSession.safety_report.risk_level}
                    </span>
                  </div>
                  <div className="safety-check-list">
                    {selectedSession.safety_report.checks.slice(0, 3).map((check, index) => (
                      <div key={`${check.title}-${index}`} className="safety-check-item">
                        <div className="safety-check-head">
                          <span className={`safety-pill risk-${check.severity}`}>{check.severity}</span>
                          <span>{check.title}</span>
                        </div>
                        <div className="safety-check-action">{check.action}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
                    {selectedSession.safety_acknowledged
                      ? "The participant explicitly acknowledged the warning before execution."
                      : "No explicit acknowledgement was required for this run."}
                  </div>
                </div>
              )}

              <div className="history-grid" style={{ marginTop: 18 }}>
                <div className="history-card">
                  <div className="history-step">Step 1</div>
                  <div className="history-title">Raw Query</div>
                  <div className="history-desc">{safeSnippet(selectedSession.raw_query)}</div>
                </div>
                <div className="history-card history-card-active">
                  <div className="history-step">Step 2</div>
                  <div className="history-title">Council Output</div>
                  <div className="history-meta">{selectedWinner?.candidate || "No winner recorded"}</div>
                  <div className="history-desc">{safeSnippet(selectedSession.optimised_prompt)}</div>
                </div>
                <div className={`history-card ${selectedSession.optimised_prompt !== selectedSession.final_prompt ? "history-card-user" : ""}`}>
                  <div className="history-step">Step 3</div>
                  <div className="history-title">Final Prompt</div>
                  <div className="history-meta">
                    {selectedDiffSummary.added || selectedDiffSummary.removed
                      ? `+${selectedDiffSummary.added} / -${selectedDiffSummary.removed} lines`
                      : "No user edits"}
                  </div>
                  <div className="history-desc">{safeSnippet(selectedSession.final_prompt)}</div>
                </div>
              </div>

              <div className="session-detail-sections">
                <div className="session-detail-section">
                  <div className="section-label">Council Summary</div>
                  <div className="rationale-box" style={{ marginBottom: 14 }}>
                    {selectedWinner
                      ? `${selectedWinner.candidate || selectedWinner.label} won with average rank ${selectedWinner.average_rank} across ${selectedWinner.votes} reviews.`
                      : "No aggregate ranking recorded for this run."}
                  </div>
                  {selectedSession.chairman?.rationale && (
                    <div className="session-detail-textblock">
                      {selectedSession.chairman.rationale}
                    </div>
                  )}
                  {!!selectedSession.peer_reviews?.length && (
                    <div className="session-review-list">
                      {selectedSession.peer_reviews.slice(0, 3).map((review, idx) => (
                        <div className="session-review-item" key={`${review.reviewer}-${idx}`}>
                          <div className="session-review-head">
                            <span>{review.reviewer}</span>
                            <span>{review.parsed_ranking?.join(" > ") || "No ranking"}</span>
                          </div>
                          <div className="session-review-body">{safeSnippet(review.evaluation)}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="session-detail-section">
                  <div className="section-label">Outputs</div>
                  <div className="comparison-grid" style={{ marginBottom: 0 }}>
                    {selectedSession.compare_mode && (
                      <div className="cmp-card">
                        <div className="cmp-card-header">
                          <span className="badge badge-original">Baseline</span>
                          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Raw query response</span>
                        </div>
                        <div className="cmp-card-body">{safeSnippet(selectedSession.baseline_response)}</div>
                      </div>
                    )}
                    <div className={`cmp-card ${selectedSession.compare_mode ? "optimised" : ""}`}>
                      <div className="cmp-card-header">
                        <span className="badge badge-optimised">Optimised</span>
                        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Approved prompt response</span>
                      </div>
                      <div className="cmp-card-body">{safeSnippet(selectedSession.llm_response)}</div>
                    </div>
                  </div>
                </div>
              </div>

              {!!selectedSession.text && (
                <div style={{ marginTop: 18 }}>
                  <div className="section-label">Participant Comment</div>
                  <div className="session-detail-textblock">{selectedSession.text}</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

/* ── Stage 2 — Processing ── */
function StageProcessing({
  rawQuery, onComplete, domain, demoMode,
}: {
  rawQuery: string; domain: string; demoMode: boolean;
  onComplete: (state: PipelineState) => void;
}) {
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("Initialising pipeline");
  const [agentStates, setAgentStates] = useState<string[]>(Array(7).fill("waiting"));
  const [error, setError] = useState<string | null>(null);
  const [agentModels, setAgentModels] = useState<Record<string, string>>({});

  React.useEffect(() => {
    fetch(`${API_URL}/api/config`)
      .then(r => r.json())
      .then(d => setAgentModels(d.models || {}))
      .catch(() => {});
  }, []);

  const agents = [
    { code: "S1", name: "Intent Extractor", model: agentModels.intent_extractor || "loading...", desc: "Parses domains, gaps, and constraints" },
    { code: "S2A", name: "Autonomous Prompt Engineer A", model: agentModels.rewriter_a || "loading...", desc: "Dynamic prompt optimization strategy" },
    { code: "S2B", name: "Autonomous Prompt Engineer B", model: agentModels.rewriter_b || "loading...", desc: "Dynamic prompt optimization strategy" },
    { code: "S2C", name: "Autonomous Prompt Engineer C", model: agentModels.rewriter_c || "loading...", desc: "Dynamic prompt optimization strategy" },
    { code: "CE-A", name: "Cross-Examination: Agent A", model: agentModels.reviewer_a || "loading...", desc: "Critiques opposing architectures" },
    { code: "CE-B", name: "Cross-Examination: Agent B", model: agentModels.reviewer_b || "loading...", desc: "Critiques opposing architectures" },
    { code: "CE-C", name: "Cross-Examination: Agent C", model: agentModels.reviewer_c || "loading...", desc: "Critiques opposing architectures" },
  ];

  const getAgentStatesForStage = (stage: string) => {
    switch (stage) {
      case "start":
        return ["waiting", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"];
      case "intent":
        return ["running", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"];
      case "intent_complete":
        return ["done", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"];
      case "rewriters":
        return ["done", "running", "running", "running", "waiting", "waiting", "waiting"];
      case "rewriters_complete":
        return ["done", "done", "done", "done", "waiting", "waiting", "waiting"];
      case "review":
        return ["done", "done", "done", "done", "running", "running", "running"];
      case "review_complete":
        return ["done", "done", "done", "done", "done", "done", "done"];
      case "chairman":
      case "complete":
        return ["done", "done", "done", "done", "done", "done", "done"];
      default:
        return ["waiting", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"];
    }
  };

  React.useEffect(() => {
    const controller = new AbortController();
    let hasReceivedStreamEvent = false;
    let hasCompleted = false;

    setProgress(2);
    setStatusMsg("Connecting to backend");
    setAgentStates(getAgentStatesForStage("start"));
    setError(null);

    const fallbackFetch = async () => {
      try {
        const res = await fetch(`${API_URL}/api/optimize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ raw_query: rawQuery, domain, demo_mode: demoMode }),
        });
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        const data = await res.json();
        setProgress(100);
        setAgentStates(getAgentStatesForStage("complete"));
        setStatusMsg("Consensus reached");
        hasCompleted = true;
        onComplete(data);
      } catch (err: any) {
        setError(`Could not reach the backend at ${API_URL}. Start the server with: uvicorn main:app --port 8000`);
      }
    };

    const streamPipeline = async () => {
      try {
        const res = await fetch(`${API_URL}/api/optimize/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ raw_query: rawQuery, domain, demo_mode: demoMode }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`Server responded with ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const chunk of events) {
            const lines = chunk
              .split("\n")
              .filter((line) => line.startsWith("data: "))
              .map((line) => line.slice(6));

            for (const line of lines) {
              const event = JSON.parse(line);

              if (event.type === "start") {
                hasReceivedStreamEvent = true;
                setProgress(event.progress || 2);
                setStatusMsg(event.message || "Connecting to backend");
                setAgentStates(getAgentStatesForStage("start"));
              }

              if (event.type === "progress") {
                hasReceivedStreamEvent = true;
                setProgress(event.progress || 0);
                setStatusMsg(event.message || "Working");
                setAgentStates(getAgentStatesForStage(event.stage));
              }

              if (event.type === "result") {
                hasReceivedStreamEvent = true;
                hasCompleted = true;
                setProgress(100);
                setStatusMsg("Consensus reached");
                setAgentStates(getAgentStatesForStage("complete"));
                onComplete(event.data);
              }

              if (event.type === "error") {
                throw new Error(event.message || "Streaming request failed");
              }
            }
          }
        }
      } catch (err: any) {
        if (controller.signal.aborted) return;
        if (!hasReceivedStreamEvent && !hasCompleted) {
          setStatusMsg("Streaming unavailable — falling back to standard request");
          await fallbackFetch();
          return;
        }

        if (!hasCompleted) {
          setError("The live progress connection was interrupted after work started. Retry this run to avoid double-submitting the pipeline.");
          setStatusMsg("Streaming interrupted");
        }
      }
    };

    streamPipeline();
    return () => controller.abort();
  }, [rawQuery, domain, demoMode, onComplete]);

  return (
    <>
      <h1 className="page-title">Processing</h1>
      <p className="page-subtitle">Rewriting, peer review, and consensus synthesis.</p>

      <div style={{
        fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-muted)",
        background: "var(--bg-surface)", border: "1px solid var(--border)",
        borderRadius: "var(--radius)", padding: "12px 16px", marginBottom: 28,
      }}>{rawQuery}</div>

      <div className="progress-bar-track" style={{ marginBottom: 8 }}>
        <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--mono)", marginBottom: 24 }}>
        {statusMsg}
      </div>

      {error && (
        <div style={{
          background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)",
          borderRadius: "var(--radius)", padding: "14px 18px", marginBottom: 20,
          fontSize: 13, color: "#ef4444", lineHeight: 1.6,
        }}>{error}</div>
      )}

      <div className="glass-card">
        {agents.map((agent, i) => (
          <div className="agent-row" key={agent.code}>
            <div className={`agent-indicator ${agentStates[i]}`} />
            <div className="agent-info">
              <div className="agent-name">{agent.name}</div>
              <div className="agent-desc">{agent.desc}</div>
            </div>
            <span className="agent-meta">{agent.code} / {agent.model}</span>
            <span className={`agent-status ${agentStates[i]}`}>
              {agentStates[i] === "done" ? "complete" : agentStates[i] === "running" ? "active" : "queued"}
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

/* ── Stage 3 — Council Deliberation ── */
function StageCouncil({
  pipelineState,
  theme,
  onContinue,
}: {
  pipelineState: PipelineState;
  theme: string;
  onContinue: () => void;
}) {
  const { peer_reviews, aggregate_rankings, label_map, chairman, candidate_a, candidate_b, candidate_c, perspectives } = pipelineState;
  const [revealedReviewers, setRevealedReviewers] = useState(0);
  const [showRankings, setShowRankings] = useState(false);
  const [showWinner, setShowWinner] = useState(false);
  const [activeReview, setActiveReview] = useState(-1);
  const [activeTab, setActiveTab] = useState<string>("a");
  const [showScores, setShowScores] = useState(false);
  const [councilPhase, setCouncilPhase] = useState<CouncilPhase>("idle");

  // Animate reveal of reviewers one by one + sync 3D phase
  React.useEffect(() => {
    const timers: NodeJS.Timeout[] = [];

    // Start reviewing phase
    timers.push(setTimeout(() => setCouncilPhase("reviewing"), 200));

    // Reveal each reviewer card with a delay
    peer_reviews.forEach((_, i) => {
      timers.push(setTimeout(() => setRevealedReviewers(i + 1), 800 * (i + 1)));
    });

    // Show rankings after all reviewers
    const rankTime = 800 * (peer_reviews.length + 1);
    timers.push(setTimeout(() => {
      setShowRankings(true);
      setCouncilPhase("ranking");
    }, rankTime));

    // Show winner
    const winnerTime = rankTime + 1200;
    timers.push(setTimeout(() => {
      setShowWinner(true);
      setCouncilPhase("winner");
    }, winnerTime));

    // Synthesis phase
    timers.push(setTimeout(() => setCouncilPhase("synthesis"), winnerTime + 1500));

    // Done
    timers.push(setTimeout(() => setCouncilPhase("done"), winnerTime + 4000));

    return () => timers.forEach(clearTimeout);
  }, [peer_reviews]);

  const winner = aggregate_rankings.length > 0 ? aggregate_rankings[0] : null;
  let winnerAgent = winner ? label_map[winner.label] || winner.label : "";
  if (winnerAgent && perspectives && perspectives[winnerAgent]) {
    winnerAgent = `${winnerAgent} (${perspectives[winnerAgent]})`;
  }

  // Map candidate letter to text
  const candidateText: Record<string, string> = { A: candidate_a, B: candidate_b, C: candidate_c };

  return (
    <>
      <h1 className="page-title">Council Deliberation</h1>
      <p className="page-subtitle">
        Three independent reviewers evaluated all candidate prompts using anonymised labels to prevent bias.
        Each reviewer ranked the candidates, and their rankings were aggregated to determine consensus.
      </p>

      {/* 3D Council Visualization */}
      <CouncilScene
        peerReviews={peer_reviews}
        aggregateRankings={aggregate_rankings}
        labelMap={label_map}
        perspectives={perspectives}
        phase={councilPhase}
        revealedCount={revealedReviewers}
        theme={theme}
      />

      {/* Peer review cards — animate in */}
      <div className="section-label">Peer Reviews</div>
      <div className="council-grid">
        {peer_reviews.slice(0, revealedReviewers).map((review, i) => (
          <div
            key={i}
            className={`council-card ${i === activeReview ? "council-card-active" : ""}`}
            onClick={() => setActiveReview(activeReview === i ? -1 : i)}
            style={{ animationDelay: `${i * 0.1}s`, cursor: 'pointer', display: 'flex', flexDirection: 'column' }}
          >
            <div className="council-card-header">
              <span className="council-card-title">{review.reviewer}</span>
              <span className="council-card-model">{review.model}</span>
            </div>
            <div className="council-card-ranking">
              {review.parsed_ranking.map((label, pos) => (
                <div key={label} className={`council-rank-item ${pos === 0 ? "council-rank-first" : ""}`}>
                  <span className="council-rank-pos">#{pos + 1}</span>
                  <span className="council-rank-label">{label_map[label] || label}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Selected reviewer's full evaluation */}
      {revealedReviewers > 0 && activeReview !== -1 && (
        <div style={{ marginTop: 16, marginBottom: 28, animation: "fadeIn 0.2s" }}>
          <div className="section-label" style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span>{peer_reviews[activeReview]?.reviewer} — Full Evaluation</span>
            <span style={{ color: "var(--terracotta)", background: "var(--bg-surface)", padding: "4px 10px", borderRadius: 4, fontSize: 13, textTransform: "none", fontWeight: 500 }}>
              Top Pick: {peer_reviews[activeReview]?.parsed_ranking[0]} = {label_map[peer_reviews[activeReview]?.parsed_ranking[0]] || peer_reviews[activeReview]?.parsed_ranking[0]}
            </span>
          </div>
          <div style={{
            fontFamily: "var(--mono)", fontSize: 13, lineHeight: 1.8,
            color: "var(--text-secondary)", whiteSpace: "pre-wrap",
            background: "var(--bg-surface)", border: "1px solid var(--border)",
            borderRadius: "var(--radius)", padding: "18px 22px",
          }}>
            {peer_reviews[activeReview]?.evaluation}
          </div>
        </div>
      )}

      {/* Agent candidates */}
      {revealedReviewers > 0 && (
        <>
          <button className="collapsible-header" onClick={() => setShowScores(!showScores)}
            style={{ marginBottom: showScores ? 0 : 28 }}>
            <span>View individual agent candidates</span>
            <span style={{ transition: "transform 0.2s", transform: showScores ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
          </button>
          {showScores && (
            <div className="glass-card" style={{ borderTopLeftRadius: 0, borderTopRightRadius: 0, marginBottom: 28 }}>
              <div className="tabs-header">
                <button className={`tab-btn ${activeTab === "a" ? "active" : ""}`} onClick={() => setActiveTab("a")}>
                  A: {perspectives?.["Candidate A"] ? (perspectives["Candidate A"].length > 25 ? perspectives["Candidate A"].slice(0, 25) + "..." : perspectives["Candidate A"]) : "Dynamic Strategy A"}
                </button>
                <button className={`tab-btn ${activeTab === "b" ? "active" : ""}`} onClick={() => setActiveTab("b")}>
                  B: {perspectives?.["Candidate B"] ? (perspectives["Candidate B"].length > 25 ? perspectives["Candidate B"].slice(0, 25) + "..." : perspectives["Candidate B"]) : "Dynamic Strategy B"}
                </button>
                <button className={`tab-btn ${activeTab === "c" ? "active" : ""}`} onClick={() => setActiveTab("c")}>
                  C: {perspectives?.["Candidate C"] ? (perspectives["Candidate C"].length > 25 ? perspectives["Candidate C"].slice(0, 25) + "..." : perspectives["Candidate C"]) : "Dynamic Strategy C"}
                </button>
              </div>
              <div style={{
                fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.8,
                color: "var(--text-secondary)", whiteSpace: "pre-wrap", padding: "12px 0",
              }}>
                {activeTab === "a" ? candidate_a : activeTab === "b" ? candidate_b : candidate_c}
              </div>
            </div>
          )}
        </>
      )}

      {/* Aggregate rankings with animated bars */}
      {showRankings && (
        <div style={{ marginBottom: 28 }} className="council-fade-in">
          <div className="section-label">Aggregate Consensus</div>
          <div className="glass-card">
            {aggregate_rankings.map((rank, i) => {
              const barWidth = Math.max(10, 100 - (rank.average_rank - 1) * 30);
              const isWinner = i === 0;
              return (
                <div key={rank.label} className={`council-agg-row ${isWinner ? "council-agg-winner" : ""}`}>
                  <div className="council-agg-rank">#{i + 1}</div>
                  <div className="council-agg-info">
                    <div className="council-agg-label">{label_map[rank.label] || rank.label}</div>
                  </div>
                  <div className="council-agg-bar-track">
                    <div
                      className={`council-agg-bar-fill ${isWinner ? "council-agg-bar-winner" : ""}`}
                      style={{ width: `${barWidth}%`, transition: "width 0.8s ease-out" }}
                    />
                  </div>
                  <div className="council-agg-score">
                    avg {rank.average_rank}
                  </div>
                  <div className="council-agg-votes">
                    {rank.ranks.map((r, j) => (
                      <span key={j} className="council-vote-chip">#{r}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Winner announcement */}
      {showWinner && winner && (
        <div className="council-winner-card council-fade-in">
          <div className="council-winner-badge">Consensus Winner</div>
          <div className="council-winner-label">{winnerAgent}</div>
          <div className="council-winner-stats">
            Average rank: {winner.average_rank} across {winner.votes} reviewers — unanimous first place
          </div>
          {chairman.rationale && (
            <div className="council-winner-rationale">
              <strong>Chairman synthesis:</strong> {chairman.rationale}
            </div>
          )}
        </div>
      )}

      {showWinner && (
        <div style={{ marginTop: 24 }}>
          <button className="btn btn-primary" onClick={onContinue}>
            Continue to review
          </button>
        </div>
      )}
    </>
  );
}

/* ── Stage 4 — Review ── */
function StageReview({
  pipelineState, onApprove, onStartOver,
}: {
  pipelineState: PipelineState;
  onApprove: (finalPrompt: string) => void;
  onStartOver: () => void;
}) {
  const { raw_query, intent, optimised_prompt, candidate_a, candidate_b, candidate_c,
    aggregate_rankings, label_map, chairman, perspectives } = pipelineState;
  const [editedPrompt, setEditedPrompt] = useState(optimised_prompt);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const ok = await copyToClipboard(editedPrompt);
    if (ok) { setCopied(true); setTimeout(() => setCopied(false), 2000); }
  };

  const origWords = raw_query.split(/\s+/).filter(Boolean).length;
  const optWords = optimised_prompt.split(/\s+/).filter(Boolean).length;
  const winner = aggregate_rankings.length > 0 ? aggregate_rankings[0] : null;
  const rawToOptimisedDiff = React.useMemo(() => buildLineDiff(raw_query, optimised_prompt), [raw_query, optimised_prompt]);
  const optimisedToFinalDiff = React.useMemo(() => buildLineDiff(optimised_prompt, editedPrompt), [optimised_prompt, editedPrompt]);
  const rawToOptimisedSummary = React.useMemo(() => summarizeDiff(rawToOptimisedDiff), [rawToOptimisedDiff]);
  const optimisedToFinalSummary = React.useMemo(() => summarizeDiff(optimisedToFinalDiff), [optimisedToFinalDiff]);
  const hasUserEdits = editedPrompt !== optimised_prompt;

  return (
    <>
      <h1 className="page-title">Review</h1>
      <p className="page-subtitle">
        The council reached consensus. Compare your original prompt with the synthesised version below.
        Edit if needed — your version is what gets sent.
      </p>

      <div className="trust-banner">
        Nothing is dispatched to the target model without your explicit approval. Edit, approve, or reject below.
      </div>

      {/* Council result summary */}
      {winner && (
        <div style={{
          display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20,
        }}>
          <span className="badge badge-optimised" style={{ fontSize: 11 }}>
            Council winner: {label_map[winner.label] || winner.label}
            {perspectives?.[label_map[winner.label] || winner.label] && ` (${perspectives[label_map[winner.label] || winner.label]})`}
          </span>
          <span className="badge badge-original" style={{ fontSize: 11 }}>
            Avg rank: {winner.average_rank} / {winner.votes} votes
          </span>
          {chairman.model && (
            <span className="badge badge-original" style={{ fontSize: 11 }}>
              Chairman: {chairman.model}
            </span>
          )}
        </div>
      )}

      {/* Intent chips */}
      {intent && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
          {intent.topic_domain && <span className="badge badge-optimised" style={{ fontSize: 11 }}>Topic: {intent.topic_domain}</span>}
          {intent.format_domain && <span className="badge badge-optimised" style={{ fontSize: 11 }}>Format: {intent.format_domain}</span>}
          {intent.query_quality && <span className="badge badge-original" style={{ fontSize: 11 }}>quality: {intent.query_quality}</span>}
          {(intent.missing_info || []).slice(0, 3).map((m: string, i: number) => (
            <span key={i} className="badge badge-original" style={{ fontSize: 11 }}>
              gap: {m.length > 45 ? m.slice(0, 45) + "\u2026" : m}
            </span>
          ))}
        </div>
      )}

      {/* Side-by-side */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div className="section-label">Prompt Comparison</div>
        <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)" }}>
          {origWords} words {"\u2192"} {optWords} words (+{Math.round(((optWords - origWords) / Math.max(origWords, 1)) * 100)}%)
        </span>
      </div>
      <div className="comparison-grid">
        <div className="cmp-card">
          <div className="cmp-card-header"><span className="badge badge-original">Original</span></div>
          <div className="cmp-card-body">{raw_query}</div>
        </div>
        <div className="cmp-card optimised">
          <div className="cmp-card-header"><span className="badge badge-optimised">Optimised</span></div>
          <div className="cmp-card-body">{optimised_prompt}</div>
        </div>
      </div>

      <div className="section-label">Prompt History</div>
      <div className="history-grid">
        <div className="history-card">
          <div className="history-step">Step 1</div>
          <div className="history-title">Raw Query</div>
          <div className="history-meta">{origWords} words</div>
          <div className="history-desc">Your original natural-language request before any agent intervention.</div>
        </div>
        <div className="history-card history-card-active">
          <div className="history-step">Step 2</div>
          <div className="history-title">Council Output</div>
          <div className="history-meta">{optWords} words</div>
          <div className="history-desc">
            Synthesised by the chairman after peer review.
            {rawToOptimisedSummary.added > 0 || rawToOptimisedSummary.removed > 0
              ? ` ${rawToOptimisedSummary.added} lines added, ${rawToOptimisedSummary.removed} lines replaced or removed.`
              : " No structural change detected."}
          </div>
        </div>
        <div className={`history-card ${hasUserEdits ? "history-card-user" : ""}`}>
          <div className="history-step">Step 3</div>
          <div className="history-title">Your Final Prompt</div>
          <div className="history-meta">{editedPrompt.split(/\s+/).filter(Boolean).length} words</div>
          <div className="history-desc">
            {hasUserEdits
              ? `You edited the council output before approval. ${optimisedToFinalSummary.added} lines added, ${optimisedToFinalSummary.removed} lines removed or replaced.`
              : "No user edits yet. If you approve now, the council output will be sent as-is."}
          </div>
        </div>
      </div>

      <div className="section-label">What Changed</div>
      <div className="comparison-grid" style={{ marginBottom: 20 }}>
        <div className="cmp-card">
          <div className="cmp-card-header" style={{ justifyContent: "space-between" }}>
            <span className="badge badge-original">Raw → Council Output</span>
            <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
              +{rawToOptimisedSummary.added} / -{rawToOptimisedSummary.removed} lines
            </span>
          </div>
          <div className="diff-card-body">
            {rawToOptimisedDiff.map((line, index) => (
              <div key={`raw-opt-${index}`} className={`diff-line diff-${line.type}`}>
                <span className="diff-marker">
                  {line.type === "added" ? "+" : line.type === "removed" ? "-" : " "}
                </span>
                <span>{line.text || " "}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="cmp-card optimised">
          <div className="cmp-card-header" style={{ justifyContent: "space-between" }}>
            <span className="badge badge-optimised">Council Output → Final Prompt</span>
            <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
              +{optimisedToFinalSummary.added} / -{optimisedToFinalSummary.removed} lines
            </span>
          </div>
          <div className="diff-card-body">
            {optimisedToFinalDiff.map((line, index) => (
              <div key={`opt-final-${index}`} className={`diff-line diff-${line.type}`}>
                <span className="diff-marker">
                  {line.type === "added" ? "+" : line.type === "removed" ? "-" : " "}
                </span>
                <span>{line.text || " "}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Edit field */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 16 }}>
        <div className="section-label">Final prompt (editable)</div>
        <button onClick={handleCopy} className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 12px" }}>
          {copied ? "Copied" : "Copy to clipboard"}
        </button>
      </div>
      <textarea className="textarea-field" value={editedPrompt} onChange={(e) => setEditedPrompt(e.target.value)} rows={8} />
      <div style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)", marginTop: 6, marginBottom: 8 }}>
        {editedPrompt.split(/\s+/).filter(Boolean).length} words
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 12 }}>
        <button className="btn btn-primary" onClick={() => onApprove(editedPrompt)}>Approve and execute</button>
        <button className="btn btn-danger" onClick={onStartOver}>Discard and start over</button>
      </div>
    </>
  );
}

/* ── Stage 5 — Execute ── */
function StageExecute({
  finalPrompt, rawQuery, domain, intent, targetModel, demoMode, compareMode, onBackToReview, onRate,
}: {
  finalPrompt: string;
  rawQuery: string;
  domain: string;
  intent: Record<string, any>;
  targetModel: string;
  demoMode: boolean;
  compareMode: boolean;
  onBackToReview: () => void;
  onRate: (responses: {
    optimised: string;
    baseline: string;
    safetyReport: SafetyReport | null;
    safetyAcknowledged: boolean;
  }) => void;
}) {
  const [optimisedResponse, setOptimisedResponse] = useState<string | null>(null);
  const [baselineResponse, setBaselineResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showPrompt, setShowPrompt] = useState(false);
  const [copiedTarget, setCopiedTarget] = useState<"optimised" | "baseline" | null>(null);
  const [safetyReport, setSafetyReport] = useState<SafetyReport | null>(null);
  const [awaitingSafetyAck, setAwaitingSafetyAck] = useState(false);
  const [safetyAcknowledged, setSafetyAcknowledged] = useState(false);

  const runExecution = React.useCallback(async () => {
    const runPrompt = async (prompt: string) => {
      const res = await fetch(`${API_URL}/api/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ final_prompt: prompt, target_model: targetModel, demo_mode: demoMode }),
      });
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      return data.response as string;
    };

    setLoading(true);
    try {
      const [optimised, baseline] = await Promise.all([
        runPrompt(finalPrompt),
        compareMode ? runPrompt(rawQuery) : Promise.resolve(""),
      ]);
      setOptimisedResponse(optimised);
      setBaselineResponse(compareMode ? baseline : null);
    } catch {
      const errorText = "Could not reach the backend. Verify the server is running on port 8000.";
      setOptimisedResponse(errorText);
      setBaselineResponse(compareMode ? errorText : null);
    }
    setLoading(false);
  }, [compareMode, demoMode, finalPrompt, rawQuery, targetModel]);

  React.useEffect(() => {
    const checkSafety = async () => {
      setLoading(true);
      setAwaitingSafetyAck(false);
      setSafetyReport(null);
      setSafetyAcknowledged(false);
      setOptimisedResponse(null);
      setBaselineResponse(null);
      try {
        const res = await fetch(`${API_URL}/api/safety-check`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            final_prompt: finalPrompt,
            raw_query: rawQuery,
            domain,
            intent,
          }),
        });
        if (!res.ok) throw new Error(`Status ${res.status}`);
        const report = await res.json();
        setSafetyReport(report);
        if (report.requires_acknowledgement) {
          setAwaitingSafetyAck(true);
          setLoading(false);
          return;
        }
        await runExecution();
      } catch {
        await runExecution();
      }
    };
    checkSafety();
  }, [domain, finalPrompt, intent, rawQuery, runExecution]);

  const ow = rawQuery.split(/\s+/).filter(Boolean).length;
  const fw = finalPrompt.split(/\s+/).filter(Boolean).length;
  const pct = Math.round(((fw - ow) / Math.max(ow, 1)) * 100);

  const handleCopy = async (target: "optimised" | "baseline") => {
    const text = target === "optimised" ? optimisedResponse : baselineResponse;
    if (text) {
      const ok = await copyToClipboard(text);
      if (ok) {
        setCopiedTarget(target);
        setTimeout(() => setCopiedTarget(null), 2000);
      }
    }
  };

  return (
    <>
      <h1 className="page-title">Result</h1>
      <p className="page-subtitle">
        Response generated by <span style={{ color: "var(--terracotta)", fontFamily: "var(--mono)", fontSize: 13 }}>{targetModel}</span> using your approved prompt.
      </p>

      {safetyReport && safetyReport.checks.length > 0 && (
        <div className={`safety-panel risk-${safetyReport.risk_level}`}>
          <div className="safety-panel-header">
            <div>
              <div className="safety-panel-kicker">Pre-execution Safety Check</div>
              <div className="safety-panel-title">
                {safetyReport.risk_level === "high"
                  ? "Prompt needs explicit confirmation"
                  : safetyReport.risk_level === "medium"
                    ? "Prompt has cautionary checks"
                    : "Prompt has minor execution notes"}
              </div>
            </div>
            <span className={`safety-badge risk-${safetyReport.risk_level}`}>
              {safetyReport.domain} · {safetyReport.risk_level}
            </span>
          </div>
          <div className="safety-check-list">
            {safetyReport.checks.map((check, index) => (
              <div key={`${check.title}-${index}`} className="safety-check-item">
                <div className="safety-check-head">
                  <span className={`safety-pill risk-${check.severity}`}>{check.severity}</span>
                  <span>{check.title}</span>
                </div>
                <div className="safety-check-detail">{check.detail}</div>
                <div className="safety-check-action">{check.action}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {compareMode && (
        <div className="info-banner" style={{ marginBottom: 20 }}>
          Baseline comparison is enabled. Both outputs below come from the same target model: one uses the raw query directly, and one uses the council-optimised prompt.
        </div>
      )}

      <button className="collapsible-header" onClick={() => setShowPrompt(!showPrompt)} style={{ marginBottom: showPrompt ? 0 : 20 }}>
        <span>View submitted prompt</span>
        <span style={{ transition: "transform 0.2s", transform: showPrompt ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
      </button>
      {showPrompt && <div className="collapsible-body" style={{ marginBottom: 20 }}>{finalPrompt}</div>}

      {awaitingSafetyAck ? (
        <div className="glass-card" style={{ marginBottom: 24 }}>
          <div className="section-label">Execution Paused</div>
          <p style={{ fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 18 }}>
            This prompt touches a higher-risk execution condition for the selected domain. Review the checks above, then either revise the prompt or continue with explicit acknowledgement.
          </p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <button className="btn btn-secondary" onClick={onBackToReview}>Back to review</button>
            <button
              className="btn btn-primary"
              onClick={async () => {
                setAwaitingSafetyAck(false);
                setSafetyAcknowledged(true);
                await runExecution();
              }}
            >
              Execute with acknowledgement
            </button>
          </div>
        </div>
      ) : loading ? (
        <div className="loading-shimmer" style={{ height: 200, marginBottom: 24 }} />
      ) : compareMode ? (
        <div className="comparison-grid" style={{ alignItems: "start", marginBottom: 24 }}>
          <div className="cmp-card">
            <div className="cmp-card-header" style={{ justifyContent: "space-between" }}>
              <span className="badge badge-original">Direct Raw Prompt</span>
              <button onClick={() => handleCopy("baseline")} className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 12px" }}>
                {copiedTarget === "baseline" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="response-card" style={{ border: "none", borderRadius: 0, padding: 20 }} dangerouslySetInnerHTML={{ __html: formatMarkdown(baselineResponse || "") }} />
          </div>
          <div className="cmp-card optimised">
            <div className="cmp-card-header" style={{ justifyContent: "space-between" }}>
              <span className="badge badge-optimised">ConsensusPrompt</span>
              <button onClick={() => handleCopy("optimised")} className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 12px" }}>
                {copiedTarget === "optimised" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="response-card" style={{ border: "none", borderRadius: 0, padding: 20 }} dangerouslySetInnerHTML={{ __html: formatMarkdown(optimisedResponse || "") }} />
          </div>
        </div>
      ) : (
        <div style={{ position: "relative" }}>
          <div className="response-card" dangerouslySetInnerHTML={{ __html: formatMarkdown(optimisedResponse || "") }} />
          <button onClick={() => handleCopy("optimised")} className="btn btn-secondary"
            style={{ position: "absolute", top: 12, right: 12, fontSize: 11, padding: "4px 12px" }}>
            {copiedTarget === "optimised" ? "Copied" : "Copy"}
          </button>
        </div>
      )}

      <div className="metrics-strip">
        <div className="metric-cell"><div className="metric-value">{ow}</div><div className="metric-label">Original words</div></div>
        <div className="metric-cell"><div className="metric-value">{fw}</div><div className="metric-label">Optimised words</div></div>
        <div className="metric-cell"><div className="metric-value">+{pct}%</div><div className="metric-label">Specificity gain</div></div>
        <div className="metric-cell"><div className="metric-value">{compareMode ? 2 : 1}</div><div className="metric-label">Prompt conditions</div></div>
      </div>

      <button
        className="btn btn-primary"
        onClick={() => onRate({
          optimised: optimisedResponse || "",
          baseline: baselineResponse || "",
          safetyReport,
          safetyAcknowledged,
        })}
        disabled={loading || awaitingSafetyAck}
      >
        {compareMode ? "Rate this comparison" : "Rate this response"}
      </button>
    </>
  );
}

/* ── Stage 6 — Feedback ── */
function StageFeedback({
  onReset,
  rawQuery,
  domain,
  optimisedPrompt,
  finalPrompt,
  llmResponse,
  baselineResponse,
  targetModel,
  chairmanModel,
  compareMode,
  safetyReport,
  safetyAcknowledged,
  pipelineState,
  onSessionSaved,
}: {
  onReset: () => void;
  rawQuery: string;
  domain: string;
  optimisedPrompt: string;
  finalPrompt: string;
  llmResponse: string;
  baselineResponse: string;
  targetModel: string;
  chairmanModel: string;
  compareMode: boolean;
  safetyReport: SafetyReport | null;
  safetyAcknowledged: boolean;
  pipelineState: PipelineState | null;
  onSessionSaved: () => void;
}) {
  const [submitted, setSubmitted] = useState(false);
  const [savedSessionId, setSavedSessionId] = useState("");
  const [quality, setQuality] = useState(4);
  const [improvement, setImprovement] = useState(4);
  const [trust, setTrust] = useState(4);
  const [control, setControl] = useState(5);
  const [comment, setComment] = useState("");
  const labels: Record<number, string> = { 1: "Poor", 2: "Fair", 3: "Neutral", 4: "Good", 5: "Excellent" };

  const handleSubmit = async () => {
    try {
      const res = await fetch(`${API_URL}/api/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          quality,
          improvement,
          trust,
          control,
          text: comment,
          raw_query: rawQuery,
          domain,
          optimised_prompt: optimisedPrompt,
          final_prompt: finalPrompt,
          llm_response: llmResponse,
          baseline_response: baselineResponse,
          target_model: targetModel,
          chairman_model: chairmanModel,
          compare_mode: compareMode,
          safety_report: safetyReport,
          safety_acknowledged: safetyAcknowledged,
          intent: pipelineState?.intent || {},
          candidate_a: pipelineState?.candidate_a || "",
          candidate_b: pipelineState?.candidate_b || "",
          candidate_c: pipelineState?.candidate_c || "",
          peer_reviews: pipelineState?.peer_reviews || [],
          aggregate_rankings: pipelineState?.aggregate_rankings || [],
          label_map: pipelineState?.label_map || {},
          perspectives: pipelineState?.perspectives || {},
          chairman: pipelineState?.chairman || {},
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setSavedSessionId(data.session_id || "");
        onSessionSaved();
      }
    } catch {}
    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="success-state">
        <div className="success-icon">{"\u2713"}</div>
        <div className="success-title">Feedback recorded</div>
        <div className="success-sub">Your ratings contribute to ongoing evaluation of the council model.</div>
        {savedSessionId && (
          <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
            Session ID: {savedSessionId}
          </div>
        )}
        <div style={{ marginTop: 18, display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/json`} target="_blank" rel="noreferrer">
            Download JSON
          </a>
          <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/csv`} target="_blank" rel="noreferrer">
            Download CSV
          </a>
        </div>
        <div style={{ height: 28 }} />
        <button className="btn btn-secondary" onClick={onReset}>New query</button>
      </div>
    );
  }

  const RatingSlider = ({ label, hint, value, onChange }: { label: string; hint: string; value: number; onChange: (v: number) => void }) => (
    <div className="slider-container">
      <label className="field-label">{label}</label>
      <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 8, marginTop: -4 }}>{hint}</div>
      <input type="range" min={1} max={5} step={1} value={value} onChange={(e) => onChange(parseInt(e.target.value))} style={{ width: "100%" }} />
      <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--terracotta)", marginTop: 6 }}>{value}/5 — {labels[value]}</div>
    </div>
  );

  return (
    <>
      <h1 className="page-title">Feedback</h1>
      <p className="page-subtitle">
        {compareMode
          ? "Rate this comparison session across four dimensions aligned with your study goals."
          : "Rate this session across four dimensions aligned with your study goals."}
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
        <div>
          <RatingSlider label="Response quality" hint="How useful and accurate was the model output?" value={quality} onChange={setQuality} />
          <RatingSlider label="Prompt improvement" hint="Was the optimised prompt meaningfully better than your original?" value={improvement} onChange={setImprovement} />
        </div>
        <div>
          <RatingSlider label="Transparency" hint="Could you follow how and why your prompt was changed?" value={trust} onChange={setTrust} />
          <RatingSlider label="User agency" hint="Did you feel you retained meaningful control over the process?" value={control} onChange={setControl} />
        </div>
      </div>
      <label className="field-label" style={{ marginTop: 12 }}>Additional comments</label>
      <textarea className="textarea-field" value={comment} onChange={(e) => setComment(e.target.value)}
        placeholder="Optional — note anything that worked well or could be improved." rows={3} />
      <div style={{ marginTop: 16 }}>
        <button className="btn btn-primary" onClick={handleSubmit}>Submit</button>
      </div>
    </>
  );
}

/* ── Markdown formatter ── */
function formatMarkdown(text: string): string {
  return text
    .replace(/## (.*?)$/gm, "<h2>$1</h2>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^\- (.*)$/gm, "<li>$1</li>")
    .replace(/(<li>[\s\S]*<\/li>)/g, "<ul>$1</ul>")
    .replace(/\n\n/g, "<br/><br/>")
    .replace(/\|(.+)\|/g, (match) => {
      const cells = match.split("|").filter(Boolean).map((c) => c.trim());
      if (cells.every((c) => /^[-:]+$/.test(c))) return "";
      return `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
    })
    .replace(/(<tr>[\s\S]*<\/tr>)/g, "<table>$1</table>")
    .replace(/---/g, "<hr/>")
    .replace(/\n/g, "<br/>");
}

/* ── Main Page ── */
export default function Home() {
  const [theme, setTheme] = useState<"dark" | "light">("light");
  const [stage, setStage] = useState<Stage>("input");
  const [rawQuery, setRawQuery] = useState("");
  const [domain, setDomain] = useState("general");
  const [targetModel, setTargetModel] = useState("gpt-4o");
  const [demoMode, setDemoMode] = useState(true);
  const [compareMode, setCompareMode] = useState(true);
  const [pipelineState, setPipelineState] = useState<PipelineState | null>(null);
  const [finalPrompt, setFinalPrompt] = useState("");
  const [llmResponse, setLlmResponse] = useState("");
  const [baselineResponse, setBaselineResponse] = useState("");
  const [safetyReport, setSafetyReport] = useState<SafetyReport | null>(null);
  const [safetyAcknowledged, setSafetyAcknowledged] = useState(false);
  const [recentSessions, setRecentSessions] = useState<SessionSummary[]>([]);
  const [analytics, setAnalytics] = useState<SessionAnalytics | null>(null);

  React.useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const loadRecentSessions = React.useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/sessions`);
      if (!res.ok) throw new Error("Failed to load sessions");
      const data = await res.json();
      setRecentSessions(data.sessions || []);
    } catch {
      setRecentSessions([]);
    }
  }, []);

  const loadAnalytics = React.useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/sessions/analytics`);
      if (!res.ok) throw new Error("Failed to load analytics");
      const data = await res.json();
      setAnalytics(data);
    } catch {
      setAnalytics(null);
    }
  }, []);

  React.useEffect(() => {
    loadRecentSessions();
    loadAnalytics();
  }, [loadRecentSessions, loadAnalytics]);

  const handleInputSubmit = (q: string, d: string, m: string, demo: boolean, compare: boolean) => {
    setRawQuery(q); setDomain(d); setTargetModel(m); setDemoMode(demo); setCompareMode(compare); setStage("processing");
  };

  const handleProcessingComplete = React.useCallback((state: PipelineState) => {
    setPipelineState(state);
    setFinalPrompt(state.optimised_prompt);
    setStage("council");
  }, []);

  const handleApprove = (prompt: string) => { setFinalPrompt(prompt); setStage("execute"); };

  const handleReset = () => {
    setStage("input"); setRawQuery(""); setPipelineState(null); setFinalPrompt(""); setLlmResponse(""); setBaselineResponse(""); setSafetyReport(null); setSafetyAcknowledged(false);
  };

  return (
    <div className="app-shell">
      <Header theme={theme} toggleTheme={() => setTheme(t => t === "dark" ? "light" : "dark")} />
      <StageNav current={stage} />
      {stage === "input" && <StageInput onSubmit={handleInputSubmit} recentSessions={recentSessions} analytics={analytics} />}
      {stage === "processing" && (
        <StageProcessing rawQuery={rawQuery} domain={domain} demoMode={demoMode} onComplete={handleProcessingComplete} />
      )}
      {stage === "council" && pipelineState && (
        <StageCouncil pipelineState={pipelineState} theme={theme} onContinue={() => setStage("review")} />
      )}
      {stage === "review" && pipelineState && (
        <StageReview pipelineState={pipelineState} onApprove={handleApprove} onStartOver={handleReset} />
      )}
      {stage === "execute" && (
        <StageExecute
          finalPrompt={finalPrompt}
          rawQuery={rawQuery}
          domain={domain}
          intent={pipelineState?.intent || {}}
          targetModel={targetModel}
          demoMode={demoMode}
          compareMode={compareMode}
          onBackToReview={() => setStage("review")}
          onRate={({ optimised, baseline, safetyReport: report, safetyAcknowledged: acknowledged }) => {
            setLlmResponse(optimised);
            setBaselineResponse(baseline);
            setSafetyReport(report);
            setSafetyAcknowledged(acknowledged);
            setStage("feedback");
          }}
        />
      )}
      {stage === "feedback" && (
        <StageFeedback
          onReset={handleReset}
          rawQuery={rawQuery}
          domain={domain}
          optimisedPrompt={pipelineState?.optimised_prompt || ""}
          finalPrompt={finalPrompt}
          llmResponse={llmResponse}
          baselineResponse={baselineResponse}
          targetModel={targetModel}
          chairmanModel={pipelineState?.chairman?.model || ""}
          compareMode={compareMode}
          safetyReport={safetyReport}
          safetyAcknowledged={safetyAcknowledged}
          pipelineState={pipelineState}
          onSessionSaved={() => {
            loadRecentSessions();
            loadAnalytics();
          }}
        />
      )}
    </div>
  );
}
