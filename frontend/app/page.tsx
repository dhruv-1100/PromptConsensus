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

interface ConsensusDiagnostics {
  winner_label?: string | null;
  winner_candidate?: string | null;
  winner_average_rank?: number | null;
  winner_margin?: number;
  first_place_support_pct?: number;
  reviewer_agreement_pct?: number;
  consensus_strength_pct?: number;
  consensus_label?: string;
  is_unanimous_winner?: boolean;
  needs_human_review?: boolean;
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
  consensus_diagnostics?: ConsensusDiagnostics;
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
  consensus_diagnostics?: ConsensusDiagnostics;
  perspectives?: Record<string, string>;
  chairman?: { model?: string; rationale?: string };
  research_insights?: ResearchInsights;
  intervention_labels?: string[];
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
  avg_consensus_strength: number;
  avg_rewrite_diversity: number;
  avg_human_intervention: number;
  acceptance_rate: number;
}

interface ResearchInsights {
  winner_candidate?: string | null;
  reviewer_count: number;
  winner_first_place_support_pct: number;
  reviewer_agreement_pct: number;
  consensus_strength_pct: number;
  consensus_label: string;
  winner_margin: number;
  rewrite_diversity_pct: number;
  diversity_label: string;
  optimization_shift_pct: number;
  human_edit_shift_pct: number;
  human_edit_level: string;
  accepted_without_edit: boolean;
  consensus_response?: string;
  accepted_with_refinement?: boolean;
  overrode_consensus?: boolean;
  compare_mode: boolean;
  response_length_delta_pct?: number | null;
  safety_acknowledged: boolean;
  intervention_labels: string[];
  // Candidate diversity metrics
  candidate_pattern_convergence?: string;
  candidate_token_diversity?: number;
  candidate_structural_diversity?: number;
  candidate_style_divergence?: number;
}

interface PreferenceStats {
  total_pairs: number;
  edited_pairs: number;
  accepted_pairs: number;
  edit_rate_pct: number;
  domain_distribution: Record<string, number>;
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

function getConsensusReasons(peerReviews: PeerReview[], chairmanRationale = "") {
  const buckets = [
    {
      label: "Clearer output structure",
      keywords: ["structure", "structured", "template", "format", "sections", "table", "layout"],
    },
    {
      label: "Stronger constraints",
      keywords: ["constraint", "constraints", "requirement", "requirements", "compliance", "guardrail", "specific"],
    },
    {
      label: "More complete coverage",
      keywords: ["complete", "completeness", "thorough", "coverage", "comprehensive"],
    },
    {
      label: "Better reasoning guidance",
      keywords: ["reasoning", "step-by-step", "scaffold", "systematic", "guidance"],
    },
    {
      label: "Better domain alignment",
      keywords: ["domain", "clinical", "legal", "education", "fit", "usable", "context"],
    },
    {
      label: "Stronger expert framing",
      keywords: ["persona", "role", "expert", "authority"],
    },
  ];

  const corpus = `${peerReviews.map((review) => review.evaluation).join(" ")} ${chairmanRationale}`.toLowerCase();
  const ranked = buckets
    .map((bucket) => ({
      label: bucket.label,
      score: bucket.keywords.reduce((count, keyword) => count + (corpus.includes(keyword) ? 1 : 0), 0),
    }))
    .filter((bucket) => bucket.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((bucket) => bucket.label);

  return ranked.length ? ranked : ["Balanced structure and clarity", "More explicit instructions", "Stronger prompt reliability"];
}

function getPromptImprovements(rawQuery: string, optimisedPrompt: string, intent: Record<string, any>) {
  const raw = rawQuery.toLowerCase();
  const optimised = optimisedPrompt.toLowerCase();
  const improvements: string[] = [];

  const maybeAdd = (condition: boolean, label: string) => {
    if (condition && !improvements.includes(label)) improvements.push(label);
  };

  maybeAdd(
    ["format", "section", "sections", "table", "json", "bullet", "heading", "template"].some((term) => optimised.includes(term) && !raw.includes(term)),
    "Added explicit output structure",
  );
  maybeAdd(
    ["you are", "act as", "role", "expert"].some((term) => optimised.includes(term) && !raw.includes(term)),
    "Introduced clearer model framing",
  );
  maybeAdd(
    ["constraint", "must", "do not", "only use", "requirements", "follow these"].some((term) => optimised.includes(term) && !raw.includes(term)),
    "Added stronger constraints",
  );
  maybeAdd(
    ["step 1", "step-by-step", "reason through", "first,", "then,"].some((term) => optimised.includes(term) && !raw.includes(term)),
    "Improved reasoning scaffolding",
  );
  maybeAdd(
    ["follow-up", "return", "next steps", "questions", "assumption"].some((term) => optimised.includes(term) && !raw.includes(term)),
    "Made missing-information handling explicit",
  );
  maybeAdd(
    Boolean(intent?.topic_domain || intent?.format_domain),
    `Aligned the prompt to ${[intent?.topic_domain, intent?.format_domain].filter(Boolean).join(" / ")}`,
  );

  return improvements.slice(0, 6);
}

function similarityScore(orderA: string[], orderB: string[]) {
  const items = orderA.filter((item) => orderB.includes(item));
  if (items.length < 2) return 1;
  const positionB = Object.fromEntries(orderB.map((item, index) => [item, index]));
  let concordant = 0;
  let total = 0;
  for (let i = 0; i < items.length; i += 1) {
    for (let j = i + 1; j < items.length; j += 1) {
      total += 1;
      if ((positionB[items[i]] ?? 0) < (positionB[items[j]] ?? 0)) concordant += 1;
    }
  }
  return total ? concordant / total : 1;
}

function tokenDistance(a: string, b: string) {
  const aTokens = new Set((a.toLowerCase().match(/[a-z0-9]+/g) || []));
  const bTokens = new Set((b.toLowerCase().match(/[a-z0-9]+/g) || []));
  const union = new Set(Array.from(aTokens).concat(Array.from(bTokens)));
  if (union.size === 0) return 0;
  const overlap = Array.from(aTokens).filter((token) => bTokens.has(token)).length;
  return 1 - overlap / union.size;
}

function changeRatio(before: string, after: string) {
  const diff = buildLineDiff(before, after);
  const summary = summarizeDiff(diff);
  const total = diff.length || 1;
  return (summary.added + summary.removed) / total;
}

function bucketLevel(value: number, high: number, medium: number) {
  if (value >= high) return "high";
  if (value >= medium) return "moderate";
  return "mixed";
}

function getLiveResearchInsights(
  pipelineState: PipelineState,
  rawQuery: string,
  optimisedPrompt: string,
  finalPrompt: string,
): ResearchInsights {
  const peerReviews = pipelineState.peer_reviews || [];
  const aggregateRankings = pipelineState.aggregate_rankings || [];
  const diagnostics = pipelineState.consensus_diagnostics || {};
  const winner = aggregateRankings[0];
  const winnerLabel = winner?.label || "";
  const reviewerCount = peerReviews.length;
  const firstPlaceSupport = reviewerCount
    ? peerReviews.filter((review) => review.parsed_ranking?.[0] === winnerLabel).length / reviewerCount
    : 0;

  const similarities: number[] = [];
  for (let i = 0; i < peerReviews.length; i += 1) {
    for (let j = i + 1; j < peerReviews.length; j += 1) {
      similarities.push(similarityScore(peerReviews[i].parsed_ranking || [], peerReviews[j].parsed_ranking || []));
    }
  }
  const reviewerAgreement = similarities.length
    ? similarities.reduce((sum, value) => sum + value, 0) / similarities.length
    : reviewerCount ? 1 : 0;
  const consensusStrength = reviewerCount ? (firstPlaceSupport + reviewerAgreement) / 2 : 0;

  const candidates = [pipelineState.candidate_a, pipelineState.candidate_b, pipelineState.candidate_c].filter(Boolean);
  const diversityScores: number[] = [];
  for (let i = 0; i < candidates.length; i += 1) {
    for (let j = i + 1; j < candidates.length; j += 1) {
      diversityScores.push(tokenDistance(candidates[i], candidates[j]));
    }
  }
  const rewriteDiversity = diversityScores.length
    ? diversityScores.reduce((sum, value) => sum + value, 0) / diversityScores.length
    : 0;
  const humanEditShift = changeRatio(optimisedPrompt, finalPrompt);
  const consensusResponse =
    humanEditShift === 0
      ? "accepted"
      : humanEditShift >= 0.35
        ? "overrode"
        : "refined";
  return {
    winner_candidate: winner?.candidate || winnerLabel || null,
    reviewer_count: reviewerCount,
    winner_first_place_support_pct: diagnostics.first_place_support_pct ?? Number((firstPlaceSupport * 100).toFixed(1)),
    reviewer_agreement_pct: diagnostics.reviewer_agreement_pct ?? Number((reviewerAgreement * 100).toFixed(1)),
    consensus_strength_pct: diagnostics.consensus_strength_pct ?? Number((consensusStrength * 100).toFixed(1)),
    consensus_label: diagnostics.consensus_label ?? bucketLevel(consensusStrength, 0.8, 0.55),
    winner_margin: diagnostics.winner_margin ?? Number((((aggregateRankings[1]?.average_rank || 0) - (winner?.average_rank || 0))).toFixed(2)),
    rewrite_diversity_pct: Number((rewriteDiversity * 100).toFixed(1)),
    diversity_label: bucketLevel(rewriteDiversity, 0.6, 0.35),
    optimization_shift_pct: Number((changeRatio(rawQuery, optimisedPrompt) * 100).toFixed(1)),
    human_edit_shift_pct: Number((humanEditShift * 100).toFixed(1)),
    human_edit_level: humanEditShift === 0 ? "none" : humanEditShift < 0.2 ? "light" : "substantial",
    accepted_without_edit: consensusResponse === "accepted" && humanEditShift === 0,
    consensus_response: consensusResponse,
    accepted_with_refinement: consensusResponse === "refined",
    overrode_consensus: consensusResponse === "overrode",
    compare_mode: false,
    response_length_delta_pct: null,
    safety_acknowledged: false,
    intervention_labels: [],
  };
}

function describeConsensusResponse(response?: string): string {
  if (response === "accepted") return "Accepted";
  if (response === "refined") return "Refined";
  if (response === "overrode") return "Overrode";
  return "Pending";
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
  targetModels,
}: {
  onSubmit: (q: string, domain: string, model: string, demo: boolean, compare: boolean) => void;
  recentSessions: SessionSummary[];
  analytics: SessionAnalytics | null;
  targetModels: string[];
}) {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("General");
  const [model, setModel] = useState(targetModels[0] || "tencent/hy3-preview:free");
  const [demo, setDemo] = useState(true);
  const [compareMode, setCompareMode] = useState(true);
  const [showHow, setShowHow] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  React.useEffect(() => {
    setSelectedSessionId((current) => (
      current && recentSessions.some((session) => session.session_id === current)
        ? current
        : null
    ));
  }, [recentSessions]);

  React.useEffect(() => {
    if (targetModels.length > 0 && !targetModels.includes(model)) {
      setModel(targetModels[0]);
    }
  }, [targetModels, model]);

  const loadExample = (ex: typeof EXAMPLE_QUERIES[number]) => { setQuery(ex.query); setDomain(ex.domain); };
  const selectedSession = recentSessions.find((session) => session.session_id === selectedSessionId) || null;
  const selectedWinner = selectedSession?.aggregate_rankings?.[0];
  const selectedDiff = selectedSession
    ? buildLineDiff(selectedSession.optimised_prompt || "", selectedSession.final_prompt || "")
    : [];
  const selectedDiffSummary = summarizeDiff(selectedDiff);
  const selectedInsights = selectedSession?.research_insights || null;

  return (
    <div className={`input-workspace sidebar-${sidebarOpen ? "open" : "closed"}`}>
      <aside className={`input-sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-rail">
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label={sidebarOpen ? "Collapse study sidebar" : "Expand study sidebar"}
            title={sidebarOpen ? "Collapse study sidebar" : "Expand study sidebar"}
          >
            {sidebarOpen ? "\u2039" : "\u203A"}
          </button>
          {!sidebarOpen && (
            <div className="sidebar-rail-meta">
              <span className="sidebar-rail-count">{recentSessions.length}</span>
              <span className="sidebar-rail-label">Runs</span>
            </div>
          )}
        </div>

        {sidebarOpen && (
          <div className="input-sidebar-inner">
            <div className="sidebar-panel">
              <div className="section-label">Study Analytics</div>
              <div className="sidebar-metrics">
                <div className="sidebar-metric-row">
                  <span>Saved sessions</span>
                  <strong>{analytics?.total_sessions ?? 0}</strong>
                </div>
                <div className="sidebar-metric-row">
                  <span>Avg quality</span>
                  <strong>{analytics?.avg_quality?.toFixed(1) ?? "0.0"}</strong>
                </div>
                <div className="sidebar-metric-row">
                  <span>Avg trust</span>
                  <strong>{analytics?.avg_trust?.toFixed(1) ?? "0.0"}</strong>
                </div>
                <div className="sidebar-metric-row">
                  <span>Accepted as-is</span>
                  <strong>{analytics ? `${analytics.acceptance_rate}%` : "0%"}</strong>
                </div>
              </div>
              <div className="analytics-footnote" style={{ marginBottom: 0 }}>
                Consensus {analytics?.avg_consensus_strength?.toFixed(1) ?? "0.0"}% • Acceptance {analytics ? `${analytics.acceptance_rate}%` : "0%"} • Top domain {analytics?.top_domain || "n/a"}
              </div>
            </div>

            <div className="sidebar-panel sidebar-panel-history">
              <div className="section-label">Session History</div>
              <div className="session-panel">
                <div className="session-panel-header">
                  <div>
                    <div className="session-panel-title">Study Runs</div>
                    <div className="session-panel-subtitle">Open a saved run to inspect its consensus process in the main panel.</div>
                  </div>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/json`} target="_blank" rel="noreferrer">
                      JSON
                    </a>
                    <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/csv`} target="_blank" rel="noreferrer">
                      CSV
                    </a>
                    <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/preferences`} target="_blank" rel="noreferrer">
                      JSONL
                    </a>
                  </div>
                </div>
                {recentSessions.length === 0 ? (
                  <div className="session-empty">No saved sessions yet. Complete a run and submit feedback to build your study log.</div>
                ) : (
                  <div className="session-list">
                    {recentSessions.slice(0, 10).map((session) => {
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
              </div>
            </div>
          </div>
        )}
      </aside>

      <div className="input-main">
        {selectedSession ? (
          <>
            <div className="main-panel-header">
              <div>
                <div className="section-label">Session Details</div>
                <h1 className="page-title" style={{ marginBottom: 8 }}>{selectedSession.raw_query || "Untitled session"}</h1>
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
                <button type="button" className="btn btn-secondary" onClick={() => setSelectedSessionId(null)}>
                  New prompt
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => {
                    const restoredDomain = selectedSession.domain
                      ? selectedSession.domain.charAt(0).toUpperCase() + selectedSession.domain.slice(1)
                      : "General";
                    setQuery(selectedSession.raw_query || "");
                    setDomain(restoredDomain);
                    if (selectedSession.target_model) setModel(selectedSession.target_model);
                    setCompareMode(Boolean(selectedSession.compare_mode));
                    setSelectedSessionId(null);
                  }}
                >
                  Reuse query
                </button>
              </div>
            </div>

            <div className="session-detail-grid">
              <div className="session-stat">
                <span className="session-stat-label">Quality</span>
                <span className="session-stat-value">{selectedSession.quality ?? "-"}</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-label">Trust</span>
                <span className="session-stat-value">{selectedSession.trust ?? "-"}</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-label">Improvement</span>
                <span className="session-stat-value">{selectedSession.improvement ?? "-"}</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-label">Control</span>
                <span className="session-stat-value">{selectedSession.control ?? "-"}</span>
              </div>
            </div>

            {selectedInsights && (
              <div className="research-strip" style={{ marginTop: 18 }}>
                <div className="research-metric">
                  <span className="research-label">Consensus strength</span>
                  <strong>{selectedInsights.consensus_strength_pct}%</strong>
                  <span>{selectedInsights.consensus_label}</span>
                </div>
                <div className="research-metric">
                  <span className="research-label">Reviewer agreement</span>
                  <strong>{selectedInsights.reviewer_agreement_pct}%</strong>
                  <span>{selectedInsights.winner_first_place_support_pct}% first-place support</span>
                </div>
                <div className="research-metric">
                  <span className="research-label">Rewrite diversity</span>
                  <strong>{selectedInsights.rewrite_diversity_pct}%</strong>
                  <span>{selectedInsights.diversity_label}</span>
                </div>
                <div className="research-metric">
                  <span className="research-label">Human intervention</span>
                  <strong>{selectedInsights.human_edit_shift_pct}%</strong>
                  <span>{selectedInsights.human_edit_level}</span>
                </div>
                <div className="research-metric">
                  <span className="research-label">Human response</span>
                  <strong>{describeConsensusResponse(selectedInsights.consensus_response)}</strong>
                  <span>
                    {selectedInsights.accepted_without_edit
                      ? "accepted directly"
                      : selectedInsights.overrode_consensus
                        ? "consensus replaced"
                        : "consensus refined"}
                  </span>
                </div>
              </div>
            )}

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

            <div className="section-label" style={{ marginTop: 24 }}>Prompt History</div>
            <div className="session-history-list">
              <div className="session-history-item">
                <div className="session-history-marker">01</div>
                <div className="session-history-content">
                  <div className="session-history-header">
                    <span className="session-history-title">Raw Query</span>
                  </div>
                  <div className="session-history-body">{safeSnippet(selectedSession.raw_query)}</div>
                </div>
              </div>
              <div className="session-history-item session-history-item-active">
                <div className="session-history-marker">02</div>
                <div className="session-history-content">
                  <div className="session-history-header">
                    <span className="session-history-title">Council Output</span>
                    <span className="session-history-note">{selectedWinner?.candidate || "No winner recorded"}</span>
                  </div>
                  <div className="session-history-body">{safeSnippet(selectedSession.optimised_prompt)}</div>
                </div>
              </div>
              <div className={`session-history-item ${selectedSession.optimised_prompt !== selectedSession.final_prompt ? "session-history-item-human" : ""}`}>
                <div className="session-history-marker">03</div>
                <div className="session-history-content">
                  <div className="session-history-header">
                    <span className="session-history-title">Final Prompt</span>
                    <span className="session-history-note">
                      {selectedDiffSummary.added || selectedDiffSummary.removed
                        ? `+${selectedDiffSummary.added} / -${selectedDiffSummary.removed} lines`
                        : "No user edits"}
                    </span>
                  </div>
                  <div className="session-history-body">{safeSnippet(selectedSession.final_prompt)}</div>
                </div>
              </div>
            </div>

            <div className="session-detail-sections">
              <div className="session-detail-section">
                <div className="section-label">Council Summary</div>
                <div className="rationale-box session-inline-note" style={{ marginBottom: 14 }}>
                  {selectedWinner
                    ? `${selectedWinner.candidate || selectedWinner.label} won with average rank ${selectedWinner.average_rank} across ${selectedWinner.votes} reviews.`
                    : "No aggregate ranking recorded for this run."}
                </div>
                {selectedSession.consensus_diagnostics && (
                  <div className="consensus-signal-strip">
                    <div className={`consensus-signal consensus-${selectedSession.consensus_diagnostics.consensus_label || "unknown"}`}>
                      <span className="consensus-signal-label">Council confidence</span>
                      <strong>{selectedSession.consensus_diagnostics.consensus_label || "unknown"}</strong>
                      <span>{selectedSession.consensus_diagnostics.consensus_strength_pct ?? 0}% strength</span>
                    </div>
                    <div className="consensus-signal">
                      <span className="consensus-signal-label">Reviewer agreement</span>
                      <strong>{selectedSession.consensus_diagnostics.reviewer_agreement_pct ?? 0}%</strong>
                      <span>{selectedSession.consensus_diagnostics.first_place_support_pct ?? 0}% first-place support</span>
                    </div>
                    <div className="consensus-signal">
                      <span className="consensus-signal-label">Human outcome</span>
                      <strong>{describeConsensusResponse(selectedInsights?.consensus_response)}</strong>
                      <span>
                        {selectedSession.consensus_diagnostics.needs_human_review
                          ? "review was especially important here"
                          : "council outcome was comparatively stable"}
                      </span>
                    </div>
                  </div>
                )}
                {selectedSession.chairman?.rationale && (
                  <div className="session-detail-textblock">
                    {selectedSession.chairman.rationale}
                  </div>
                )}
                {selectedInsights && (
                  <div className="session-inline-note" style={{ marginTop: 14 }}>
                    {selectedInsights.accepted_without_edit
                      ? "The participant approved the council output without editing it, which is a direct trust-and-adoption signal."
                      : selectedInsights.overrode_consensus
                        ? "The participant overrode the council outcome, which is especially valuable evidence about where model consensus diverged from human intent."
                        : `The participant refined the council output at a ${selectedInsights.human_edit_level} level, which preserves the consensus trace while showing where human tailoring still mattered.`}
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
                <div className="session-output-stack">
                  {selectedSession.compare_mode && (
                    <div className="cmp-card session-output-card">
                      <div className="cmp-card-header">
                        <span className="badge badge-original">Baseline</span>
                        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Raw query response</span>
                      </div>
                      <div className="cmp-card-body">{safeSnippet(selectedSession.baseline_response)}</div>
                    </div>
                  )}
                  <div className={`cmp-card session-output-card ${selectedSession.compare_mode ? "optimised" : ""}`}>
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

            {!!selectedSession.intervention_labels?.length && (
              <div style={{ marginTop: 18 }}>
                <div className="section-label">Intervention Notes</div>
                <div className="consensus-reasons">
                  {selectedSession.intervention_labels.map((label) => (
                    <span key={label} className="consensus-reason-chip">{label}</span>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="composer-hero">
              <div className="section-label">Human-Centred Prompt Optimisation</div>
              <h1 className="page-title">Write naturally. Let the <span className="accent">council</span> optimize. You stay in control.</h1>
              <p className="page-subtitle">
                ConsensusPrompt rewrites your prompt through three independent strategies, asks a council to rank them anonymously,
                and returns the strongest version for human review before anything is executed.
              </p>
            </div>

            <div className="composer-shell">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
                <label className="field-label">Your Query</label>
                <span style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
                  {query.length > 0 ? `${query.split(/\s+/).filter(Boolean).length} words` : ""}
                </span>
              </div>
              <textarea className="textarea-field composer-textarea" value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe what you need in plain language. The system will optimize structure, constraints, and domain framing for you." rows={8} />

              <div className="composer-actions">
                <button className="btn btn-primary" disabled={!query.trim()}
                  onClick={() => onSubmit(query.trim(), domain.toLowerCase(), model, demo, compareMode)}>
                  Optimise prompt
                </button>
                <button className="btn btn-secondary" onClick={() => setShowOptions(!showOptions)}>
                  {showOptions ? "Hide options" : "Options"}
                </button>
              </div>

              {showOptions && (
                <div className="composer-options">
                  <div>
                    <label className="field-label">Domain</label>
                    <select className="select-field" value={domain} onChange={(e) => setDomain(e.target.value)}>
                      {["General", "Healthcare", "Education", "Legal", "Research", "Business", "Technology"].map((d) => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="field-label">Target Model</label>
                    <select className="select-field" value={model} onChange={(e) => setModel(e.target.value)}>
                      {targetModels.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                  <label className="option-toggle">
                    <input type="checkbox" checked={demo} onChange={() => setDemo(!demo)} style={{ accentColor: "var(--accent)" }} />
                    <span>Demo mode</span>
                    <small>Use fixture data instead of live inference.</small>
                  </label>
                  <label className="option-toggle">
                    <input type="checkbox" checked={compareMode} onChange={() => setCompareMode(!compareMode)} style={{ accentColor: "var(--accent)" }} />
                    <span>Baseline comparison</span>
                    <small>Compare the raw query response against the optimized-prompt response.</small>
                  </label>
                </div>
              )}
            </div>

            <div style={{ marginTop: 20 }}>
              <div className="section-label">Example Queries</div>
              <div className="example-pill-row">
                {EXAMPLE_QUERIES.map((ex) => (
                  <button key={ex.label} className="example-pill" onClick={() => loadExample(ex)}>{ex.label}</button>
                ))}
              </div>
            </div>

            <button className="collapsible-header composer-collapsible" onClick={() => setShowHow(!showHow)} style={{ marginBottom: showHow ? 0 : 20 }}>
              <span>How does this work?</span>
              <span style={{ transition: "transform 0.2s", transform: showHow ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
            </button>
            {showHow && (
              <div style={{ marginBottom: 28 }}>
                <div className="how-steps" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
                  <div className="how-step">
                    <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>01</div>
                    <div className="how-step-title">Write naturally</div>
                    <div className="how-step-desc">Describe what you need in your own words. No prompt-engineering knowledge required.</div>
                  </div>
                  <div className="how-step">
                    <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>02</div>
                    <div className="how-step-title">Agents rewrite</div>
                    <div className="how-step-desc">Three agents independently restructure your prompt using different strategies.</div>
                  </div>
                  <div className="how-step">
                    <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>03</div>
                    <div className="how-step-title">Council reviews</div>
                    <div className="how-step-desc">Each model reviews the others anonymously, preventing bias. Rankings are aggregated and the best is synthesised.</div>
                  </div>
                  <div className="how-step">
                    <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)" }}>04</div>
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
          </>
        )}
      </div>
    </div>
  );
}

/* ── Stage 2 — Processing ── */
function StageProcessing({
  rawQuery, onComplete, domain, demoMode, retryToken, onRetry,
}: {
  rawQuery: string; domain: string; demoMode: boolean;
  retryToken: number;
  onComplete: (state: PipelineState) => void;
  onRetry: () => void;
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
          const detail = err?.message || "The live progress connection was interrupted after work started.";
          setError(detail);
          setStatusMsg("Streaming interrupted");
        }
      }
    };

    streamPipeline();
    return () => controller.abort();
  }, [rawQuery, domain, demoMode, onComplete, retryToken]);

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
        }}>
          <div style={{ marginBottom: 12 }}>{error}</div>
          <button className="btn btn-secondary" onClick={onRetry}>
            Retry safely
          </button>
        </div>
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
  const { peer_reviews, aggregate_rankings, label_map, chairman, candidate_a, candidate_b, candidate_c, perspectives, consensus_diagnostics } = pipelineState;
  const [revealedReviewers, setRevealedReviewers] = useState(0);
  const [showWinner, setShowWinner] = useState(false);
  const [activeReview, setActiveReview] = useState(-1);
  const [activeTab, setActiveTab] = useState<string>("a");
  const [showScores, setShowScores] = useState(false);
  const [showPeerReviews, setShowPeerReviews] = useState(false);
  const [showRankings, setShowRankings] = useState(false);
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
  const consensusReasons = React.useMemo(
    () => getConsensusReasons(peer_reviews, chairman.rationale || ""),
    [peer_reviews, chairman.rationale],
  );

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

      {/* Aggregate rankings with animated bars */}
      {showRankings && (
        <div style={{ marginBottom: 28 }} className="council-fade-in">
          <div className="section-label">Aggregate Consensus</div>
          <div className="council-rankings-shell">
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
            Average rank: {winner.average_rank} across {winner.votes} reviewers
            {consensus_diagnostics?.is_unanimous_winner ? " — unanimous first place" : ""}
          </div>
          {consensus_diagnostics && (
            <div className="consensus-signal-strip">
              <div className={`consensus-signal consensus-${consensus_diagnostics.consensus_label || "unknown"}`}>
                <span className="consensus-signal-label">Consensus strength</span>
                <strong>{consensus_diagnostics.consensus_strength_pct ?? 0}%</strong>
                <span>{consensus_diagnostics.consensus_label || "unknown"}</span>
              </div>
              <div className="consensus-signal">
                <span className="consensus-signal-label">Reviewer agreement</span>
                <strong>{consensus_diagnostics.reviewer_agreement_pct ?? 0}%</strong>
                <span>{consensus_diagnostics.first_place_support_pct ?? 0}% first-place support</span>
              </div>
              <div className={`consensus-signal ${consensus_diagnostics.needs_human_review ? "consensus-review-needed" : ""}`}>
                <span className="consensus-signal-label">Human review</span>
                <strong>{consensus_diagnostics.needs_human_review ? "Recommended" : "Routine"}</strong>
                <span>
                  {consensus_diagnostics.needs_human_review
                    ? "the council was not fully aligned"
                    : "the council outcome was relatively stable"}
                </span>
              </div>
            </div>
          )}
          <div className="consensus-reasons">
            {consensusReasons.map((reason) => (
              <span key={reason} className="consensus-reason-chip">{reason}</span>
            ))}
          </div>
          {chairman.rationale && (
            <div className="council-winner-rationale">
              <strong>Chairman synthesis:</strong> {chairman.rationale}
            </div>
          )}
        </div>
      )}

      {revealedReviewers > 0 && (
        <>
          <button className="collapsible-header council-collapse" onClick={() => setShowPeerReviews(!showPeerReviews)}
            style={{ marginBottom: showPeerReviews ? 0 : 20 }}>
            <span>Peer reviews</span>
            <span style={{ transition: "transform 0.2s", transform: showPeerReviews ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
          </button>
          {showPeerReviews && (
            <div className="council-detail-block">
              <div className="council-grid">
                {peer_reviews.slice(0, revealedReviewers).map((review, i) => (
                  <div
                    key={i}
                    className={`council-card ${i === activeReview ? "council-card-active" : ""}`}
                    onClick={() => setActiveReview(activeReview === i ? -1 : i)}
                    style={{ animationDelay: `${i * 0.1}s`, cursor: "pointer", display: "flex", flexDirection: "column" }}
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

              {activeReview !== -1 && (
                <div className="council-review-detail">
                  <div className="section-label" style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                    <span>{peer_reviews[activeReview]?.reviewer} — Full Evaluation</span>
                    <span className="council-inline-kicker">
                      Top Pick: {peer_reviews[activeReview]?.parsed_ranking[0]} = {label_map[peer_reviews[activeReview]?.parsed_ranking[0]] || peer_reviews[activeReview]?.parsed_ranking[0]}
                    </span>
                  </div>
                  <div className="council-review-body">
                    {peer_reviews[activeReview]?.evaluation}
                  </div>
                </div>
              )}
            </div>
          )}

          <button className="collapsible-header council-collapse" onClick={() => setShowScores(!showScores)}
            style={{ marginBottom: showScores ? 0 : 28 }}>
            <span>Candidate prompts</span>
            <span style={{ transition: "transform 0.2s", transform: showScores ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
          </button>
          {showScores && (
            <div className="council-detail-block council-detail-block-last">
              <div className="tabs-header">
                <button className={`tab-btn ${activeTab === "a" ? "active" : ""}`} onClick={() => setActiveTab("a")}>
                  A: {perspectives?.["Candidate A"] ? (perspectives["Candidate A"].length > 25 ? `${perspectives["Candidate A"].slice(0, 25)}...` : perspectives["Candidate A"]) : "Dynamic Strategy A"}
                </button>
                <button className={`tab-btn ${activeTab === "b" ? "active" : ""}`} onClick={() => setActiveTab("b")}>
                  B: {perspectives?.["Candidate B"] ? (perspectives["Candidate B"].length > 25 ? `${perspectives["Candidate B"].slice(0, 25)}...` : perspectives["Candidate B"]) : "Dynamic Strategy B"}
                </button>
                <button className={`tab-btn ${activeTab === "c" ? "active" : ""}`} onClick={() => setActiveTab("c")}>
                  C: {perspectives?.["Candidate C"] ? (perspectives["Candidate C"].length > 25 ? `${perspectives["Candidate C"].slice(0, 25)}...` : perspectives["Candidate C"]) : "Dynamic Strategy C"}
                </button>
              </div>
              <div className="candidate-text-panel">
                {activeTab === "a" ? candidate_a : activeTab === "b" ? candidate_b : candidate_c}
              </div>
            </div>
          )}
        </>
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
  const { raw_query, intent, optimised_prompt, aggregate_rankings, label_map, chairman, perspectives, consensus_diagnostics } = pipelineState;
  const [editedPrompt, setEditedPrompt] = useState(optimised_prompt);
  const [copied, setCopied] = useState(false);
  const [showReferences, setShowReferences] = useState(false);
  const [showDiffs, setShowDiffs] = useState(false);

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
  const promptImprovements = React.useMemo(
    () => getPromptImprovements(raw_query, optimised_prompt, intent || {}),
    [raw_query, optimised_prompt, intent],
  );
  const consensusReasons = React.useMemo(
    () => getConsensusReasons(pipelineState.peer_reviews || [], chairman.rationale || ""),
    [pipelineState.peer_reviews, chairman.rationale],
  );
  const researchInsights = React.useMemo(
    () => getLiveResearchInsights(pipelineState, raw_query, optimised_prompt, editedPrompt),
    [pipelineState, raw_query, optimised_prompt, editedPrompt],
  );

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
          {consensus_diagnostics?.consensus_label && (
            <span className={`badge badge-original consensus-badge consensus-${consensus_diagnostics.consensus_label}`} style={{ fontSize: 11 }}>
              Consensus {consensus_diagnostics.consensus_label} · {consensus_diagnostics.consensus_strength_pct ?? 0}%
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

      <div className="review-editor-shell">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 8 }}>
          <div className="section-label">Final Prompt</div>
          <button onClick={handleCopy} className="btn btn-secondary" style={{ fontSize: 11, padding: "4px 12px" }}>
            {copied ? "Copied" : "Copy to clipboard"}
          </button>
        </div>
        <textarea className="textarea-field review-editor" value={editedPrompt} onChange={(e) => setEditedPrompt(e.target.value)} rows={11} />
        <div className="review-editor-meta">
          <span>{editedPrompt.split(/\s+/).filter(Boolean).length} words</span>
          <span>{hasUserEdits ? `Edited: +${optimisedToFinalSummary.added} / -${optimisedToFinalSummary.removed} lines` : "No user edits yet"}</span>
        </div>
      </div>

      <div className="review-actions">
        <button className="btn btn-primary" onClick={() => onApprove(editedPrompt)}>Approve and execute</button>
        <button className="btn btn-danger" onClick={onStartOver}>Discard and start over</button>
      </div>

      <div className="section-label">Optimization Summary</div>
      <div className="summary-grid">
        <div className="summary-block">
          <div className="summary-title">What the optimization added</div>
          <div className="summary-list">
            {promptImprovements.map((item) => (
              <div key={item} className="summary-item">
                <span className="summary-dot" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="summary-block summary-block-human">
          <div className="summary-title">Why the council selected it</div>
          <div className="summary-list">
            {consensusReasons.map((item) => (
              <div key={item} className="summary-item">
                <span className="summary-dot" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="section-label">Research Lens</div>
      <div className="research-strip" style={{ marginBottom: 22 }}>
        <div className="research-metric">
          <span className="research-label">Consensus strength</span>
          <strong>{researchInsights.consensus_strength_pct}%</strong>
          <span>{researchInsights.consensus_label}</span>
        </div>
        <div className="research-metric">
          <span className="research-label">Reviewer agreement</span>
          <strong>{researchInsights.reviewer_agreement_pct}%</strong>
          <span>{researchInsights.winner_first_place_support_pct}% first-place support</span>
        </div>
        <div className="research-metric">
          <span className="research-label">Rewrite diversity</span>
          <strong>{researchInsights.rewrite_diversity_pct}%</strong>
          <span>{researchInsights.diversity_label}</span>
        </div>
        {researchInsights.candidate_pattern_convergence && researchInsights.candidate_pattern_convergence !== "unknown" && (
          <div className="research-metric">
            <span className="research-label">Pattern convergence</span>
            <strong>{researchInsights.candidate_pattern_convergence}</strong>
            <span>
              token {researchInsights.candidate_token_diversity?.toFixed(0) ?? 0}%
              {" / "}
              struct {researchInsights.candidate_structural_diversity?.toFixed(0) ?? 0}%
            </span>
          </div>
        )}
        <div className="research-metric">
          <span className="research-label">Your intervention</span>
          <strong>{researchInsights.human_edit_shift_pct}%</strong>
          <span>{researchInsights.human_edit_level}</span>
        </div>
        <div className="research-metric">
          <span className="research-label">Consensus response</span>
          <strong>{describeConsensusResponse(researchInsights.consensus_response)}</strong>
          <span>
            {researchInsights.accepted_without_edit
              ? "direct adoption"
              : researchInsights.overrode_consensus
                ? "human override"
                : "human refinement"}
          </span>
        </div>
      </div>
      <div className="session-inline-note" style={{ marginBottom: 22 }}>
        {researchInsights.accepted_without_edit
          ? "If you approve this prompt as-is, the session records a pure council adoption signal: consensus formed and the human accepted it without revision."
          : researchInsights.overrode_consensus
            ? "Your edits currently read like an override rather than a light adjustment, which is especially useful HCAI evidence about where human intent diverged from council consensus."
            : "Your edits are part of the core HCAI evidence here: they show where consensus still needed human correction, tailoring, or domain-specific refinement."}
      </div>
       {consensus_diagnostics?.needs_human_review && (
        <div className="session-inline-note" style={{ marginBottom: 22 }}>
          The council agreement on this run was not especially strong, so your review matters more than usual here. This is the right place to confirm whether the winner preserved your intent or whether a different strategy should have prevailed.
        </div>
      )}


      <button className="collapsible-header review-collapse" onClick={() => setShowReferences(!showReferences)} style={{ marginBottom: showReferences ? 0 : 18 }}>
        <span>View original prompt and council output</span>
        <span style={{ transition: "transform 0.2s", transform: showReferences ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
      </button>
      {showReferences && (
        <div className="review-reference-block">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div className="section-label">Prompt Comparison</div>
            <span style={{ fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-dim)" }}>
              {origWords} words {"\u2192"} {optWords} words (+{Math.round(((optWords - origWords) / Math.max(origWords, 1)) * 100)}%)
            </span>
          </div>
          <div className="comparison-grid review-reference-grid">
            <div className="cmp-card">
              <div className="cmp-card-header"><span className="badge badge-original">Original</span></div>
              <div className="cmp-card-body">{raw_query}</div>
            </div>
            <div className="cmp-card optimised">
              <div className="cmp-card-header"><span className="badge badge-optimised">Council Output</span></div>
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
        </div>
      )}

      <button className="collapsible-header review-collapse" onClick={() => setShowDiffs(!showDiffs)} style={{ marginBottom: showDiffs ? 0 : 20 }}>
        <span>Inspect detailed diffs</span>
        <span style={{ transition: "transform 0.2s", transform: showDiffs ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
      </button>
      {showDiffs && (
        <div className="review-reference-block">
          <div className="section-label">What Changed</div>
          <div className="comparison-grid review-reference-grid" style={{ marginBottom: 0 }}>
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
        </div>
      )}
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
      if (!res.ok) {
        let detail = `Status ${res.status}`;
        try {
          const data = await res.json();
          detail = data?.detail || data?.message || detail;
        } catch {
          try {
            detail = await res.text();
          } catch {}
        }
        throw new Error(detail);
      }
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
    } catch (err: any) {
      const errorText = err?.message || "Could not reach the backend. Verify the server is running on port 8000.";
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
  const [interventionLabels, setInterventionLabels] = useState<string[]>([]);
  const labels: Record<number, string> = { 1: "Poor", 2: "Fair", 3: "Neutral", 4: "Good", 5: "Excellent" };
  const interventionOptions = [
    "Accepted council output as-is",
    "Edited for clarity or structure",
    "Edited for domain specificity",
    "Edited to restore my original intent",
    "Edited for caution or safety",
    "Preferred the baseline/raw condition more",
  ];

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
          intervention_labels: interventionLabels,
          intent: pipelineState?.intent || {},
          candidate_a: pipelineState?.candidate_a || "",
          candidate_b: pipelineState?.candidate_b || "",
          candidate_c: pipelineState?.candidate_c || "",
          peer_reviews: pipelineState?.peer_reviews || [],
          aggregate_rankings: pipelineState?.aggregate_rankings || [],
          label_map: pipelineState?.label_map || {},
          consensus_diagnostics: pipelineState?.consensus_diagnostics || {},
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
    const wasEdited = optimisedPrompt !== finalPrompt;
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

        {/* Preference Signal Card (Lecture 8: DPO) */}
        <div className="preference-card" style={{ textAlign: "left", marginTop: 20, maxWidth: 560, marginLeft: "auto", marginRight: "auto" }}>
          <div className="preference-card-header">
            <span className="preference-card-title">Preference Signal Generated</span>
            <span className="preference-card-badge">{wasEdited ? "Edited pair" : "Accepted"}</span>
          </div>
          <div className="preference-card-body">
            {wasEdited
              ? "Your edits created a natural (chosen, rejected) preference pair: your final prompt is the chosen output, and the council\u2019s original is the rejected output. This data can be used for Direct Preference Optimization (DPO) alignment."
              : "You accepted the council output without editing, recording a positive adoption signal. This indicates the council\u2019s output aligned with your intent."}
          </div>
          {wasEdited && (
            <div className="preference-pair-preview">
              <div>
                <div className="preference-pair-label chosen">Chosen</div>
                <div className="preference-pair-text">{finalPrompt.slice(0, 80)}{finalPrompt.length > 80 ? "\u2026" : ""}</div>
              </div>
              <div className="preference-pair-arrow">{"\u2192"}</div>
              <div>
                <div className="preference-pair-label rejected">Rejected</div>
                <div className="preference-pair-text">{optimisedPrompt.slice(0, 80)}{optimisedPrompt.length > 80 ? "\u2026" : ""}</div>
              </div>
            </div>
          )}
        </div>

        <div style={{ marginTop: 18, display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/json`} target="_blank" rel="noreferrer">
            Download JSON
          </a>
          <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/csv`} target="_blank" rel="noreferrer">
            Download CSV
          </a>
          <a className="btn btn-secondary" href={`${API_URL}/api/sessions/export/preferences`} target="_blank" rel="noreferrer">
            Download Preferences (JSONL)
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
      <div className="section-label" style={{ marginTop: 18 }}>Intervention Notes</div>
      <div className="consensus-reasons" style={{ marginBottom: 14 }}>
        {interventionOptions.map((option) => {
          const active = interventionLabels.includes(option);
          return (
            <button
              key={option}
              type="button"
              className={`consensus-reason-chip ${active ? "active" : ""}`}
              onClick={() => setInterventionLabels((current) => (
                current.includes(option)
                  ? current.filter((item) => item !== option)
                  : [...current, option]
              ))}
            >
              {option}
            </button>
          );
        })}
      </div>
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
  const [targetModel, setTargetModel] = useState("tencent/hy3-preview:free");
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
  const [processingRetryToken, setProcessingRetryToken] = useState(0);
  const [targetModels, setTargetModels] = useState<string[]>([
    "openai/gpt-5.4-nano",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v3.2",
    "nvidia/nemotron-3-super-120b-a12b",
  ]);

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

  React.useEffect(() => {
    fetch(`${API_URL}/api/config`)
      .then((r) => r.json())
      .then((data) => {
        const models = data?.target_models;
        if (Array.isArray(models) && models.length > 0) {
          setTargetModels(models);
          setTargetModel((current) => (models.includes(current) ? current : models[0]));
        }
      })
      .catch(() => {});
  }, []);

  const handleInputSubmit = (q: string, d: string, m: string, demo: boolean, compare: boolean) => {
    setRawQuery(q); setDomain(d); setTargetModel(m); setDemoMode(demo); setCompareMode(compare); setProcessingRetryToken(0); setStage("processing");
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
      {stage === "input" && (
        <StageInput
          onSubmit={handleInputSubmit}
          recentSessions={recentSessions}
          analytics={analytics}
          targetModels={targetModels}
        />
      )}
      {stage === "processing" && (
        <StageProcessing
          rawQuery={rawQuery}
          domain={domain}
          demoMode={demoMode}
          retryToken={processingRetryToken}
          onRetry={() => setProcessingRetryToken((current) => current + 1)}
          onComplete={handleProcessingComplete}
        />
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
