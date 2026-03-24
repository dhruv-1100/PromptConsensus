"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
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

/* ── Header ── */
function Header({ theme, toggleTheme }: { theme: string; toggleTheme: () => void }) {
  return (
    <div className="brand-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <img src="/logo.png" alt="ConsensusPrompt Logo" style={{ width: 56, height: 56, borderRadius: 8 }} />
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
function StageInput({ onSubmit }: { onSubmit: (q: string, domain: string, model: string, demo: boolean) => void }) {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("General");
  const [model, setModel] = useState("gpt-4o");
  const [demo, setDemo] = useState(true);
  const [showHow, setShowHow] = useState(false);

  const loadExample = (ex: typeof EXAMPLE_QUERIES[number]) => { setQuery(ex.query); setDomain(ex.domain); };

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
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--amber)" }}>01</div>
              <div className="how-step-title">Write naturally</div>
              <div className="how-step-desc">Describe what you need in your own words. No prompt-engineering knowledge required.</div>
            </div>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--amber)" }}>02</div>
              <div className="how-step-title">Agents rewrite</div>
              <div className="how-step-desc">Three agents independently restructure your prompt using different strategies.</div>
            </div>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--amber)" }}>03</div>
              <div className="how-step-title">Council reviews</div>
              <div className="how-step-desc">Each model reviews the others anonymously, preventing bias. Rankings are aggregated and the best is synthesised.</div>
            </div>
            <div className="how-step">
              <div className="how-step-icon" style={{ fontSize: 18, fontWeight: 700, color: "var(--amber)" }}>04</div>
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
            <input type="checkbox" checked={demo} onChange={() => setDemo(!demo)} style={{ accentColor: "var(--amber)" }} />
            Demo mode
          </label>
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
          onClick={() => onSubmit(query.trim(), domain.toLowerCase(), model, demo)}>
          Optimise prompt
        </button>
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
  const [started, setStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const agents = [
    { code: "S1", name: "Intent Extractor", model: "gemma-3-1b-it", desc: "Parses domains, gaps, and constraints" },
    { code: "S2A", name: "Autonomous Prompt Engineer A", model: "gemma-3-1b-it", desc: "Dynamic prompt optimization strategy" },
    { code: "S2B", name: "Autonomous Prompt Engineer B", model: "gemma-3-1b-it", desc: "Dynamic prompt optimization strategy" },
    { code: "S2C", name: "Autonomous Prompt Engineer C", model: "gemma-3-1b-it", desc: "Dynamic prompt optimization strategy" },
    { code: "R1", name: "Peer Reviewer 1", model: "gemma-3-1b-it", desc: "Anonymised evaluation of all candidates" },
    { code: "R2", name: "Peer Reviewer 2", model: "gemma-3-1b-it", desc: "Anonymised evaluation of all candidates" },
    { code: "R3", name: "Peer Reviewer 3", model: "gemma-3-1b-it", desc: "Anonymised evaluation of all candidates" },
  ];

  React.useEffect(() => {
    if (started) return;
    setStarted(true);

    const simulate = async () => {
      // S1
      setProgress(8); setStatusMsg("Extracting intent from query");
      setAgentStates(["running", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"]);
      await new Promise((r) => setTimeout(r, 700));
      setProgress(18); setAgentStates(["done", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"]);
      setStatusMsg("Intent extraction complete");
      await new Promise((r) => setTimeout(r, 300));

      // S2 parallel
      setProgress(25); setStatusMsg("Rewriting with three parallel strategies");
      setAgentStates(["done", "running", "running", "running", "waiting", "waiting", "waiting"]);
      await new Promise((r) => setTimeout(r, 1100));
      setProgress(50); setAgentStates(["done", "done", "done", "done", "waiting", "waiting", "waiting"]);
      setStatusMsg("All rewriters finished");
      await new Promise((r) => setTimeout(r, 300));

      // Council peer review
      setProgress(55); setStatusMsg("Council: anonymised peer review in progress");
      setAgentStates(["done", "done", "done", "done", "running", "running", "running"]);
      await new Promise((r) => setTimeout(r, 1200));
      setProgress(80); setAgentStates(["done", "done", "done", "done", "done", "done", "done"]);
      setStatusMsg("Council review complete — aggregating rankings");
      await new Promise((r) => setTimeout(r, 400));

      setProgress(90); setStatusMsg("Chairman synthesising final prompt");
      await new Promise((r) => setTimeout(r, 600));

      // Call backend
      try {
        const res = await fetch(`${API_URL}/api/optimize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ raw_query: rawQuery, domain, demo_mode: demoMode }),
        });
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        const data = await res.json();
        setProgress(100);
        await new Promise((r) => setTimeout(r, 500));
        onComplete(data);
      } catch (err: any) {
        setError(`Could not reach the backend at ${API_URL}. Start the server with: uvicorn main:app --port 8000`);
      }
    };
    simulate();
  }, [started, rawQuery, domain, demoMode, onComplete]);

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
  const [activeReview, setActiveReview] = useState(0);
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
  const winnerAgent = winner ? label_map[winner.label] || winner.label : "";

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
            onClick={() => setActiveReview(i)}
            style={{ animationDelay: `${i * 0.1}s` }}
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
      {revealedReviewers > 0 && (
        <div style={{ marginTop: 16, marginBottom: 28 }}>
          <div className="section-label">
            {peer_reviews[activeReview]?.reviewer} — Full Evaluation
          </div>
          <div style={{
            fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1.8,
            color: "var(--text-secondary)", whiteSpace: "pre-wrap",
            background: "var(--bg-surface)", border: "1px solid var(--border)",
            borderRadius: "var(--radius)", padding: "16px 18px",
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
    aggregate_rankings, label_map, chairman } = pipelineState;
  const [editedPrompt, setEditedPrompt] = useState(optimised_prompt);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const ok = await copyToClipboard(editedPrompt);
    if (ok) { setCopied(true); setTimeout(() => setCopied(false), 2000); }
  };

  const origWords = raw_query.split(/\s+/).filter(Boolean).length;
  const optWords = optimised_prompt.split(/\s+/).filter(Boolean).length;
  const winner = aggregate_rankings.length > 0 ? aggregate_rankings[0] : null;

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
  finalPrompt, rawQuery, targetModel, demoMode, onRate,
}: {
  finalPrompt: string; rawQuery: string; targetModel: string; demoMode: boolean; onRate: () => void;
}) {
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showPrompt, setShowPrompt] = useState(false);
  const [copied, setCopied] = useState(false);

  React.useEffect(() => {
    const exec = async () => {
      try {
        const res = await fetch(`${API_URL}/api/execute`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ final_prompt: finalPrompt, target_model: targetModel, demo_mode: demoMode }),
        });
        if (!res.ok) throw new Error(`Status ${res.status}`);
        const data = await res.json();
        setResponse(data.response);
      } catch {
        setResponse("Could not reach the backend. Verify the server is running on port 8000.");
      }
      setLoading(false);
    };
    exec();
  }, [finalPrompt, targetModel, demoMode]);

  const ow = rawQuery.split(/\s+/).filter(Boolean).length;
  const fw = finalPrompt.split(/\s+/).filter(Boolean).length;
  const pct = Math.round(((fw - ow) / Math.max(ow, 1)) * 100);

  const handleCopy = async () => {
    if (response) {
      const ok = await copyToClipboard(response);
      if (ok) { setCopied(true); setTimeout(() => setCopied(false), 2000); }
    }
  };

  return (
    <>
      <h1 className="page-title">Result</h1>
      <p className="page-subtitle">
        Response generated by <span style={{ color: "var(--amber)", fontFamily: "var(--mono)", fontSize: 13 }}>{targetModel}</span> using your approved prompt.
      </p>

      <button className="collapsible-header" onClick={() => setShowPrompt(!showPrompt)} style={{ marginBottom: showPrompt ? 0 : 20 }}>
        <span>View submitted prompt</span>
        <span style={{ transition: "transform 0.2s", transform: showPrompt ? "rotate(180deg)" : "rotate(0)" }}>{"\u25BE"}</span>
      </button>
      {showPrompt && <div className="collapsible-body" style={{ marginBottom: 20 }}>{finalPrompt}</div>}

      {loading ? (
        <div className="loading-shimmer" style={{ height: 200, marginBottom: 24 }} />
      ) : (
        <div style={{ position: "relative" }}>
          <div className="response-card" dangerouslySetInnerHTML={{ __html: formatMarkdown(response || "") }} />
          <button onClick={handleCopy} className="btn btn-secondary"
            style={{ position: "absolute", top: 12, right: 12, fontSize: 11, padding: "4px 12px" }}>
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      )}

      <div className="metrics-strip">
        <div className="metric-cell"><div className="metric-value">{ow}</div><div className="metric-label">Original words</div></div>
        <div className="metric-cell"><div className="metric-value">{fw}</div><div className="metric-label">Optimised words</div></div>
        <div className="metric-cell"><div className="metric-value">+{pct}%</div><div className="metric-label">Specificity gain</div></div>
        <div className="metric-cell"><div className="metric-value">3</div><div className="metric-label">Agents used</div></div>
      </div>

      <button className="btn btn-primary" onClick={onRate}>Rate this response</button>
    </>
  );
}

/* ── Stage 6 — Feedback ── */
function StageFeedback({ onReset }: { onReset: () => void }) {
  const [submitted, setSubmitted] = useState(false);
  const [quality, setQuality] = useState(4);
  const [improvement, setImprovement] = useState(4);
  const [trust, setTrust] = useState(4);
  const [control, setControl] = useState(5);
  const [comment, setComment] = useState("");
  const labels: Record<number, string> = { 1: "Poor", 2: "Fair", 3: "Neutral", 4: "Good", 5: "Excellent" };

  const handleSubmit = async () => {
    try {
      await fetch(`${API_URL}/api/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quality, improvement, trust, control, text: comment }),
      });
    } catch {}
    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="success-state">
        <div className="success-icon">{"\u2713"}</div>
        <div className="success-title">Feedback recorded</div>
        <div className="success-sub">Your ratings contribute to ongoing evaluation of the council model.</div>
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
      <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--amber)", marginTop: 6 }}>{value}/5 — {labels[value]}</div>
    </div>
  );

  return (
    <>
      <h1 className="page-title">Feedback</h1>
      <p className="page-subtitle">Rate this session across four dimensions aligned with the System Usability Scale.</p>
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
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [stage, setStage] = useState<Stage>("input");
  const [rawQuery, setRawQuery] = useState("");
  const [domain, setDomain] = useState("general");
  const [targetModel, setTargetModel] = useState("gpt-4o");
  const [demoMode, setDemoMode] = useState(true);
  const [pipelineState, setPipelineState] = useState<PipelineState | null>(null);
  const [finalPrompt, setFinalPrompt] = useState("");

  React.useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const handleInputSubmit = (q: string, d: string, m: string, demo: boolean) => {
    setRawQuery(q); setDomain(d); setTargetModel(m); setDemoMode(demo); setStage("processing");
  };

  const handleProcessingComplete = React.useCallback((state: PipelineState) => {
    setPipelineState(state);
    setFinalPrompt(state.optimised_prompt);
    setStage("council");
  }, []);

  const handleApprove = (prompt: string) => { setFinalPrompt(prompt); setStage("execute"); };

  const handleReset = () => {
    setStage("input"); setRawQuery(""); setPipelineState(null); setFinalPrompt("");
  };

  return (
    <div className="app-shell">
      <Header theme={theme} toggleTheme={() => setTheme(t => t === "dark" ? "light" : "dark")} />
      <StageNav current={stage} />
      {stage === "input" && <StageInput onSubmit={handleInputSubmit} />}
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
        <StageExecute finalPrompt={finalPrompt} rawQuery={rawQuery} targetModel={targetModel} demoMode={demoMode} onRate={() => setStage("feedback")} />
      )}
      {stage === "feedback" && <StageFeedback onReset={handleReset} />}
    </div>
  );
}
