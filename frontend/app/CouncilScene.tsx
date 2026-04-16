"use client";

import React, { useMemo } from "react";

export type CouncilPhase = "idle" | "reviewing" | "ranking" | "winner" | "synthesis" | "done";

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

interface CouncilSceneProps {
  peerReviews: PeerReview[];
  aggregateRankings: AggregateRank[];
  labelMap: Record<string, string>;
  perspectives?: Readonly<Record<string, string>>;
  phase: CouncilPhase;
  revealedCount: number;
  theme?: string;
}

const CANDIDATE_LABELS = ["X", "Y", "Z"];

export default function CouncilScene({
  peerReviews,
  aggregateRankings,
  labelMap,
  perspectives,
  phase,
  revealedCount,
  theme,
}: CouncilSceneProps) {

  // topPicks gives the index [0..2] of the candidate that reviewer i picked first
  const topPicks = useMemo(() => {
    return peerReviews.map((r) => {
      const firstPick = r.parsed_ranking[0];
      const letter = firstPick?.replace("Response ", "");
      const letterToIdx: Record<string, number> = { X: 0, Y: 1, Z: 2 };
      return letterToIdx[letter] ?? 0;
    });
  }, [peerReviews]);

  // winnerIndex gives the index [0..2] of the winning candidate
  const winnerIndex = useMemo(() => {
    if (aggregateRankings.length === 0) return 0;
    const winnerLabel = aggregateRankings[0].label;
    const letter = winnerLabel?.replace("Response ", "");
    const letterToIdx: Record<string, number> = { X: 0, Y: 1, Z: 2 };
    return letterToIdx[letter] ?? 0;
  }, [aggregateRankings]);

  const candidateAgentNames = useMemo(() => {
    return CANDIDATE_LABELS.map((l) => {
      const orig = labelMap[`Response ${l}`];
      if (!orig) return `Response ${l}`;
      const p = perspectives?.[orig];
      if (p) {
        return p.length > 25 ? p.slice(0, 24) + "…" : p;
      }
      return orig;
    });
  }, [labelMap, perspectives]);

  // Layout constants for SVG viewBox="0 0 1000 500"
  const Y_POS = [22, 50, 78]; // HTML percentages
  const Y_POS_SVG = [110, 250, 390]; // SVG Y coords (500 * Y_POS / 100)
  const REV_RIGHT_X = 270;     // x coord where lines start from Reviewers
  const CAN_LEFT_X = 380;      // x coord where lines enter Candidates
  const CAN_RIGHT_X = 620;     // x coord where lines start from Candidates
  const CON_LEFT_X = 720;      // x coord where lines enter Consensus

  function getPath(x1: number, y1: number, x2: number, y2: number) {
    const cp = (x2 - x1) / 2; // cubic bezier curve depth
    return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
  }

  return (
    <div className={`flowchart-scene theme-${theme || 'dark'} phase-${phase}`}>
      
      {/* SVG Connection Layer */}
      <svg className="flowchart-svg" viewBox="0 0 1000 500" preserveAspectRatio="none">
        
        <defs>
          <marker id="arrow-dim" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" className="marker-dim" />
          </marker>
          <marker id="arrow-dim-light" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" className="marker-dim-light" />
          </marker>
          <marker id="arrow-vote" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--indigo)" />
          </marker>
          <marker id="arrow-winner" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--terracotta)" />
          </marker>
        </defs>
        
        {/* Phase: Reviewing -> Draw all faint connections to show "investigating" phase */}
        {phase === "reviewing" && Y_POS_SVG.map((yRev, rIdx) => 
          Y_POS_SVG.map((yCan, cIdx) => {
            if (rIdx >= revealedCount) return null;
            return (
              <path 
                key={`all-${rIdx}-${cIdx}`}
                d={getPath(REV_RIGHT_X, yRev, CAN_LEFT_X, yCan)}
                className="flow-path-dim"
                markerEnd={`url(#arrow-dim${theme === 'light' ? '-light' : ''})`}
              />
            )
          })
        )}

        {/* Phase: Ranking and onwards -> Draw bold connections for actual votes */}
        {(phase === "ranking" || phase === "winner" || phase === "synthesis" || phase === "done") && 
          peerReviews.map((review, rIdx) => {
            if (rIdx >= revealedCount) return null;
            const targetCanIdx = topPicks[rIdx];
            const isWinnerPath = targetCanIdx === winnerIndex && (phase === "winner" || phase === "synthesis" || phase === "done");
            return (
              <path 
                key={`vote-${rIdx}`}
                d={getPath(REV_RIGHT_X, Y_POS_SVG[rIdx], CAN_LEFT_X, Y_POS_SVG[targetCanIdx])}
                className={`flow-path-vote ${isWinnerPath ? 'path-winner' : ''} ${phase === "ranking" ? 'path-animated' : ''}`}
                markerEnd={isWinnerPath ? 'url(#arrow-winner)' : 'url(#arrow-vote)'}
              />
            );
          })
        }

        {/* Connections from Candidates to Final Consensus (only winner line is bright) */}
        {(phase === "ranking" || phase === "winner" || phase === "synthesis" || phase === "done") &&
          CANDIDATE_LABELS.map((_, cIdx) => {
            const isWinner = cIdx === winnerIndex;
            const drawPath = isWinner && (phase === "winner" || phase === "synthesis" || phase === "done");
            return (
              <path
                key={`to-con-${cIdx}`}
                d={getPath(CAN_RIGHT_X, Y_POS_SVG[cIdx], CON_LEFT_X, 250)}
                className={drawPath ? "flow-path-winner-final path-animated" : "flow-path-dim-static"}
                markerEnd={drawPath ? 'url(#arrow-winner)' : `url(#arrow-dim${theme === 'light' ? '-light' : ''})`}
              />
            );
          })
        }
      </svg>

      {/* HTML Layer */}
      <div className="flowchart-nodes">
        <div className="flow-column-label" style={{ left: "16%", top: "10%" }}>
          <span className="flow-column-kicker">Council Input</span>
          <span className="flow-column-title">Reviewers</span>
        </div>
        <div className="flow-column-label" style={{ left: "50%", top: "10%" }}>
          <span className="flow-column-kicker">Anonymous Prompt Set</span>
          <span className="flow-column-title">Candidates</span>
        </div>
        <div className="flow-column-label flow-column-label-right" style={{ left: "84%", top: "10%" }}>
          <span className="flow-column-kicker">Chairman Output</span>
          <span className="flow-column-title">Consensus</span>
        </div>
        
        {/* Reviewers Column (Left) */}
        {Y_POS.map((y, i) => {
          const isRevealed = i < revealedCount;
          const review = peerReviews[i];
          const isActive = phase !== "idle" && phase !== "done";
          return (
            <div key={`rev-${i}`} 
                 className={`flow-pill reviewer ${isRevealed ? 'revealed' : ''} ${isActive ? 'active' : ''}`} 
                 style={{ left: '16%', top: `${y}%`, width: '22%' }}>
              <div className="flow-icon">
                {/* Generic bot icon */}
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
              </div>
              <div className="flow-text">
                <div className="flow-title">Reviewer {i+1}</div>
                <div className="flow-subtitle">{isRevealed && review ? review.model : 'Awaiting...'}</div>
              </div>
            </div>
          );
        })}

        {/* Candidates Column (Middle) */}
        {CANDIDATE_LABELS.map((label, i) => {
          const isWinner = i === winnerIndex;
          // In ranking phase, all are active. In winner phase, losers dim.
          let pillClass = '';
          if (phase === "winner" || phase === "synthesis" || phase === "done") {
            pillClass = isWinner ? 'winner' : 'loser';
          }

          return (
            <div key={`can-${label}`}
                 className={`flow-pill candidate ${pillClass}`}
                 style={{ left: '50%', top: `${Y_POS[i]}%`, width: '24%' }}>
              <div className="flow-icon candidate-icon">
                {/* Output icon */}
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
              </div>
              <div className="flow-text">
                <div className="flow-title">Candidate {label}</div>
                <div className="flow-subtitle">{candidateAgentNames[i]}</div>
              </div>
            </div>
          );
        })}

        {/* Consensus Box (Right) */}
        <div className={`flow-consensus-box ${(phase === "winner" || phase === "synthesis" || phase === "done") ? 'revealed' : ''}`}
             style={{ left: '84%', top: '50%', width: '24%' }}>
          <div className="consensus-badge">Consensus Finalised</div>
          <div className="consensus-title">{candidateAgentNames[winnerIndex] || '---'}</div>
          {phase === "synthesis" && <div className="consensus-flash">Synthesising output...</div>}
        </div>
      </div>
      
      {/* Label at the top (like "RUN TIME" in the image) */}
      <div className="flow-stage-label">
        {phase === "reviewing" && `Reviewing Candidates (${revealedCount}/3)`}
        {phase === "ranking" && "Voting in Progress"}
        {phase === "winner" && "Consensus Determined"}
        {phase === "synthesis" && "Finalising Prompts"}
        {phase === "done" && "Task Complete"}
      </div>
    </div>
  );
}
