#!/usr/bin/env python3
"""
Redrob Hackathon — candidate ranker.

Pipeline:
  1. Stream-load candidates.jsonl (or candidates.jsonl.gz) into memory.
  2. Build free-text per candidate and compute a corpus-wide TF-IDF
     semantic-fit score against the JD (semantic.py) in one vectorized
     pass — this is what lets us catch "Tier 5" candidates who describe
     relevant work without using the JD's exact keywords.
  3. Score every candidate (scoring.py: rule-based sub-scores + semantic
     score + honeypot filter + behavioral multiplier).
  4. Sort, take the top 100, write the required submission CSV plus an
     optional extended debug CSV with the individual sub-scores that
     produced each rank (useful for the Stage 4/5 defend-your-work
     review, not part of the required submission format).

Usage:
    python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import argparse
import gzip
import json
import time

from scoring import calculate_candidate_score
from reasoning import generate_custom_reasoning
from semantic import build_candidate_text, compute_semantic_scores

CANDIDATE_ID_PATTERN_LEN = len("CAND_0000000")


def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def rank_candidates(candidates_path, output_path):
    t0 = time.time()

    # ---- 1. Load ---------------------------------------------------
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    with _open(candidates_path) as f:
        for line in f:
            if not line.strip():
                continue
            candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates in {time.time() - t0:.2f}s")

    # Sanity check: every id must match the expected schema so a bad row
    # never silently corrupts the submission (spec section 3: "Every
    # candidate_id must exist in the released candidates.jsonl").
    seen_ids = set()
    for c in candidates:
        cid = c.get("candidate_id", "")
        if not cid.startswith("CAND_") or len(cid) != CANDIDATE_ID_PATTERN_LEN:
            raise ValueError(f"Malformed candidate_id encountered: {cid!r}")
        if cid in seen_ids:
            raise ValueError(f"Duplicate candidate_id in source data: {cid!r}")
        seen_ids.add(cid)

    # ---- 2. Semantic layer (corpus-wide, one pass) ------------------
    t_sem = time.time()
    texts = [build_candidate_text(c) for c in candidates]
    _, _, semantic_scores = compute_semantic_scores(texts)
    print(f"Computed semantic scores for {len(candidates)} candidates in {time.time() - t_sem:.2f}s")

    # ---- 3. Score every candidate ------------------------------------
    t_score = time.time()
    scored_candidates = []
    for cand, sem_score in zip(candidates, semantic_scores):
        cid = cand["candidate_id"]
        score, reasons, meta = calculate_candidate_score(cand, sem_score)
        score_rounded = round(score, 3)
        scored_candidates.append((score_rounded, cid, cand, meta))
    print(f"Scored {len(scored_candidates)} candidates in {time.time() - t_score:.2f}s")

    # Sort by score descending, candidate_id ascending as deterministic
    # tiebreaker (matches submission_spec.md section 3's tie-break rule).
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))

    # ---- 4. Honeypot rate self-check on the shortlist ----------------
    from honeypots import is_honeypot
    top_100 = scored_candidates[:100]
    honeypot_hits = sum(1 for _, _, cand, _ in top_100 if is_honeypot(cand))
    if honeypot_hits > 0:
        # Should never happen — honeypots are forced to score 0.0 — but
        # verify explicitly rather than trusting that silently.
        print(f"WARNING: {honeypot_hits} honeypot(s) leaked into the top 100!")
    else:
        print("Honeypot self-check: 0/100 honeypots in the shortlist.")

    # ---- 5. Write the compliant submission CSV -----------------------
    print(f"Writing compliant top 100 to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("candidate_id,rank,score,reasoning\n")
        for i, (score, cid, cand, meta) in enumerate(top_100):
            rank = i + 1
            reasoning = generate_custom_reasoning(cand, rank, meta)
            escaped_reasoning = reasoning.replace('"', '""')
            out_f.write(f'{cid},{rank},{score:.3f},"{escaped_reasoning}"\n')

    # ---- 6. Write the extended debug CSV (not part of the required
    #         submission format — useful for defending the ranking) ----
    detailed_path = output_path.replace(".csv", "_detailed.csv")
    print(f"Writing detailed analysis report to {detailed_path}...")
    with open(detailed_path, "w", encoding="utf-8") as detailed_f:
        detailed_f.write(
            "candidate_id,rank,score,semantic_score,career_velocity_bonus,"
            "expected_roi,junior_title_flag,cv_speech_robotics_flag,"
            "langchain_only_flag,reasoning\n"
        )
        for i, (score, cid, cand, meta) in enumerate(top_100):
            rank = i + 1
            reasoning = generate_custom_reasoning(cand, rank, meta)
            escaped_reasoning = reasoning.replace('"', '""')
            vel = meta.get("career_velocity", 0.0)
            roi = meta.get("expected_roi", "Standard")
            sem = meta.get("semantic_score", 0.0)
            detailed_f.write(
                f'{cid},{rank},{score:.3f},{sem:.3f},{vel:.1f},{roi},'
                f'{meta.get("junior_title_flag")},{meta.get("cv_speech_robotics_flag")},'
                f'{meta.get("langchain_only_flag")},"{escaped_reasoning}"\n'
            )

    total_t = time.time() - t0
    print(f"Completed in {total_t:.2f} seconds.")
    if total_t > 300:
        print("WARNING: exceeded the 5-minute (300s) compute budget!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob hackathon.")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl(.gz) file")
    parser.add_argument("--out", type=str, required=True, help="Path to write submission.csv")
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out)
