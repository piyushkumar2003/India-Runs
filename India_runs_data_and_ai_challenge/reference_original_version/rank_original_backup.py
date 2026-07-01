#!/usr/bin/env python3
import json
import argparse
import time
import os
from scoring import calculate_candidate_score
from reasoning import generate_custom_reasoning

def rank_candidates(candidates_path, output_path):
    print(f"Loading candidates from {candidates_path}...")
    start_time = time.time()
    
    scored_candidates = []
    
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand['candidate_id']
            score, reasons, meta = calculate_candidate_score(cand)
            score_rounded = round(score, 3)
            scored_candidates.append((score_rounded, cid, cand, meta))
            
    # Sort by score descending, and candidate_id ascending as tiebreaker
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))
    
    print(f"Scored {len(scored_candidates)} candidates in {time.time() - start_time:.2f} seconds.")
    
    # Select top 100
    top_100 = scored_candidates[:100]
    
    # 1. Write the compliant submission CSV
    print(f"Writing compliant top 100 to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("candidate_id,rank,score,reasoning\n")
        for i, (score, cid, cand, meta) in enumerate(top_100):
            rank = i + 1
            reasoning = generate_custom_reasoning(cand, rank)
            # Escape quotes in reasoning
            escaped_reasoning = reasoning.replace('"', '""')
            out_f.write(f'{cid},{rank},{score:.3f},"{escaped_reasoning}"\n')
            
    # 2. Write the detailed CSV (Idea A and B visible in separate columns)
    detailed_path = output_path.replace('.csv', '_detailed.csv')
    print(f"Writing detailed analysis report to {detailed_path}...")
    with open(detailed_path, 'w', encoding='utf-8') as detailed_f:
        detailed_f.write("candidate_id,rank,score,career_velocity_bonus,expected_roi,reasoning\n")
        for i, (score, cid, cand, meta) in enumerate(top_100):
            rank = i + 1
            reasoning = generate_custom_reasoning(cand, rank)
            escaped_reasoning = reasoning.replace('"', '""')
            vel = meta.get('career_velocity', 0.0)
            roi = meta.get('expected_roi', 'Standard')
            detailed_f.write(f'{cid},{rank},{score:.3f},{vel:.1f},{roi},"{escaped_reasoning}"\n')
            
    print(f"Completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob hackathon.")
    parser.add_argument('--candidates', type=str, required=True, help="Path to candidates.jsonl file")
    parser.add_argument('--out', type=str, required=True, help="Path to write submission.csv")
    args = parser.parse_args()
    
    rank_candidates(args.candidates, args.out)
