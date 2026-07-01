# Bravo — Redrob Hackathon Submission

Ranker for the *Intelligent Candidate Discovery & Ranking Challenge* (Senior
AI Engineer — Founding Team, Redrob AI).

## What this is

A hybrid, fully local, CPU-only ranking pipeline:

1. **Semantic layer** (`semantic.py`) — a corpus-wide TF-IDF vector space
   fit over every candidate's free text (headline, summary, every role's
   description, skill names) plus two reference documents distilled from
   `job_description.md` (what a fitting candidate's writing sounds like,
   and what an explicitly-unwanted candidate's writing sounds like).
   This is what catches "Tier 5" candidates who describe relevant work in
   plain language without using the JD's exact keywords — the thing the
   JD explicitly asks ranking systems to do and pure keyword matching
   cannot.
2. **Rule-based layer** (`scoring.py`) — experience-band fit, title fit,
   keyword skill match (corroborating signal, not primary), and
   career/company stability (consulting-only, pure-research-only
   penalties; product-company and promotion-velocity bonuses). Also
   applies the JD's explicit disqualifiers: CV/speech/robotics background
   with no NLP/IR exposure, AI experience limited to recent
   LangChain/OpenAI usage without deeper retrieval/ranking depth, and a
   junior/entry-level current title despite the role needing senior
   judgment.
3. **Behavioral multiplier** (`behavior.py`) — the 23 `redrob_signals`
   fields (recency, recruiter response rate, notice period,
   location/relocation fit, verification, etc.) applied as a
   multiplicative modifier on top of the fit score, per the JD's
   instruction to down-weight perfect-on-paper-but-unavailable
   candidates.
4. **Honeypot filter** (`honeypots.py`) — internal-consistency checks
   (expert-proficiency skills with 0 months of use, career-history
   date/duration mismatches, total-history-vs-stated-YOE mismatches)
   force trap profiles to a score of 0.0 so they never enter the top 100.
5. **Reasoning generation** (`reasoning.py`) — builds the required
   1–2 sentence justification per candidate from the *same* metadata that
   produced the score (skills actually present in the profile only — no
   hallucination — plus any disqualifier flags), so tone stays consistent
   with rank.

## Reproduce

```bash
pip install -r requirements.txt
python3 rank.py --candidates ./candidates.jsonl --out ./output_bravo.csv
python3 validate_submission.py output_bravo.csv
```

`rank.py` also accepts a gzipped `candidates.jsonl.gz` directly.

Produces `output_bravo.csv` (the required submission) and
`output_bravo_detailed.csv` (an extended debug view with each
sub-score — semantic score, career-velocity bonus, ROI flag, and which
disqualifier flags fired — not part of the required submission format,
included for the Stage 4/5 defend-your-work review).

## Measured resource use (full 100K pool, this machine)

- Wall time: ~65–70s (budget: 300s)
- Peak RSS: ~3.0 GB (budget: 16 GB)
- CPU only, no network calls during ranking, no GPU.
- Honeypot rate in top 100: 0% (self-checked at ranking time; see
  console output from `rank.py`).

## Known limitations

- The semantic layer is TF-IDF, not a dense embedding model. This was a
  deliberate choice, not an oversight: it requires no downloaded model
  weights and no network access at ranking time, which trivially
  satisfies the compute constraints, and it is fully reproducible from
  the candidate corpus alone. It is weaker than a real sentence-embedding
  model at true paraphrase/synonymy (e.g. won't necessarily equate
  "personalization engine" with "recommendation system" as strongly as a
  dense model would), so there's real headroom here if a pre-downloaded
  local embedding model is added to the repo and precomputed offline —
  the ranking step's compute budget only covers the ranking step itself,
  not precomputation, per submission_spec.md section 10.3.
- We did not implement a hard penalty for "moved into architecture/
  tech-lead roles and hasn't coded in 18 months" or general
  title-trajectory regression across a candidate's full career — both
  patterns proved too noisy to detect reliably from title text alone
  in this dataset (roughly 20–25% of the pool would be flagged, which is
  clearly not what the JD means by those disqualifiers) and risked
  penalizing genuinely strong senior engineers who legitimately hold
  Lead/Staff titles. We flag the narrower, reliably-detectable version
  of this instead (a literally junior/entry-level *current* title).
- Honeypot detection catches internal-consistency violations (~67 of the
  documented ~80 traps in this pool) but cannot catch traps that require
  external knowledge not present in the dataset (e.g. a specific
  company's real founding year), since the "product companies" in this
  dataset include fictional placeholders (Hooli, Stark Industries, etc.)
  alongside real ones — guessing real-world founding dates for the real
  ones would be an unreliable, hallucination-prone heuristic, so it was
  deliberately left out rather than risk false-flagging genuine
  candidates.
