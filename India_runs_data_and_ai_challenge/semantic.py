"""
Semantic fit layer.

The original scoring.py only counted hits against fixed keyword lists
(faiss, pinecone, ndcg, ...). The job description explicitly warns against
exactly that failure mode:

    "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
    profile, but if their career history shows they built a recommendation
    system at a product company, they're a fit."

Pure keyword matching cannot see that candidate. This module adds a
corpus-wide TF-IDF semantic layer over the *entire* free-text of each
profile (headline, summary, every role's description, skill names) and
scores it against two hand-written reference documents distilled from
job_description.md:

  - JD_POSITIVE_TEXT: what the ideal candidate's own writing sounds like
  - JD_NEGATIVE_TEXT: what a JD-defined non-fit's writing sounds like
    (pure research, consulting-only, framework-tutorial blogging,
    CV/speech/robotics without NLP, title-chasing)

A candidate's semantic_score rewards topical closeness to the JD even with
zero exact keyword overlap, and penalizes closeness to the explicitly
unwanted archetypes even if they happen to share surface vocabulary with
the JD (e.g. an "AI" title with no real IR/ranking substance).

Design constraints honored:
  - CPU only, no network calls, no downloaded model weights.
  - Fits in seconds on 100K documents (sparse TF-IDF, not embeddings),
    well inside the 5-minute / 16GB budget.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Distilled from job_description.md: "Things you absolutely need",
# "What you'd actually be doing", and "How to read between the lines".
# This is *not* a keyword list — it's prose, so TF-IDF picks up the
# vocabulary a genuinely-fitting candidate would plausibly use themselves,
# even in paraphrase (recommendation systems, search relevance, matching
# pipelines, evaluation frameworks, etc.), not just the JD's own jargon.
JD_POSITIVE_TEXT = """
Senior AI engineer owning the intelligence layer of a product: ranking,
retrieval, and matching systems that decide what recruiters see when they
search for candidates and what candidates see when they search for roles.
Production experience with embeddings-based retrieval systems deployed to
real users: sentence-transformers, OpenAI embeddings, BGE, E5 style dense
representations. Handled embedding drift, index refresh, retrieval quality
regression in production. Production experience with vector databases or
hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS, BM25, hybrid dense plus sparse retrieval.
Strong Python and code quality. Designed evaluation frameworks for ranking
systems: NDCG, MRR, MAP, offline to online correlation, A/B test
interpretation, recruiter engagement metrics, click through rate.
Shipped an end to end ranking, search, or recommendation system to real
users at meaningful scale at a product company, not just a research
prototype. Built a recommendation engine, personalization system, search
relevance pipeline, matching algorithm, candidate-job matching, query
understanding, semantic search, learning to rank model, LLM based
re-ranking, retrieval augmented generation system. Comfortable auditing an
existing BM25 plus rule based scoring system and shipping a v2 ranking
system with hybrid retrieval. Experience with LLM fine-tuning, LoRA,
QLoRA, PEFT, learning-to-rank models such as XGBoost based or neural
ranking, distributed systems, large scale inference optimization,
open-source contributions in AI or ML. Scrappy product engineering
attitude, willing to ship a working ranker quickly, thinks about systems
not frameworks, mentors other engineers, works closely with product on
recruiter experience, has strong informed opinions about retrieval and
evaluation and can defend design choices with reference to systems
actually built and shipped.
"""

# Distilled from "Things we explicitly do NOT want" plus the disqualifiers
# section. Deliberately written to sound like a resume/summary itself
# (first person, plausible profile language) so it captures the *style* of
# a mismatched profile, not just isolated negative buzzwords.
JD_NEGATIVE_TEXT = """
Pure academic or research lab background with no production deployment
experience, publishing papers without shipping systems to real users.
Career spent entirely at IT services and consulting firms such as TCS,
Infosys, Wipro, Accenture, Cognizant, Capgemini, Tech Mahindra, HCL,
Genpact, staffed onto client projects rather than owning a product.
Recent AI experience limited to calling OpenAI APIs through LangChain for
under a year, tutorial-style demo projects, blog posts about the latest
hot framework, no deep infrastructure or evaluation experience. Career
trajectory optimized for chasing titles, senior to staff to principal,
switching companies every year or two, not planning to stay anywhere long.
Primary expertise in computer vision, image classification, object
detection, speech recognition, or robotics, with no natural language
processing, information retrieval, search, or ranking exposure. Moved
into pure architecture, tech lead, or management responsibilities and has
not personally written production code in a long time. Entirely
closed-source proprietary work for many years with no external validation,
no talks, no writing, no open-source contributions, hard to verify how
they actually think. Marketing manager, HR generalist, recruiter,
business analyst, scrum master, sales, accountant, mechanical, civil, or
electrical engineering background with no software or ML experience,
skills section lists many AI keywords but job title and actual
responsibilities are unrelated, keyword stuffed profile with no
substantive project descriptions.
"""


def build_candidate_text(cand: dict) -> str:
    """Concatenate every piece of free text in a candidate's profile.

    Deliberately excludes structured fields already handled by the rule
    engine (skills' proficiency/duration, redrob_signals, etc.) — this is
    purely the prose a human would actually read to judge fit.
    """
    profile = cand.get("profile", {})
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for role in cand.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for s in cand.get("skills", []):
        parts.append(s.get("name", ""))
    for edu in cand.get("education", []):
        parts.append(edu.get("field_of_study", ""))
    return " ".join(p for p in parts if p)


def compute_semantic_scores(candidate_texts):
    """Vectorize the whole candidate corpus + the two JD reference docs once.

    Returns three numpy arrays (len == len(candidate_texts)):
      pos_sim   - raw cosine similarity to JD_POSITIVE_TEXT
      neg_sim   - raw cosine similarity to JD_NEGATIVE_TEXT
      semantic_score - pos_sim - 0.5*neg_sim, min-max normalized to [0, 1]
                        across the pool so it composes cleanly with the
                        0-1-scaled rule-based sub-scores.
    """
    vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    corpus = candidate_texts + [JD_POSITIVE_TEXT, JD_NEGATIVE_TEXT]
    tfidf = vectorizer.fit_transform(corpus)

    cand_matrix = tfidf[:-2]
    pos_vec = tfidf[-2]
    neg_vec = tfidf[-1]

    # TfidfVectorizer output rows are L2-normalized by default, so the dot
    # product against another L2-normalized vector *is* cosine similarity.
    pos_sim = np.asarray(cand_matrix.dot(pos_vec.T).todense()).ravel()
    neg_sim = np.asarray(cand_matrix.dot(neg_vec.T).todense()).ravel()

    raw = pos_sim - 0.5 * neg_sim
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        semantic_score = (raw - lo) / (hi - lo)
    else:
        semantic_score = np.zeros_like(raw)

    return pos_sim, neg_sim, semantic_score
