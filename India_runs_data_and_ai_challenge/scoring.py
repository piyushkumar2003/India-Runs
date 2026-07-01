import re
from behavior import compute_behavioral_multiplier
from honeypots import is_honeypot

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'tech mahindra', 'l&t infotech', 'mindtree', 'mphasis', 'hcltech',
    'tata consultancy', 'lnt infotech', 'hcl technologies', 'genpact',
    'cognizant technology solutions', 'capgemini technology services',
    'wipro limited', 'infosys limited', 'tata consultancy services'
}

PRODUCT_COMPANIES = {
    'stripe', 'uber', 'grab', 'gojek', 'razorpay', 'cred', 'swiggy', 'zomato',
    'paytm', 'phonepe', 'hooli', 'stark industries', 'wayne enterprises', 'initech',
    'globex', 'acme', 'dunder mifflin', 'vegas', 'observe.ai', 'verloop.io', 'locobuzz',
    'aganitha', 'sarvam ai', 'rephrase.ai', 'google', 'microsoft', 'adobe', 'meta',
    'amazon', 'netflix', 'apple', 'flipkart', 'ola', 'inmobi', 'meesho', 'zepto'
}

# JD: "People whose primary expertise is computer vision, speech, or
# robotics without significant NLP/IR exposure ... you'd be re-learning
# fundamentals here."
CV_SPEECH_ROBOTICS_KEYWORDS = [
    'computer vision', 'image classification', 'object detection',
    'image segmentation', 'resnet', 'convolutional neural network',
    'robotics', 'speech recognition', 'autonomous driving', 'lidar',
    'image moderation',
]
NLP_IR_KEYWORDS = [
    'nlp', 'natural language', 'retrieval', 'ranking', 'embeddings',
    'language model', 'llm', 'search relevance', 'text classification',
    'named entity', 'tokeniz', 'bm25', 'vector search', 'semantic search',
    'information retrieval', 'recommendation',
]

# JD: "'AI experience' consists primarily of recent (under 12 months)
# projects using LangChain to call OpenAI ... unless you can demonstrate
# substantial pre-LLM-era ML production experience."
DEEP_PRE_LLM_KEYWORDS = [
    'sentence-transformers', 'sentence_transformers', 'faiss', 'pinecone',
    'qdrant', 'weaviate', 'milvus', 'bm25', 'elasticsearch', 'fine-tun',
    'fine tun', 'bge', 'e5 embedding', 'opensearch', 'hybrid search',
    'learning to rank', 'learning-to-rank', 'xgboost', 'lightgbm',
]

JUNIOR_TITLE_PATTERN = re.compile(r'\b(junior|jr\.?|intern|trainee|entry.level)\b')


def clean_text(text):
    if not text:
        return ""
    return text.lower().strip()


def get_title_level(title):
    t_lower = title.lower()
    if any(w in t_lower for w in ['vp', 'director', 'head', 'founder', 'principal', 'staff', 'architect', 'lead', 'manager', 'lead engineer', 'team lead']):
        return 4
    if 'senior' in t_lower or 'sr' in t_lower:
        return 3
    if any(w in t_lower for w in ['junior', 'jr', 'intern', 'associate', 'trainee', 'student']):
        return 1
    return 2


def calculate_candidate_score(cand, semantic_score=0.0):
    """Compute a candidate's composite fit score.

    semantic_score: precomputed TF-IDF semantic-fit score in [0, 1] from
    semantic.compute_semantic_scores(), covering the whole candidate pool
    at once (see semantic.py). Passed in rather than recomputed per
    candidate because it requires corpus-wide statistics.
    """
    metadata = {
        'career_velocity': 0.0,
        'expected_roi': 'Standard',
        'semantic_score': round(float(semantic_score), 3),
        'cv_speech_robotics_flag': False,
        'langchain_only_flag': False,
        'junior_title_flag': False,
    }

    # Honeypot Check: Trap candidates are forced to 0.0
    if is_honeypot(cand):
        return 0.0, ["Honeypot/Trap detected"], metadata

    profile = cand.get('profile', {})
    history = cand.get('career_history', [])
    skills = cand.get('skills', [])

    reasons = []

    # -------------------------------------------------------------
    # 1. Experience Agent (YoE sweet spot check)
    # -------------------------------------------------------------
    yoe = profile.get('years_of_experience', 0.0)
    exp_score = 0.0
    if 6.0 <= yoe <= 8.0:
        exp_score = 1.0
        reasons.append(f"Ideal YOE ({yoe} years)")
    elif 5.0 <= yoe <= 9.0:
        exp_score = 0.9
        reasons.append(f"Target YOE ({yoe} years)")
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0:
        exp_score = 0.7
        reasons.append(f"Acceptable YOE ({yoe} years)")
    elif 3.0 <= yoe < 4.0 or 11.0 < yoe <= 13.0:
        exp_score = 0.4
        reasons.append(f"Suboptimal YOE ({yoe} years)")
    else:
        exp_score = 0.1
        reasons.append(f"Poor YOE match ({yoe} years)")

    # -------------------------------------------------------------
    # 2. Domain & Skills Match Agent (Retrieval, Ranking, Production ML)
    #    NOTE: this is a *corroborating* keyword signal, not the primary
    #    fit signal. The semantic layer (see below) is what catches
    #    plain-language "Tier 5" candidates who don't use these exact
    #    terms. This section is deliberately weighted lower than in the
    #    original version of this file.
    # -------------------------------------------------------------
    profile_text = clean_text(profile.get('headline', '')) + " " + clean_text(profile.get('summary', ''))
    for role in history:
        profile_text += " " + clean_text(role.get('title', '')) + " " + clean_text(role.get('description', ''))

    retrieval_keywords = ['faiss', 'milvus', 'qdrant', 'pinecone', 'weaviate', 'opensearch', 'elasticsearch', 'hybrid search', 'dense retrieval', 'vector search', 'rag', 'retrieval']
    ranking_keywords = ['ndcg', 'mrr', 'map', 'ltr', 'learning to rank', 'learning-to-rank', 'ranking', 'relevance', 'recommendation']
    production_ml_keywords = ['pytorch', 'tensorflow', 'scikit-learn', 'numpy', 'pandas', 'xgboost', 'lightgbm', 'fastapi', 'docker', 'kubernetes', 'ci/cd', 'deployment', 'serving', 'inference', 'scaling', 'latency']
    llm_keywords = ['lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'embeddings', 'sentence-transformers', 'sentence_transformers', 'transformer']

    retrieval_hits = sum(1 for kw in retrieval_keywords if kw in profile_text)
    ranking_hits = sum(1 for kw in ranking_keywords if kw in profile_text)
    production_ml_hits = sum(1 for kw in production_ml_keywords if kw in profile_text)
    llm_hits = sum(1 for kw in llm_keywords if kw in profile_text)

    explicit_skills_score = 0.0
    for s in skills:
        name = clean_text(s.get('name'))
        prof = s.get('proficiency', 'beginner')
        dur = s.get('duration_months', 0)

        prof_mult = 1.0
        if prof == 'expert':
            prof_mult = 1.5
        elif prof == 'advanced':
            prof_mult = 1.2
        elif prof == 'intermediate':
            prof_mult = 0.8
        elif prof == 'beginner':
            prof_mult = 0.4

        matched_weight = 0.0
        for kw in retrieval_keywords + ranking_keywords + production_ml_keywords + llm_keywords:
            if kw in name or name in kw:
                if kw in retrieval_keywords or kw in ranking_keywords:
                    matched_weight = max(matched_weight, 3.0)
                elif kw in llm_keywords:
                    matched_weight = max(matched_weight, 2.0)
                else:
                    matched_weight = max(matched_weight, 1.5)

        if matched_weight > 0:
            dur_contrib = min(dur / 24.0, 2.0)
            explicit_skills_score += matched_weight * prof_mult * (1.0 + dur_contrib)

    retrieval_score = min(retrieval_hits / 4.0, 1.0)
    ranking_score = min(ranking_hits / 3.0, 1.0)
    production_score = min(production_ml_hits / 5.0, 1.0)
    llm_score = min(llm_hits / 3.0, 1.0)

    skills_fit_score = min(explicit_skills_score / 20.0, 1.0)

    skills_agent_score = (retrieval_score * 0.30 +
                          ranking_score * 0.30 +
                          production_score * 0.15 +
                          llm_score * 0.10 +
                          skills_fit_score * 0.15)

    if retrieval_score > 0.5:
        reasons.append("Strong retrieval foundation (keyword match)")
    if ranking_score > 0.5:
        reasons.append("Experienced in ranking and relevance evaluation (keyword match)")
    if skills_agent_score < 0.1 and semantic_score > 0.6:
        reasons.append("Plain-language profile: no exact JD keyword hits, but career narrative reads as a strong semantic match")

    # -------------------------------------------------------------
    # 3. Role/Title Fit Agent
    # -------------------------------------------------------------
    title_score = 0.0
    current_title = clean_text(profile.get('current_title', ''))

    positive_title_patterns = [
        r'\bai\b', r'\bml\b', r'machine learning', r'nlp', r'search', r'recommendation',
        r'retrieval', r'ranking', r'data scientist', r'applied ml', r'applied science'
    ]
    is_positive_title = any(re.search(pat, current_title) for pat in positive_title_patterns)

    negative_title_patterns = [
        r'marketing', r'qa', r'test', r'hr', r'recruiter', r'product manager', r'designer',
        r'business analyst', r'scrum master', r'sales', r'accountant', r'mechanical engineer',
        r'civil engineer', r'electrical engineer', r'hardware engineer'
    ]
    is_negative_title = any(re.search(pat, current_title) for pat in negative_title_patterns)

    if is_positive_title and not is_negative_title:
        title_score = 1.0
        reasons.append("Current title matches AI/ML domain")
    elif is_negative_title:
        title_score = -1.0
        reasons.append("Current title is outside the target technical scope")
    else:
        title_score = 0.6
        reasons.append("Current title is backend/software engineering")

    # JD targets "5-9 years, senior judgment" — a literally junior/intern
    # current title is a real seniority-mismatch signal worth surfacing,
    # even if the person's skills read fine. Cap rather than zero it out,
    # since titles are noisy (company-specific leveling).
    if title_score > 0 and JUNIOR_TITLE_PATTERN.search(current_title):
        title_score = min(title_score, 0.35)
        metadata['junior_title_flag'] = True
        reasons.append("Current title reads as junior/entry-level despite the role needing senior judgment — seniority signal is weak")

    # -------------------------------------------------------------
    # 4. Career History, Company Fit & Stability Agent
    # -------------------------------------------------------------
    total_companies = len(history)
    consulting_companies_count = 0
    product_companies_count = 0
    academic_research_roles = 0
    total_duration_months = 0
    job_hopper_risk = False

    title_levels = []

    for role in history:
        comp = clean_text(role.get('company', ''))
        title = clean_text(role.get('title', ''))
        dur = role.get('duration_months', 0)
        size = role.get('company_size', '')

        total_duration_months += dur
        title_levels.append(get_title_level(title))

        if any(c in comp for c in CONSULTING_FIRMS):
            consulting_companies_count += 1

        if any(p in comp for p in PRODUCT_COMPANIES) or size in ['1-10', '11-50', '51-200']:
            product_companies_count += 1

        if 'research' in title or 'academic' in title or 'postdoc' in title or 'intern' in title:
            academic_research_roles += 1

    if total_companies > 0:
        avg_tenure = total_duration_months / total_companies
        if avg_tenure < 18.0 and yoe > 3.0:
            job_hopper_risk = True

    company_score = 0.5

    if total_companies > 0:
        if consulting_companies_count == total_companies:
            company_score = 0.1
            reasons.append("Entire career at consulting/services firms")
        elif consulting_companies_count > 0:
            company_score *= 0.7
            reasons.append("Some career history at consulting/services firms")

        if academic_research_roles == total_companies:
            company_score = 0.1
            reasons.append("Pure academic/research history")

        if product_companies_count > 0:
            company_score = min(company_score + 0.4, 1.0)
            reasons.append("Strong startup or product company background")

    if job_hopper_risk:
        company_score *= 0.7
        reasons.append("Concern: Short role durations (job-hopper / title-chasing profile)")

    velocity_bonus = 0.0
    if len(title_levels) >= 2:
        title_levels_chrono = list(reversed(title_levels))
        first_level = title_levels_chrono[0]
        max_level = max(title_levels_chrono)
        if max_level > first_level:
            velocity_bonus = float(max_level - first_level) * 2.5
            reasons.append("Demonstrated career progression and promotion velocity")

    metadata['career_velocity'] = velocity_bonus

    # -------------------------------------------------------------
    # 5. JD-specific disqualifier checks (explicit "do NOT want" list)
    # -------------------------------------------------------------
    cv_speech_robotics_hit = any(kw in profile_text for kw in CV_SPEECH_ROBOTICS_KEYWORDS)
    nlp_ir_hit = any(kw in profile_text for kw in NLP_IR_KEYWORDS)
    cv_penalty = 1.0
    if cv_speech_robotics_hit and not nlp_ir_hit:
        cv_penalty = 0.4
        metadata['cv_speech_robotics_flag'] = True
        reasons.append("Background reads as CV/speech/robotics-focused with no NLP/IR exposure — JD explicitly flags this as a mismatch")

    langchain_penalty = 1.0
    has_langchain = 'langchain' in profile_text
    has_deep_pre_llm = any(kw in profile_text for kw in DEEP_PRE_LLM_KEYWORDS)
    if has_langchain and not has_deep_pre_llm and yoe < 3.0:
        langchain_penalty = 0.7
        metadata['langchain_only_flag'] = True
        reasons.append("AI experience appears limited to recent LangChain/OpenAI-API integration without demonstrated retrieval/ranking depth")

    # -------------------------------------------------------------
    # 6. Composite Base Score Construction
    #    Weights re-balanced from the original keyword-only version to
    #    give the semantic layer real influence (30 pts) while reducing
    #    reliance on exact keyword hits (40 -> 20 pts), per the JD's own
    #    instruction to look past keywords.
    # -------------------------------------------------------------
    base_score = (
        exp_score * 15.0 +
        max(0.0, title_score) * 15.0 +
        skills_agent_score * 20.0 +
        company_score * 15.0 +
        semantic_score * 30.0 +
        velocity_bonus
    )

    if title_score < 0:
        base_score *= 0.1

    # FIX: the original version zeroed out (x0.05) any candidate with no
    # exact keyword hits at all, regardless of semantic fit. That directly
    # punishes the "Tier 5 plain-language candidate" the JD explicitly asks
    # us to find. Now the strict penalty only fires when the candidate is
    # weak on *both* keyword and semantic signals.
    if (retrieval_score == 0.0 and ranking_score == 0.0 and llm_score == 0.0
            and skills_fit_score == 0.0 and semantic_score < 0.15):
        base_score *= 0.05
        reasons.append("No keyword or semantic signal of retrieval/ranking/ML domain fit")

    base_score *= cv_penalty
    base_score *= langchain_penalty

    base_score = max(0.0, min(base_score, 100.0))

    # -------------------------------------------------------------
    # 7. Behavioral Multiplier Adjustment
    # -------------------------------------------------------------
    expected_salary = cand.get('redrob_signals', {}).get('expected_salary_range_inr_lpa', {})
    expected_min = expected_salary.get('min', 0.0)
    if expected_min > 0.0 and yoe > 0.0:
        if yoe <= 2.0:
            baseline = 10.0
        elif yoe <= 5.0:
            baseline = 10.0 + (yoe - 2.0) * 4.0
        elif yoe <= 9.0:
            baseline = 22.0 + (yoe - 5.0) * 6.0
        else:
            baseline = 46.0 + (yoe - 9.0) * 3.0

        if expected_min < baseline * 0.8:
            metadata['expected_roi'] = 'High ROI (Bargain)'
        elif expected_min > baseline * 1.4:
            metadata['expected_roi'] = 'Low ROI (Premium)'

    behavior_multiplier = compute_behavioral_multiplier(cand)
    final_score = base_score * behavior_multiplier

    return final_score, reasons, metadata
