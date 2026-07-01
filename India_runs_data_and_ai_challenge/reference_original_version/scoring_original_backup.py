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

def calculate_candidate_score(cand):
    metadata = {
        'career_velocity': 0.0,
        'expected_roi': 'Standard'
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
    # -------------------------------------------------------------
    # Build complete profile text for token matching (headlines, summary, role descriptions)
    profile_text = clean_text(profile.get('headline', '')) + " " + clean_text(profile.get('summary', ''))
    for role in history:
        profile_text += " " + clean_text(role.get('title', '')) + " " + clean_text(role.get('description', ''))
        
    # Core domain concepts with matching weights
    retrieval_keywords = ['faiss', 'milvus', 'qdrant', 'pinecone', 'weaviate', 'opensearch', 'elasticsearch', 'hybrid search', 'dense retrieval', 'vector search', 'rag', 'retrieval']
    ranking_keywords = ['ndcg', 'mrr', 'map', 'ltr', 'learning to rank', 'learning-to-rank', 'ranking', 'relevance', 'recommendation']
    production_ml_keywords = ['pytorch', 'tensorflow', 'scikit-learn', 'numpy', 'pandas', 'xgboost', 'lightgbm', 'fastapi', 'docker', 'kubernetes', 'ci/cd', 'deployment', 'serving', 'inference', 'scaling', 'latency']
    llm_keywords = ['lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'embeddings', 'sentence-transformers', 'sentence_transformers', 'transformer']
    
    # Calculate presence counts
    retrieval_hits = sum(1 for kw in retrieval_keywords if kw in profile_text)
    ranking_hits = sum(1 for kw in ranking_keywords if kw in profile_text)
    production_ml_hits = sum(1 for kw in production_ml_keywords if kw in profile_text)
    llm_hits = sum(1 for kw in llm_keywords if kw in profile_text)
    
    # Also score explicit skills section
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
                # Assign core weights based on JD priority
                if kw in retrieval_keywords or kw in ranking_keywords:
                    matched_weight = max(matched_weight, 3.0)
                elif kw in llm_keywords:
                    matched_weight = max(matched_weight, 2.0)
                else:
                    matched_weight = max(matched_weight, 1.5)
                    
        if matched_weight > 0:
            dur_contrib = min(dur / 24.0, 2.0)
            explicit_skills_score += matched_weight * prof_mult * (1.0 + dur_contrib)
            
    # Normalize skill scores
    retrieval_score = min(retrieval_hits / 4.0, 1.0)
    ranking_score = min(ranking_hits / 3.0, 1.0)
    production_score = min(production_ml_hits / 5.0, 1.0)
    llm_score = min(llm_hits / 3.0, 1.0)
    
    skills_fit_score = min(explicit_skills_score / 20.0, 1.0)
    
    # Combine skills agent
    skills_agent_score = (retrieval_score * 0.30 +
                          ranking_score * 0.30 +
                          production_score * 0.15 +
                          llm_score * 0.10 +
                          skills_fit_score * 0.15)
                          
    if retrieval_score > 0.5:
        reasons.append("Strong retrieval foundation")
    if ranking_score > 0.5:
        reasons.append("Experienced in ranking and relevance evaluation")
        
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
        
    # -------------------------------------------------------------
    # 4. Career History, Company Fit & Stability Agent
    # -------------------------------------------------------------
    total_companies = len(history)
    consulting_companies_count = 0
    product_companies_count = 0
    academic_research_roles = 0
    total_duration_months = 0
    job_hopper_risk = False
    
    # Career Velocity (Idea A) calculation setup
    title_levels = []
    
    for role in history:
        comp = clean_text(role.get('company', ''))
        title = clean_text(role.get('title', ''))
        dur = role.get('duration_months', 0)
        size = role.get('company_size', '')
        
        total_duration_months += dur
        
        # Track title level for career velocity
        title_levels.append(get_title_level(title))
        
        # Check consulting/services firms
        if any(c in comp for c in CONSULTING_FIRMS):
            consulting_companies_count += 1
            
        # Check product/startup indicators
        if any(p in comp for p in PRODUCT_COMPANIES) or size in ['1-10', '11-50', '51-200']:
            product_companies_count += 1
            
        if 'research' in title or 'academic' in title or 'postdoc' in title or 'intern' in title:
            academic_research_roles += 1
            
    # Check job stability: average tenure < 18 months indicates high switch rate
    if total_companies > 0:
        avg_tenure = total_duration_months / total_companies
        if avg_tenure < 18.0 and yoe > 3.0:
            job_hopper_risk = True
            
    # Calculate company/stability score
    company_score = 0.5 # default starting baseline
    
    if total_companies > 0:
        if consulting_companies_count == total_companies:
            company_score = 0.1 # Heavily penalize all-consulting history
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
        reasons.append("Concern: Short role durations (job-hopper profile)")
        
    # Career Velocity (Idea A): Boost if candidate grew significantly over their career
    velocity_bonus = 0.0
    if len(title_levels) >= 2:
        # Chronological order is reverse of history (which has current role first)
        title_levels_chrono = list(reversed(title_levels))
        first_level = title_levels_chrono[0]
        max_level = max(title_levels_chrono)
        if max_level > first_level:
            velocity_bonus = float(max_level - first_level) * 2.5 # Boost up to 7.5 points
            reasons.append("Demonstrated career progression and promotion velocity")
            
    metadata['career_velocity'] = velocity_bonus
    
    # -------------------------------------------------------------
    # 5. Composite Base Score Construction
    # -------------------------------------------------------------
    # Scale components to 100 max points
    # Weight distribution: YOE (20 points), Title Fit (20 points), Skills (40 points), Career Company Fit (20 points)
    base_score = (
        exp_score * 20.0 +
        max(0.0, title_score) * 20.0 +
        skills_agent_score * 40.0 +
        company_score * 20.0 +
        velocity_bonus
    )
    
    # Apply negative title penalty directly
    if title_score < 0:
        base_score *= 0.1
        
    # Apply strict skill filtering: if no retrieval/ranking/LLM expertise at all, penalize heavily
    if retrieval_score == 0.0 and ranking_score == 0.0 and llm_score == 0.0 and skills_fit_score == 0.0:
        base_score *= 0.05
        
    # Ensure score bounds
    base_score = max(0.0, min(base_score, 100.0))
    
    # -------------------------------------------------------------
    # 6. Behavioral Multiplier Adjustment
    # -------------------------------------------------------------
    # Compute behavioral multiplier, and also get the ROI flag inside metadata
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
