def generate_custom_reasoning(cand, rank):
    profile = cand.get('profile', {})
    signals = cand.get('redrob_signals', {})
    skills = cand.get('skills', [])
    history = cand.get('career_history', [])
    
    yoe = profile.get('years_of_experience', 0.0)
    title = profile.get('current_title', 'Engineer')
    company = profile.get('current_company', 'Startup')
    notice = signals.get('notice_period_days', 90)
    loc = profile.get('location', '')
    
    # 1. Identify actual skills present in candidate profile
    profile_skills = {s.get('name') for s in skills if s.get('name')}
    
    # Define JD matching target skill keywords (exact lowercase checks)
    jd_skills = ['faiss', 'milvus', 'qdrant', 'pinecone', 'weaviate', 'opensearch', 'elasticsearch', 
                 'ndcg', 'mrr', 'map', 'ltr', 'learning-to-rank', 'learning to rank', 'rag', 'hybrid search',
                 'pytorch', 'tensorflow', 'scikit-learn', 'xgboost', 'lightgbm', 'fastapi', 'docker', 'kubernetes',
                 'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'embeddings', 'sentence-transformers']
                 
    matched_skills = []
    for ps in profile_skills:
        ps_lower = ps.lower()
        if any(js in ps_lower or ps_lower in js for js in jd_skills):
            matched_skills.append(ps)
            
    # Take a subset of unique matched skills to list (no hallucination!)
    actual_skills_str = ", ".join(list(set(matched_skills))[:3])
    
    # Identify company backgrounds
    consulting_companies = []
    product_companies = []
    consulting_targets = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl', 'tech mahindra'}
    
    for role in history:
        comp_name = role.get('company', '').lower()
        if any(c in comp_name for c in consulting_targets):
            consulting_companies.append(role.get('company'))
        else:
            product_companies.append(role.get('company'))
            
    # Notice & location strings
    notice_str = f"immediate {notice}-day notice" if notice <= 30 else f"{notice}-day notice period"
    loc_str = f"based in {loc.split(',')[0].strip()}" if loc else ""
    
    # 2. Build Rank-Specific Reasoning Blocks
    reasons_list = []
    
    if rank <= 15:
        # Glowing tone
        intro = f"Outstanding candidate with {yoe} YOE, currently in a relevant domain role as {title} at {company}."
        
        skills_phrase = ""
        if actual_skills_str:
            skills_phrase = f" Demonstrates deep expertise in core technologies like {actual_skills_str}."
            
        career_phrase = ""
        if product_companies:
            career_phrase = f" Strong track record at product engineering firms."
            
        logistics_phrase = f" Strong availability with {notice_str}."
        if loc_str:
            logistics_phrase += f" Currently {loc_str}."
            
        reasoning = f"{intro}{skills_phrase}{career_phrase} {logistics_phrase}"
        
    elif rank <= 60:
        # Balanced tone
        intro = f"Solid {title} with {yoe} YOE, currently at {company} matching the senior engineering profile."
        
        skills_phrase = ""
        if actual_skills_str:
            skills_phrase = f" Has hands-on experience utilizing {actual_skills_str}."
            
        career_phrase = ""
        if consulting_companies and not product_companies:
            career_phrase = " Career history leans towards consulting, but tech skills are sound."
        elif product_companies:
            career_phrase = " Product-focused experience aligns well."
            
        logistics_phrase = f" Logistics: {notice_str}."
        if loc_str:
            logistics_phrase += f" Currently {loc_str}."
            
        reasoning = f"{intro}{skills_phrase}{career_phrase}{logistics_phrase}"
        
    else:
        # Highlighting Gaps & Concerns
        intro = f"Competent developer as {title} with {yoe} YOE."
        
        skills_phrase = ""
        if actual_skills_str:
            skills_phrase = f" Demonstrates familiarity with {actual_skills_str}."
            
        gaps = []
        if yoe < 5.0 or yoe > 9.0:
            gaps.append(f"experience ({yoe} yrs) is outside the sweet spot")
        if notice > 60:
            gaps.append(f"notice period of {notice} days")
        if not actual_skills_str:
            gaps.append("lack of direct retrieval/ranking experience")
        if consulting_companies:
            gaps.append("career background has consulting tenure")
            
        gaps_str = f" Minor concerns: {', '.join(gaps)}." if gaps else ""
        
        reasoning = f"{intro}{skills_phrase}{gaps_str}"
        
    # Clean up duplicate whitespace
    reasoning = " ".join(reasoning.split())
    
    return reasoning
