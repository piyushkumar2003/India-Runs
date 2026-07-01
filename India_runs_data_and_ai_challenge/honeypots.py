import json
from datetime import datetime

def is_honeypot(cand):
    profile = cand.get('profile', {})
    history = cand.get('career_history', [])
    skills = cand.get('skills', [])
    
    # 1. Expert skills with 0 months duration
    for s in skills:
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
            return True
            
    # 2. Date vs duration_months mismatch in career history
    # (Allowing up to 6 months margin for formatting/rounding differences)
    for item in history:
        start = item.get('start_date')
        end = item.get('end_date')
        dur = item.get('duration_months', 0)
        if start:
            try:
                sd = datetime.strptime(start, '%Y-%m-%d')
                if end:
                    ed = datetime.strptime(end, '%Y-%m-%d')
                else:
                    ed = datetime(2026, 6, 30) # current mock/reference date
                expected_dur = (ed.year - sd.year) * 12 + (ed.month - sd.month)
                if abs(expected_dur - dur) > 6:
                    return True
            except Exception:
                pass
                
    # 3. YOE vs total career history duration mismatch
    # If years_of_experience in profile is significantly lower or higher than history
    total_history_months = sum(item.get('duration_months', 0) for item in history)
    profile_yoe = profile.get('years_of_experience', 0)
    expected_months = profile_yoe * 12
    
    if expected_months > 12:
        if total_history_months < 0.5 * expected_months:
            return True
        if total_history_months > 1.5 * expected_months + 12:
            return True
            
    return False
