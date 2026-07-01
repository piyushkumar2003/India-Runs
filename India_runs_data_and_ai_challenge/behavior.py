import json
from datetime import datetime

# Reference date for the dataset activity timeline
REF_DATE = datetime(2026, 6, 30)

def compute_behavioral_multiplier(cand):
    signals = cand.get('redrob_signals', {})
    profile = cand.get('profile', {})
    
    multiplier = 1.0
    
    # 1. Profile completeness score
    completeness = signals.get('profile_completeness_score', 100.0)
    if completeness < 50.0:
        multiplier *= 0.85
    elif completeness > 85.0:
        multiplier *= 1.05
        
    # 2. Active recency check
    last_active_str = signals.get('last_active_date')
    if last_active_str:
        try:
            la_date = datetime.strptime(last_active_str, '%Y-%m-%d')
            days_inactive = (REF_DATE - la_date).days
            if days_inactive <= 30:
                multiplier *= 1.10
            elif days_inactive > 180:
                multiplier *= 0.60
            elif days_inactive > 90:
                multiplier *= 0.80
        except Exception:
            pass
            
    # 3. Recruiter response rate & response time
    response_rate = signals.get('recruiter_response_rate', 1.0)
    if response_rate < 0.20:
        multiplier *= 0.60
    elif response_rate > 0.80:
        multiplier *= 1.15
        
    avg_resp_time = signals.get('avg_response_time_hours', 24.0)
    if avg_resp_time < 6.0:
        multiplier *= 1.05
    elif avg_resp_time > 48.0:
        multiplier *= 0.90
        
    # 4. Notice period check
    notice_days = signals.get('notice_period_days', 90)
    if notice_days <= 15:
        multiplier *= 1.20  # Immediate joiner
    elif notice_days <= 30:
        multiplier *= 1.10  # Short notice
    elif notice_days > 90:
        multiplier *= 0.75  # Long notice hurdle
    elif notice_days > 60:
        multiplier *= 0.90
        
    # 5. Location and Relocation fit check
    location_str = (profile.get('location', '') + ' ' + profile.get('country', '')).lower()
    preferred_cities = ['pune', 'noida', 'delhi', 'ncr', 'gurgaon', 'ghaziabad', 'faridabad', 'hyderabad', 'mumbai']
    is_preferred_loc = any(city in location_str for city in preferred_cities)
    is_india = 'india' in location_str or profile.get('country', '').lower() == 'india'
    
    willing_relocate = signals.get('willing_to_relocate', False)
    
    if not is_india:
        if not willing_relocate:
            multiplier *= 0.40  # Non-India and won't relocate
        else:
            multiplier *= 0.70  # Needs visa / remote relocation, harder
    else:
        if is_preferred_loc:
            multiplier *= 1.05  # Perfect match
        elif not willing_relocate:
            multiplier *= 0.75  # In India but unwilling to relocate to office hubs
            
    # 6. Open to work flag
    if signals.get('open_to_work_flag', False):
        multiplier *= 1.10
        
    # 7. Interview completion rate (reliability)
    icr = signals.get('interview_completion_rate', 1.0)
    if icr < 0.50:
        multiplier *= 0.70  # High ghosting risk
    elif icr > 0.85:
        multiplier *= 1.05
        
    # 8. GitHub and Recruiter Interest
    github_score = signals.get('github_activity_score', -1.0)
    if github_score > 50.0:
        multiplier *= 1.05
        
    saved_recruiters = signals.get('saved_by_recruiters_30d', 0)
    if saved_recruiters > 15:
        multiplier *= 1.05
        
    # 9. Profile Verification checks
    if signals.get('verified_email', False) and signals.get('verified_phone', False):
        multiplier *= 1.05
        
    # 10. Expected Salary ROI Check (Idea B)
    expected_salary = signals.get('expected_salary_range_inr_lpa', {})
    expected_min = expected_salary.get('min', 0.0)
    yoe = profile.get('years_of_experience', 0.0)
    if expected_min > 0.0 and yoe > 0.0:
        # Define baseline market LPA based on YOE curve
        if yoe <= 2.0:
            baseline = 10.0
        elif yoe <= 5.0:
            baseline = 10.0 + (yoe - 2.0) * 4.0
        elif yoe <= 9.0:
            baseline = 22.0 + (yoe - 5.0) * 6.0
        else:
            baseline = 46.0 + (yoe - 9.0) * 3.0
            
        # Check ROI
        if expected_min < baseline * 0.8:
            multiplier *= 1.08  # High value / Bargain candidate
        elif expected_min > baseline * 1.4:
            multiplier *= 0.88  # Unreasonable expectation premium
            
    return multiplier
