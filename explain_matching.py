"""
Explanation of why certain resumes match a job description
Demonstrates the matching algorithm behavior
"""
from resume_matcher import ResumeMatcher
from resume_parser import load_resumes_from_directory
from pathlib import Path

def analyze_matching(jd_path="data/jd.txt", resume_dir="data"):
    """Analyze why certain resumes match the job description."""
    
    matcher = ResumeMatcher()
    
    # Load job description
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read().strip()
    
    print("="*80)
    print("MATCHING ANALYSIS")
    print("="*80)
    print(f"\nJob Description ({len(jd_text)} characters):")
    print("-"*80)
    print(jd_text)
    print("-"*80)
    
    # Load resumes
    resumes = load_resumes_from_directory(resume_dir)
    
    print(f"\n📊 Analyzing {len(resumes)} resume(s)...\n")
    
    # Compute matches
    results = matcher.compute_match_scores(jd_text, resumes)
    
    # Show top matches with explanations
    print("\n" + "="*80)
    print("TOP MATCHES WITH ANALYSIS")
    print("="*80 + "\n")
    
    for rank, (filename, score, text) in enumerate(results[:5], 1):
        print(f"Rank {rank}: {filename}")
        print(f"Match Score: {score:.2f}%")
        
        # Analyze why it might match
        jd_lower = jd_text.lower()
        resume_lower = text.lower()
        
        # Check for key terms from JD
        jd_keywords = ['sales', 'manager', 'b2b', 'crm', 'salesforce', 
                      'lead generation', 'enterprise', 'accounts', 
                      'pipeline', 'deal', 'closure']
        
        found_keywords = []
        for keyword in jd_keywords:
            if keyword in resume_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            print(f"  ⚠ Found JD keywords: {', '.join(found_keywords)}")
        else:
            print(f"  ⚠ No JD-specific keywords found (likely generic similarity)")
        
        # Check role alignment
        resume_role_indicators = {
            'sales': ['sales', 'account executive', 'account manager', 'sales manager'],
            'marketing': ['marketing', 'brand', 'campaign', 'digital marketing'],
            'developer': ['developer', 'programmer', 'software', 'coding', 'programming'],
            'data': ['data scientist', 'analyst', 'machine learning', 'ml'],
            'devops': ['devops', 'infrastructure', 'cloud', 'kubernetes'],
            'it': ['it support', 'technical support', 'help desk']
        }
        
        detected_roles = []
        for role, indicators in resume_role_indicators.items():
            if any(ind in resume_lower for ind in indicators):
                detected_roles.append(role)
        
        if detected_roles:
            print(f"  📋 Detected roles in resume: {', '.join(detected_roles)}")
        
        # Check if role matches JD
        jd_is_sales = any(word in jd_lower for word in ['sales', 'manager'])
        resume_is_sales = 'sales' in detected_roles or 'marketing' in detected_roles
        
        if jd_is_sales and not resume_is_sales:
            print(f"  ❌ ROLE MISMATCH: JD is for Sales, but resume is for different role")
        elif jd_is_sales and resume_is_sales:
            print(f"  ✅ ROLE ALIGNMENT: Both JD and resume are sales/marketing related")
        else:
            print(f"  ⚠ UNKNOWN ROLE ALIGNMENT")
        
        print(f"  📝 Resume preview: {text[:150].replace(chr(10), ' ')}...")
        print()
    
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
1. PROBLEM: The job description is too short (only 4 lines)
   → SOLUTION: Use more detailed job descriptions (50+ words) for better matching

2. PROBLEM: Threshold too low (40-50%)
   → SOLUTION: Increase threshold to 60-70% for stricter filtering

3. PROBLEM: Generic word overlap causing false positives
   → SOLUTION: Use cosine similarity (already implemented) + role-based filtering

4. PROBLEM: No role-specific filtering
   → SOLUTION: Add keyword-based role detection and filter mismatches

5. IMPROVEMENT: Use semantic role matching
   → Add logic to detect if resume role matches JD role before computing similarity
    """)

if __name__ == "__main__":
    analyze_matching()

