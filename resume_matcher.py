"""
Enhanced Resume Matcher using Vector Similarity Search (FAISS)
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Tuple
from resume_parser import parse_resume, load_resumes_from_directory


class ResumeMatcher:
    """Main class for matching resumes against job descriptions using vector similarity."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the ResumeMatcher with an embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        # Load model silently (progress shown in UI)
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise Exception(f"Failed to load embedding model '{model_name}': {str(e)}")
    
    def load_job_description(self, jd_path):
        """
        Load job description from a text file.
        
        Args:
            jd_path: Path to job description file
            
        Returns:
            Job description text
        """
        try:
            with open(jd_path, 'r', encoding='utf-8') as f:
                jd_text = f.read().strip()
            if not jd_text:
                raise Exception("Job description file is empty")
            return jd_text
        except FileNotFoundError:
            raise Exception(f"Job description file not found: {jd_path}")
        except Exception as e:
            raise Exception(f"Error loading job description: {str(e)}")
    
    def load_resumes(self, resume_directory=None, resume_files=None):
        """
        Load resumes from directory or list of files.
        
        Args:
            resume_directory: Directory containing resume files
            resume_files: List of file paths to resume files
            
        Returns:
            List of tuples: (filename, text_content)
        """
        resumes = []
        
        if resume_directory:
            resumes = load_resumes_from_directory(resume_directory)
        
        if resume_files:
            for file_path in resume_files:
                try:
                    text = parse_resume(file_path)
                    filename = Path(file_path).name
                    resumes.append((filename, text))
                    print(f"✓ Loaded: {filename}")
                except Exception as e:
                    print(f"✗ Error loading {file_path}: {str(e)}")
        
        if not resumes:
            raise ValueError("No resumes loaded. Check file paths or directory.")
        
        return resumes
    
    def compute_match_scores(self, jd_text: str, resumes: List[Tuple[str, str]]) -> List[Tuple[str, float, str]]:
        """
        Compute similarity scores between job description and resumes.
        
        Args:
            jd_text: Job description text
            resumes: List of (filename, text) tuples
            
        Returns:
            List of (filename, match_score, text) tuples sorted by score (descending)
        """
        if not resumes:
            return []
        
        # Generate embeddings
        print(f"\n[DEBUG] Generating embeddings for {len(resumes)} resume(s)...")
        try:
            print(f"[DEBUG] Encoding job description...")
            jd_embedding = self.model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
            print(f"[DEBUG] Encoding {len(resumes)} resumes...")
            resume_texts = [resume[1] for resume in resumes]
            resume_embeddings = self.model.encode(resume_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
            print(f"[DEBUG] Embeddings generated successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to generate embeddings: {str(e)}")
            raise Exception(f"Error generating embeddings: {str(e)}")
        
        # Use Inner Product for cosine similarity (since embeddings are normalized)
        # Cosine similarity = dot product when vectors are normalized
        print(f"[DEBUG] Creating FAISS index...")
        try:
            dimension = resume_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            index.add(resume_embeddings)
            print(f"[DEBUG] FAISS index created with {len(resumes)} vectors")
        except Exception as e:
            print(f"[ERROR] Failed to create FAISS index: {str(e)}")
            raise Exception(f"Error creating FAISS index: {str(e)}")
        
        # Search for similar resumes
        print(f"[DEBUG] Computing similarity scores...")
        try:
            k = len(resumes)
            similarities, indices = index.search(jd_embedding, k)
            print(f"[DEBUG] Similarity search completed!")
        except Exception as e:
            print(f"[ERROR] Failed to compute similarity: {str(e)}")
            raise Exception(f"Error computing similarity scores: {str(e)}")
        
        # Calculate match scores and create results
        results = []
        for rank, idx in enumerate(indices[0]):
            filename = resumes[idx][0]
            cosine_similarity = similarities[0][rank]
            
            # Convert cosine similarity (-1 to 1) to match score (0-100%)
            # Cosine similarity ranges from -1 to 1, but with normalized embeddings it's typically 0 to 1
            # Scale it to 0-100%
            match_score = max(0, cosine_similarity) * 100
            
            results.append((filename, match_score, resumes[idx][1]))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def detect_role_category(self, text: str, strict: bool = True, use_llm: bool = False, llm_matcher=None) -> List[str]:
        """
        Detect role categories in text based on keywords or LLM.
        
        Args:
            text: Text to analyze
            strict: If True, prioritize title/header mentions over generic keywords
            use_llm: If True and llm_matcher provided, use LLM for role detection
            llm_matcher: LLMEnhancedMatcher instance for LLM-based detection
            
        Returns:
            List of detected role categories
        """
        # Try LLM-based detection first if enabled
        if use_llm and llm_matcher:
            try:
                llm_role = llm_matcher.llm_detect_role(text)
                if llm_role and llm_role != 'unknown':
                    print(f"[DEBUG] LLM detected role: {llm_role}")
                    # Map LLM role to category
                    role_category = self._map_role_to_category(llm_role)
                    if role_category:
                        return [role_category]
            except Exception as e:
                print(f"[DEBUG] LLM role detection failed, falling back to keywords: {e}")
        
        # Fallback to keyword-based detection
        text_lower = text.lower()
        roles = []
        
        # Extract first few lines (usually contains title/role)
        lines = text.split('\n')[:10]
        header_text = ' '.join(lines).lower()
        
        # Primary role indicators (appear in title/header)
        primary_role_keywords = {
            'sales': ['sales manager', 'account executive', 'account manager', 'sales representative',
                     'business development', 'bd manager', 'sales specialist'],
            'marketing': ['marketing manager', 'marketing specialist', 'brand manager', 
                         'digital marketing', 'marketing coordinator'],
            'developer': ['software developer', 'software engineer', 'frontend developer',
                         'backend developer', 'full stack developer', 'web developer',
                         'mobile developer', 'application developer', 'backend engineer',
                         'frontend engineer', 'fullstack engineer'],
            'data': ['data scientist', 'data analyst', 'data engineer', 'machine learning engineer',
                    'ml engineer', 'data science', 'ml specialist', 'ai engineer'],
            'product': ['product manager', 'product owner', 'senior product manager',
                       'associate product manager', 'product lead'],
            'accountant': ['accountant', 'cpa', 'certified public accountant', 'senior accountant',
                          'financial accountant', 'cost accountant', 'tax accountant'],
            'devops': ['devops engineer', 'site reliability engineer', 'sre', 
                      'infrastructure engineer', 'cloud engineer'],
            'it_support': ['it support', 'technical support specialist', 'help desk',
                          'it technician', 'system administrator']
        }
        
        # Secondary role indicators (general keywords, less reliable)
        secondary_role_keywords = {
            'sales': ['sales', 'revenue', 'quota', 'b2b sales', 'enterprise sales', 'closing deals'],
            'marketing': ['marketing', 'brand', 'campaign', 'seo', 'sem', 'digital marketing'],
            'developer': ['developer', 'programmer', 'coding', 'programming', 'software development',
                         'backend', 'frontend', 'fullstack'],
            'data': ['data analysis', 'statistical analysis', 'machine learning', 'data pipeline',
                    'data modeling', 'predictive analytics'],
            'product': ['product management', 'product strategy', 'roadmap', 'product owner',
                       'product requirements', 'user stories', 'agile product'],
            'accountant': ['accounting', 'financial reporting', 'bookkeeping', 'tax preparation',
                          'audit', 'general ledger', 'accounts payable', 'accounts receivable'],
            'devops': ['devops', 'kubernetes', 'docker', 'ci/cd', 'infrastructure', 'deployment'],
            'it_support': ['it support', 'technical support', 'help desk', 'troubleshooting', 'it help']
        }
        
        # Check primary indicators first (from header/title)
        for role, keywords in primary_role_keywords.items():
            if any(keyword in header_text for keyword in keywords):
                roles.append(role)
                return [role]  # Return immediately if primary role found in header
        
        # If no primary role found, check secondary indicators (only if not strict)
        if not strict:
            for role, keywords in secondary_role_keywords.items():
                if role not in roles and any(keyword in text_lower for keyword in keywords):
                    roles.append(role)
        else:
            # In strict mode, if no clear role in header, try to infer from job titles
            job_title_patterns = {
                'sales': r'\b(sales|account executive|account manager|bd|business development)\b',
                'marketing': r'\b(marketing|brand manager|marketing specialist)\b',
                'developer': r'\b(developer|software engineer|backend engineer|frontend engineer|programmer)\b',
                'data': r'\b(data scientist|data analyst|ml engineer|data engineer)\b',
                'product': r'\b(product manager|product owner|product lead)\b',
                'accountant': r'\b(accountant|cpa|financial accountant)\b',
                'devops': r'\b(devops|sre|infrastructure engineer|cloud engineer)\b',
            }
            
            for role, pattern in job_title_patterns.items():
                if role not in roles and re.search(pattern, header_text):
                    roles.append(role)
        
        return roles if roles else ['unknown']
    
    def _map_role_to_category(self, role_text: str) -> str:
        """
        Map LLM-detected role text to a role category.
        
        Args:
            role_text: Role text from LLM (e.g., "software developer", "product manager")
            
        Returns:
            Role category (e.g., "developer", "product", "sales")
        """
        role_lower = role_text.lower()
        
        # Mapping rules
        if any(term in role_lower for term in ['product manager', 'product owner', 'product lead']):
            return 'product'
        elif any(term in role_lower for term in ['software developer', 'software engineer', 'backend', 'frontend', 'fullstack', 'full stack', 'web developer', 'application developer']):
            return 'developer'
        elif any(term in role_lower for term in ['data scientist', 'data analyst', 'data engineer', 'ml engineer', 'machine learning']):
            return 'data'
        elif any(term in role_lower for term in ['sales manager', 'sales representative', 'account executive', 'account manager', 'business development']):
            return 'sales'
        elif any(term in role_lower for term in ['marketing manager', 'marketing specialist', 'brand manager', 'digital marketing']):
            return 'marketing'
        elif any(term in role_lower for term in ['accountant', 'cpa', 'financial accountant']):
            return 'accountant'
        elif any(term in role_lower for term in ['devops', 'site reliability', 'sre', 'infrastructure engineer']):
            return 'devops'
        elif any(term in role_lower for term in ['it support', 'technical support', 'help desk']):
            return 'it_support'
        
        return None
    
    def filter_by_threshold(self, results: List[Tuple[str, float, str]], threshold: float = 50.0, 
                           jd_text: str = None, role_filter: bool = False, use_llm_role_detection: bool = False, 
                           llm_matcher=None):
        """
        Filter results by minimum match score threshold and optionally by role match.
        
        Args:
            results: List of (filename, score, text) tuples
            threshold: Minimum match score (0-100)
            jd_text: Job description text for role filtering
            role_filter: Whether to filter out role mismatches
            use_llm_role_detection: If True, use LLM for role detection (more accurate)
            llm_matcher: LLMEnhancedMatcher instance for LLM-based role detection
            
        Returns:
            Filtered list of results
        """
        # First filter by threshold
        filtered = [r for r in results if r[1] >= threshold]
        print(f"[DEBUG] After threshold ({threshold}%): {len(filtered)} candidates")
        
        if role_filter and jd_text:
            # Use LLM if enabled, otherwise use keyword-based
            jd_roles = self.detect_role_category(jd_text, strict=True, use_llm=use_llm_role_detection, llm_matcher=llm_matcher)
            print(f"[DEBUG] JD roles detected: {jd_roles}")
            
            if jd_roles and jd_roles != ['unknown']:
                # Define related roles (only closely related roles allowed)
                # Product manager can overlap with technical roles but shouldn't match pure developers
                related_roles = {
                    'sales': ['sales', 'marketing'],  # Sales and marketing are related
                    'marketing': ['sales', 'marketing'],
                    'developer': ['developer', 'devops'],  # Developers and DevOps are related
                    'data': ['data', 'developer'],  # Data and developers have some overlap
                    'devops': ['devops', 'developer'],
                    'product': [],  # Product manager is NOT related to developer - STRICT
                    'accountant': [],  # Accountant is NOT related to other roles - STRICT
                    'it_support': ['it_support'],  # IT support is standalone
                }
                
                # Expand JD roles to include related roles (if any)
                expanded_jd_roles = set(jd_roles)
                for role in jd_roles:
                    if role in related_roles:
                        expanded_jd_roles.update(related_roles[role])
                
                print(f"[DEBUG] Expanded JD roles (allowing): {expanded_jd_roles}")
                
                # Filter: Only keep resumes that have matching role category
                role_filtered = []
                for r in filtered:
                    resume_roles = self.detect_role_category(r[2], strict=True, use_llm=use_llm_role_detection, llm_matcher=llm_matcher)
                    print(f"[DEBUG] Resume {r[0]}: roles={resume_roles}")
                    
                    # Check if any resume role matches expanded JD roles
                    if any(resume_role in expanded_jd_roles for resume_role in resume_roles):
                        role_filtered.append(r)
                        print(f"[DEBUG]   ✓ KEPT (role match)")
                    else:
                        print(f"[DEBUG]   ✗ FILTERED OUT (role mismatch: {resume_roles} not in {expanded_jd_roles})")
                
                filtered = role_filtered
                print(f"[DEBUG] After role filtering: {len(filtered)} candidates")
            else:
                print(f"[DEBUG] JD role unknown, skipping role filter")
        
        return filtered
        
        return filtered
    
    def display_results(self, results: List[Tuple[str, float, str]], show_details=False):
        """
        Display ranked results in a formatted table.
        
        Args:
            results: List of (filename, score, text) tuples
            show_details: Whether to show full resume text
        """
        if not results:
            print("\nNo matching resumes found.")
            return
        
        print("\n" + "="*80)
        print("RANKED CANDIDATE RESULTS")
        print("="*80)
        print(f"\n{'Rank':<6} {'Candidate':<30} {'Match Score':<15}")
        print("-"*80)
        
        for rank, (filename, score, text) in enumerate(results, start=1):
            medal = ""
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"
            
            print(f"{medal} {rank:<4} {filename:<30} {score:.2f}%")
            
            if show_details:
                # Show first 200 characters of resume
                preview = text[:200].replace('\n', ' ')
                print(f"      Preview: {preview}...")
                print()
        
        print("-"*80)
        print(f"Total candidates: {len(results)}")
        print("="*80)


def main():
    """Example usage of ResumeMatcher."""
    import sys
    from pathlib import Path
    
    # Initialize matcher
    matcher = ResumeMatcher()
    
    # Default paths
    jd_path = "data/jd.txt"
    resume_dir = "data"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        jd_path = sys.argv[1]
    if len(sys.argv) > 2:
        resume_dir = sys.argv[2]
    
    # Load job description
    try:
        jd_text = matcher.load_job_description(jd_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    
    # Load resumes
    try:
        resumes = matcher.load_resumes(resume_directory=resume_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    
    # Compute matches
    results = matcher.compute_match_scores(jd_text, resumes)
    
    # Filter by threshold (optional)
    filtered_results = matcher.filter_by_threshold(results, threshold=30.0)
    
    # Display results
    matcher.display_results(filtered_results, show_details=False)
    
    # Save results to file
    output_file = "matching_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RESUME MATCHING RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Job Description: {jd_path}\n")
        f.write(f"Total Resumes Analyzed: {len(resumes)}\n")
        f.write(f"Candidates Above Threshold: {len(filtered_results)}\n\n")
        f.write("-"*80 + "\n\n")
        
        for rank, (filename, score, text) in enumerate(filtered_results, start=1):
            f.write(f"Rank {rank}: {filename}\n")
            f.write(f"Match Score: {score:.2f}%\n")
            f.write(f"Preview: {text[:300]}...\n\n")
            f.write("-"*80 + "\n\n")
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    from pathlib import Path
    main()

