"""
LLM-Enhanced Resume Matcher
Combines vector similarity search with LLM reasoning for better matching
"""
from resume_matcher import ResumeMatcher
from typing import List, Tuple, Dict
import json
import os
import re


class LLMEnhancedMatcher(ResumeMatcher):
    """Enhanced matcher that uses LLM for reranking and explanation."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", use_llm=False, llm_provider="huggingface", api_key=None):
        """
        Initialize the enhanced matcher.
        
        Args:
            model_name: SentenceTransformer model name
            use_llm: Whether to use LLM for reranking
            llm_provider: LLM provider ("huggingface", "openai", "anthropic")
            api_key: API key for LLM provider (optional, can use env var)
        """
        super().__init__(model_name)
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.api_key = api_key
    
    def _get_llm_client(self):
        """Get LLM client based on provider."""
        if self.llm_provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                return InferenceClient
            except ImportError:
                print("⚠ Hugging Face not installed. Install with: pip install huggingface_hub")
                return None
        elif self.llm_provider == "openai":
            try:
                import openai
                return openai
            except ImportError:
                print("⚠ OpenAI not installed. Install with: pip install openai")
                return None
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                return anthropic
            except ImportError:
                print("⚠ Anthropic not installed. Install with: pip install anthropic")
                return None
        else:
            print(f"⚠ LLM provider '{self.llm_provider}' not supported")
            return None
    
    def llm_rerank_candidates(self, jd_text: str, candidates: List[Tuple[str, float, str]], 
                             top_k: int = 5) -> List[Tuple[str, float, str, str]]:
        """
        Use LLM to rerank and explain candidate matches.
        Supports Hugging Face, OpenAI, and Anthropic.
        
        Args:
            jd_text: Job description
            candidates: List of (filename, score, text) tuples
            top_k: Number of top candidates to return
            
        Returns:
            List of (filename, score, text, explanation) tuples
        """
        if not self.use_llm:
            return [(c[0], c[1], c[2], "Vector similarity score") for c in candidates[:top_k]]
        
        # Prepare candidate summaries for LLM
        candidate_summaries = []
        num_candidates_for_llm = min(10, len(candidates))
        for i, (filename, score, text) in enumerate(candidates[:num_candidates_for_llm], 1):
            # Extract key info from resume (first 500 chars)
            preview = text[:500].replace('\n', ' ')
            candidate_summaries.append(
                f"{i}. {filename} (Vector Score: {score:.2f}%)\n"
                f"   Preview: {preview}...\n"
            )
        
        # Create LLM prompt
        prompt = f"""You are an expert HR recruiter. Analyze these job candidates and rank them based on how well they match the job description.

JOB DESCRIPTION:
{jd_text}

CANDIDATES:
{''.join(candidate_summaries)}

Instructions:
1. Rank the candidates from BEST to WORST match (1-{num_candidates_for_llm})
2. Provide a brief explanation (1-2 sentences) for each candidate explaining why they are a good or bad fit
3. Consider: skills match, experience level, role alignment, qualifications

Format your response as JSON:
{{
  "rankings": [
    {{"rank": 1, "filename": "resume1.pdf", "match_score": 85, "explanation": "Excellent match because..."}},
    {{"rank": 2, "filename": "resume2.pdf", "match_score": 72, "explanation": "Good match because..."}}
  ]
}}
"""
        
        # Call LLM based on provider
        try:
            llm_client_class = self._get_llm_client()
            if llm_client_class is None:
                return [(c[0], c[1], c[2], "LLM not available") for c in candidates[:top_k]]
            
            result_text = None
            
            # Hugging Face Inference API (FREE!)
            if self.llm_provider == "huggingface":
                hf_token = self.api_key or os.getenv("HUGGINGFACE_API_TOKEN", "")
                
                # HF Inference API requires authentication
                if not hf_token:
                    raise Exception(
                        "Hugging Face API token required. "
                        "Get a free token at https://huggingface.co/settings/tokens or "
                        "set HUGGINGFACE_API_TOKEN environment variable."
                    )
                
                # Initialize client with token
                print(f"[DEBUG] Initializing HF client with token: {hf_token[:10]}..." if hf_token else "[DEBUG] No HF token provided")
                client = llm_client_class(token=hf_token)
                print(f"[DEBUG] HF client initialized successfully")
                
                # Try multiple models in order of preference
                # Instruction-tuned models use chat_completion, base models use text_generation
                models_to_try = [
                    {
                        "name": "mistralai/Mistral-7B-Instruct-v0.2",
                        "api": "chat",  # Use chat_completion API
                        "format": "mistral"
                    },
                    {
                        "name": "meta-llama/Llama-2-7b-chat-hf",
                        "api": "chat",  # Use chat_completion API
                        "format": "llama"
                    },
                    {
                        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
                        "api": "chat",  # Use chat_completion API
                        "format": "llama3"
                    },
                    {
                        "name": "google/flan-t5-xxl",
                        "api": "text",  # Use text_generation API
                        "format": "simple"
                    },
                    {
                        "name": "microsoft/DialoGPT-large",
                        "api": "text",  # Use text_generation API
                        "format": "simple"
                    },
                ]
                
                last_error = None
                result_text = None
                
                for model_info in models_to_try:
                    model_name = model_info["name"]
                    api_type = model_info.get("api", "chat")  # Default to chat for instruction models
                    format_type = model_info["format"]
                    
                    try:
                        print(f"[DEBUG] Trying model: {model_name} with API: {api_type}")
                        
                        if api_type == "chat":
                            # Use chat_completion API for instruction-tuned models
                            messages = [
                                {"role": "system", "content": "You are an expert HR recruiter. Always respond with valid JSON."},
                                {"role": "user", "content": prompt}
                            ]
                            
                            response = client.chat_completion(
                                messages=messages,
                                model=model_name,
                                max_tokens=1500,
                                temperature=0.3
                            )
                            
                            # Extract text from chat completion response
                            result_text = response.choices[0].message.content if response.choices else None
                            
                        else:
                            # Use text_generation API for base models
                            result_text = client.text_generation(
                                prompt,
                                model=model_name,
                                max_new_tokens=1500,
                                temperature=0.3,
                                return_full_text=False
                            )
                        
                        print(f"[DEBUG] ✓ Model {model_name} succeeded! Response length: {len(result_text) if result_text else 0}")
                        
                        # If we get here, the model worked!
                        if result_text and len(result_text.strip()) > 0:
                            print(f"[DEBUG] ✓✓ Using model {model_name} for LLM reranking")
                            break
                        else:
                            result_text = None
                            print(f"[DEBUG] ⚠ Model {model_name} returned empty response, trying next...")
                            continue
                        
                    except Exception as e:
                        error_str = str(e).lower()
                        error_type = type(e).__name__
                        error_repr = repr(e)
                        
                        # Log the actual error for debugging
                        print(f"[DEBUG] Model {model_name} failed with {error_type}: {error_str}")
                        print(f"[DEBUG] Full error: {error_repr}")
                        
                        # Skip if model doesn't support text-generation
                        if "not supported for task text-generation" in error_str or "text-generation" in error_str:
                            last_error = e
                            continue
                        # For other errors, also continue but log
                        last_error = e
                        continue
                
                # If all models failed
                if result_text is None or len(result_text.strip()) == 0:
                    error_detail = "No response from any model"
                    if last_error:
                        # Try to extract useful info from error
                        try:
                            error_type = type(last_error).__name__
                            error_detail = str(last_error)
                            error_repr = repr(last_error)
                            
                            print(f"[DEBUG] Last error type: {error_type}")
                            print(f"[DEBUG] Last error str: {error_detail}")
                            print(f"[DEBUG] Last error repr: {error_repr}")
                            
                            # Check if it's an HF API error with response object
                            if hasattr(last_error, 'response'):
                                try:
                                    response = last_error.response
                                    print(f"[DEBUG] Error has response object: {type(response)}")
                                    
                                    if hasattr(response, 'text'):
                                        error_detail = str(response.text)
                                        print(f"[DEBUG] Response text: {error_detail[:200]}")
                                    elif hasattr(response, 'json'):
                                        try:
                                            error_data = response.json()
                                            print(f"[DEBUG] Response JSON: {error_data}")
                                            error_detail = error_data.get('error', error_data.get('message', str(last_error)))
                                        except:
                                            error_detail = str(last_error)
                                    elif hasattr(response, 'status_code'):
                                        error_detail = f"HTTP {response.status_code}: {error_detail}"
                                        print(f"[DEBUG] Status code: {response.status_code}")
                                except Exception as resp_error:
                                    print(f"[DEBUG] Error extracting response: {resp_error}")
                                    error_detail = str(last_error)
                            
                            # Check for common HF API error patterns in the error message
                            error_lower = error_detail.lower()
                            if "401" in error_detail or "unauthorized" in error_lower or "authentication" in error_lower:
                                error_detail = f"Authentication failed - invalid or missing token. Original: {error_detail}"
                            elif "403" in error_detail or "forbidden" in error_lower or "access denied" in error_lower:
                                error_detail = f"Access forbidden - may need to accept model terms. Original: {error_detail}"
                            elif "429" in error_detail or "rate limit" in error_lower:
                                error_detail = f"Rate limit exceeded. Original: {error_detail}"
                            elif "404" in error_detail or "not found" in error_lower:
                                error_detail = f"Model not found. Original: {error_detail}"
                            elif "500" in error_detail or "internal server" in error_lower:
                                error_detail = f"HF API server error. Original: {error_detail}"
                            elif "timeout" in error_lower:
                                error_detail = f"Request timeout. Original: {error_detail}"
                            elif not error_detail or error_detail.strip() == "":
                                error_detail = f"Unknown error occurred. Error type: {error_type}, Repr: {error_repr}"
                        except Exception as parse_error:
                            print(f"[DEBUG] Error parsing exception: {parse_error}")
                            error_detail = f"Error parsing exception: {str(parse_error)}. Original: {str(last_error)}"
                    
                    error_msg = f"All HF models failed. Last error: {error_detail}"
                    
                    # Add helpful tips based on error type
                    if "401" in error_detail or "unauthorized" in error_detail.lower() or "authentication" in error_detail.lower():
                        error_msg += "\n💡 Tip: Check your Hugging Face token - it may be invalid or missing."
                    elif "403" in error_detail or "forbidden" in error_detail.lower() or "access" in error_detail.lower():
                        error_msg += "\n💡 Tip: You may need to accept model terms at huggingface.co"
                    elif "429" in error_detail or "rate limit" in error_detail.lower():
                        error_msg += "\n💡 Tip: Rate limit exceeded. Wait a moment and try again."
                    elif "token" in error_detail.lower():
                        error_msg += "\n💡 Tip: Make sure your Hugging Face token is set correctly in the sidebar."
                    else:
                        error_msg += "\n💡 Tip: Make sure your token has access to the models. Some models may require accepting terms on Hugging Face."
                    
                    print(f"[DEBUG] HF LLM reranking failed with error: {error_detail}")
                    raise Exception(error_msg)
            
            # OpenAI
            elif self.llm_provider == "openai":
                openai_api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
                if not openai_api_key:
                    raise Exception("OPENAI_API_KEY not set")
                
                openai_client = llm_client_class(api_key=openai_api_key)
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert HR recruiter. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                result_text = response.choices[0].message.content
            
            # Anthropic
            elif self.llm_provider == "anthropic":
                anthropic_api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
                if not anthropic_api_key:
                    raise Exception("ANTHROPIC_API_KEY not set")
                
                client = llm_client_class(api_key=anthropic_api_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    temperature=0.3,
                    system="You are an expert HR recruiter. Always respond with valid JSON.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text
            
            else:
                return [(c[0], c[1], c[2], f"Provider {self.llm_provider} not supported") for c in candidates[:top_k]]
            
            # Parse LLM response
            if result_text:
                try:
                    # Extract JSON from response (in case there's extra text)
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                    else:
                        result = json.loads(result_text)
                    
                    # Reorder candidates based on LLM ranking
                    ranked_results = []
                    rankings = result.get("rankings", [])
                    
                    # Create a mapping for quick lookup
                    filename_to_ranking = {item["filename"]: item for item in rankings}
                    
                    # Process ranked items
                    for item in rankings[:top_k]:
                        filename = item["filename"]
                        # Find original candidate
                        original = next((c for c in candidates if c[0] == filename), None)
                        if original:
                            ranked_results.append((
                                original[0],
                                item.get("match_score", original[1]),
                                original[2],
                                item.get("explanation", "")
                            ))
                    
                    # Add any remaining candidates that weren't ranked by LLM
                    ranked_filenames = {r[0] for r in ranked_results}
                    for candidate in candidates:
                        if candidate[0] not in ranked_filenames and len(ranked_results) < top_k:
                            ranked_results.append((candidate[0], candidate[1], candidate[2], "Vector similarity score"))
                    
                    return ranked_results[:top_k]
                    
                except json.JSONDecodeError as e:
                    print(f"⚠ LLM response not in valid JSON format: {str(e)}")
                    print(f"Response preview: {result_text[:200]}...")
                    # Fallback: try to extract explanations manually
                    return [(c[0], c[1], c[2], f"LLM parsing error - {result_text[:100]}") for c in candidates[:top_k]]
            else:
                return [(c[0], c[1], c[2], "Empty LLM response") for c in candidates[:top_k]]
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"⚠ LLM reranking failed ({error_type}): {error_msg}")
            
            # Provide helpful error messages
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                error_msg = "Authentication failed. Check your Hugging Face token."
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                error_msg = "Access forbidden. The model may require a token or access request."
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                error_msg = "Rate limit exceeded. Please wait a moment and try again."
            elif "404" in error_msg or "not found" in error_msg.lower():
                error_msg = "Model not found. The model may not be available."
            
            return [(c[0], c[1], c[2], f"LLM Error: {error_msg}") for c in candidates[:top_k]]
    
    def llm_detect_role(self, text: str) -> str:
        """
        Use LLM to dynamically detect the primary role/job title from text.
        More accurate than keyword-based detection.
        
        Args:
            text: Text to analyze (JD or resume)
            
        Returns:
            Detected role (e.g., 'software developer', 'product manager', 'data scientist')
        """
        try:
            llm_client_class = self._get_llm_client()
            if llm_client_class is None or self.llm_provider != "huggingface":
                # Fallback to keyword-based if LLM not available
                return None
            
            # Extract first 500 characters (usually contains role info)
            text_preview = text[:500]
            
            prompt = f"""Extract the PRIMARY job role or job title from the following text.

Text:
{text_preview}

Respond with ONLY the job role/title in 2-3 words maximum. Examples: "software developer", "product manager", "data scientist", "sales manager", "marketing manager", "accountant".

If no clear role is found, respond with "unknown".

Role:"""
            
            # Try Hugging Face chat API first
            hf_token = self.api_key or os.getenv("HUGGINGFACE_API_TOKEN", "")
            if not hf_token:
                return None
            
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            
            # Try chat completion API
            try:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                response = client.chat_completion(
                    messages=messages,
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    max_tokens=20,
                    temperature=0.1  # Low temperature for consistent extraction
                )
                
                role = response.choices[0].message.content.strip().lower()
                
                # Clean up response
                role = role.replace('"', '').replace("'", '').strip()
                if role.startswith('role:'):
                    role = role[5:].strip()
                if role == 'unknown' or not role:
                    return None
                    
                return role
                
            except Exception as e:
                print(f"[DEBUG] LLM role detection failed: {e}")
                return None
                
        except Exception as e:
            print(f"[DEBUG] LLM role detection error: {e}")
            return None
    
    def llm_extract_requirements(self, jd_text: str) -> Dict:
        """
        Use LLM to extract structured requirements from job description.
        
        Args:
            jd_text: Job description text
            
        Returns:
            Dictionary with structured requirements
        """
        if not self.use_llm:
            return {}
        
        prompt = f"""Extract structured requirements from this job description:

{jd_text}

Extract:
1. Job title/role
2. Required skills (list)
3. Years of experience required
4. Education requirements
5. Preferred qualifications
6. Key responsibilities

Format as JSON:
{{
  "job_title": "...",
  "required_skills": ["skill1", "skill2"],
  "experience_years": 5,
  "education": "...",
  "preferred_qualifications": ["..."],
  "responsibilities": ["..."]
}}
"""
        
        try:
            llm = self._get_llm_client()
            if llm is None:
                return {}
            
            if self.llm_provider == "openai":
                response = llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract structured information. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                result_text = response.choices[0].message.content
                
                # Parse JSON
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return json.loads(result_text)
        except Exception as e:
            print(f"⚠ LLM extraction failed: {str(e)}")
            return {}
    
    def enhanced_match(self, jd_text: str, resumes: List[Tuple[str, str]], 
                      top_k: int = 5, use_llm_reranking: bool = False) -> List[Tuple[str, float, str, str]]:
        """
        Enhanced matching with optional LLM reranking.
        
        Args:
            jd_text: Job description
            resumes: List of (filename, text) tuples
            top_k: Number of top candidates
            use_llm_reranking: Whether to use LLM for reranking
            
        Returns:
            List of (filename, score, text, explanation) tuples
        """
        # Step 1: Vector similarity search (fast initial filtering)
        print("Step 1: Vector similarity search...")
        vector_results = self.compute_match_scores(jd_text, resumes)
        
        # Step 2: Optional LLM reranking
        if use_llm_reranking and self.use_llm:
            print(f"Step 2: LLM reranking top candidates...")
            ranked_results = self.llm_rerank_candidates(jd_text, vector_results, top_k=top_k)
        else:
            # Use vector results with simple explanations
            ranked_results = [
                (r[0], r[1], r[2], f"Vector similarity: {r[1]:.2f}%")
                for r in vector_results[:top_k]
            ]
        
        return ranked_results


def demo_llm_matching():
    """Demo of LLM-enhanced matching."""
    from resume_parser import load_resumes_from_directory
    
    print("="*80)
    print("LLM-Enhanced Resume Matching Demo")
    print("="*80)
    
    # Initialize matcher (set use_llm=True to enable LLM features)
    matcher = LLMEnhancedMatcher(use_llm=False)  # Set to True if you have API key
    
    # Load job description
    jd_path = "data/jd.txt"
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read().strip()
    
    # Load resumes
    resumes = load_resumes_from_directory("data")
    
    print(f"\n📄 Job Description: {jd_path}")
    print(f"📁 Resumes loaded: {len(resumes)}\n")
    
    # Enhanced matching
    results = matcher.enhanced_match(
        jd_text, 
        resumes, 
        top_k=5,
        use_llm_reranking=False  # Set to True to enable LLM reranking
    )
    
    # Display results
    print("\n" + "="*80)
    print("TOP MATCHED CANDIDATES")
    print("="*80 + "\n")
    
    for rank, (filename, score, text, explanation) in enumerate(results, 1):
        print(f"Rank {rank}: {filename}")
        print(f"Match Score: {score:.2f}%")
        print(f"Explanation: {explanation}")
        print(f"Preview: {text[:150]}...")
        print("-"*80)


if __name__ == "__main__":
    demo_llm_matching()

