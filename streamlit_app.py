"""
Streamlit Web UI for Resume Matching System
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
import time
from resume_matcher import ResumeMatcher
from llm_enhanced_matcher import LLMEnhancedMatcher
from resume_parser import parse_resume, load_resumes_from_directory
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION - Set your API tokens here
# ============================================================================
# Hugging Face API Token - Get one at: https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN = ""  # Replace with your actual token

# Optional: Other API tokens (if you want to use OpenAI or Anthropic)
OPENAI_API_KEY = ""  # Optional: Set if using OpenAI
ANTHROPIC_API_KEY = ""  # Optional: Set if using Anthropic

# ============================================================================
# Auto-set environment variables if token is provided
# ============================================================================
if HUGGINGFACE_API_TOKEN and HUGGINGFACE_API_TOKEN != "YOUR_HUGGINGFACE_TOKEN_HERE":
    os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# Page configuration
st.set_page_config(
    page_title="Resume Matcher AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .high-score {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-score {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-score {
        background-color: #f8d7da;
        color: #721c24;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matcher' not in st.session_state:
    st.session_state.matcher = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
if 'llm_explanations' not in st.session_state:
    st.session_state.llm_explanations = {}
if 'use_llm' not in st.session_state:
    st.session_state.use_llm = False
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "huggingface"
if 'matching_in_progress' not in st.session_state:
    st.session_state.matching_in_progress = False
if 'should_run_matching' not in st.session_state:
    st.session_state.should_run_matching = False

@st.cache_resource
def load_matcher(use_llm=False, llm_provider="huggingface", api_key=None):
    """Load the resume matcher model (cached)."""
    with st.spinner("Loading AI model... This may take a moment on first run."):
        if use_llm:
            matcher = LLMEnhancedMatcher(use_llm=True, llm_provider=llm_provider, api_key=api_key)
        else:
            matcher = ResumeMatcher()
        return matcher

def get_score_color(score):
    """Get color class based on score."""
    if score >= 60:
        return "high-score"
    elif score >= 40:
        return "medium-score"
    else:
        return "low-score"

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Resume Matcher AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Candidate Filtering System using Vector Similarity Search</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Threshold setting
        threshold = st.slider(
            "Match Score Threshold (%)",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            help="Minimum match score to display candidates"
        )
        
        # Role filtering
        enable_role_filter = st.checkbox(
            "Enable Role-Based Filtering",
            value=True,
            help="Filter out resumes that don't match the job role"
        )
        
        use_llm_role_detection = st.checkbox(
            "Use LLM for Role Detection (More Accurate)",
            value=False,
            help="Use AI to detect roles from text instead of keyword matching. More accurate but slower."
        )
        
        if use_llm_role_detection and not enable_role_filter:
            st.info("💡 Enable 'Role Filter' to use LLM role detection")
            use_llm_role_detection = False
        
        # Store in session state
        st.session_state.use_llm_role_detection = use_llm_role_detection
        
        # Display options
        st.header("📊 Display Options")
        show_details = st.checkbox("Show Resume Details", value=False)
        num_results = st.slider(
            "Number of Results",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )
        
        # LLM Enhancement Section
        st.markdown("---")
        st.header("🤖 LLM Enhancement")
        
        enable_llm = st.checkbox(
            "Enable LLM Reranking",
            value=st.session_state.get('use_llm', False),
            help="Use AI to rerank and explain matches (requires API key)"
        )
        
        # Update session state when checkbox changes
        st.session_state.use_llm = enable_llm
        
        if enable_llm:
            llm_provider = st.selectbox(
                "LLM Provider",
                ["huggingface", "openai", "anthropic"],
                index=0,
                help="Hugging Face is FREE with token!"
            )
            st.session_state.llm_provider = llm_provider
            
            if llm_provider == "huggingface":
                api_key = st.text_input(
                    "Hugging Face Token *",
                    type="password",
                    help="Get free token at https://huggingface.co/settings/tokens",
                    placeholder="Enter your HF token (required)"
                )
                if api_key:
                    os.environ["HUGGINGFACE_API_TOKEN"] = api_key
                    st.session_state.hf_api_key = api_key
                else:
                    st.session_state.hf_api_key = None
                if not api_key:
                    st.warning("⚠️ Token required for Hugging Face API")
                st.info("💡 Get a FREE token at: https://huggingface.co/settings/tokens")
                st.markdown("[🔗 Create Token](https://huggingface.co/settings/tokens)")
            
            elif llm_provider == "openai":
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Enter your OpenAI API key"
                )
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.session_state.openai_api_key = api_key
                else:
                    st.session_state.openai_api_key = None
                model_name = st.selectbox(
                    "Model",
                    ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
                    index=0,
                    help="gpt-4o-mini is cheapest"
                )
            
            elif llm_provider == "anthropic":
                api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    help="Enter your Anthropic API key"
                )
                if api_key:
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    st.session_state.anthropic_api_key = api_key
                else:
                    st.session_state.anthropic_api_key = None
            
            num_llm_candidates = st.slider(
                "Candidates for LLM Analysis",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of top candidates to analyze with LLM"
            )
            
            st.session_state.num_llm_candidates = num_llm_candidates
            st.session_state.use_llm = True
        else:
            st.session_state.use_llm = False
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info("""
        This tool uses:
        - **Vector Similarity Search** (FAISS)
        - **AI Embeddings** (SentenceTransformers)
        - **LLM Reranking** (Optional - Hugging Face FREE!)
        
        Upload resumes as PDF, DOCX, or TXT files.
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Job Description", "📁 Resumes", "🎯 Results"])
    
    with tab1:
        st.header("Job Description Input")
        
        # Option 1: Upload file
        st.subheader("Option 1: Upload Job Description")
        uploaded_jd = st.file_uploader(
            "Upload Job Description (TXT file)",
            type=['txt'],
            help="Upload a text file containing the job description"
        )
        
        # Option 2: Use existing file
        st.subheader("Option 2: Use Existing File")
        data_dir = Path("data")
        existing_jds = list(data_dir.glob("jd*.txt")) if data_dir.exists() else []
        
        if existing_jds:
            jd_options = [jd.name for jd in existing_jds]
            selected_jd = st.selectbox(
                "Select Job Description",
                ["-- Select --"] + jd_options
            )
            
            if selected_jd != "-- Select --":
                jd_path = data_dir / selected_jd
                with open(jd_path, 'r', encoding='utf-8') as f:
                    st.session_state.jd_text = f.read()
        else:
            st.info("No job description files found in 'data' directory.")
        
        # Option 3: Text input
        st.subheader("Option 3: Enter Text Directly")
        text_input = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Enter or paste the job description here...",
            value=st.session_state.jd_text
        )
        
        if text_input:
            st.session_state.jd_text = text_input
        
        # Display current JD
        if st.session_state.jd_text:
            st.markdown("---")
            st.subheader("Current Job Description")
            with st.expander("View Job Description", expanded=False):
                st.text(st.session_state.jd_text)
            st.success(f"✓ Job description loaded ({len(st.session_state.jd_text)} characters)")
    
    with tab2:
        st.header("Resume Management")
        
        # Option 1: Upload resumes
        st.subheader("Option 1: Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload Resume Files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files (PDF, DOCX, or TXT)"
        )
        
        # Option 2: Use existing resumes
        st.subheader("Option 2: Use Existing Resumes")
        data_dir = Path("data")
        if data_dir.exists():
            existing_resumes = [
                f for f in data_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.txt'] 
                and not f.name.lower().startswith('jd')
            ]
            
            if existing_resumes:
                resume_options = {f.name: f for f in existing_resumes}
                selected_resumes = st.multiselect(
                    "Select Resumes",
                    options=list(resume_options.keys()),
                    default=[f.name for f in existing_resumes[:5]],
                    help="Select one or more resume files"
                )
                
                if selected_resumes:
                    st.info(f"✓ {len(selected_resumes)} resume(s) selected")
            else:
                st.info("No resume files found in 'data' directory.")
        else:
            st.warning("'data' directory not found. Please create it and add resume files.")
        
        # Display uploaded files
        if uploaded_files:
            st.markdown("---")
            st.subheader("Uploaded Files")
            for file in uploaded_files:
                st.text(f"📄 {file.name} ({file.size:,} bytes)")
    
    with tab3:
        st.header("Matching Results")
        
        # Load matcher (check if LLM settings changed)
        use_llm = st.session_state.get('use_llm', False)
        llm_provider = st.session_state.get('llm_provider', 'huggingface')
        
        # Get API key based on provider
        # Priority: User input (sidebar) > Session state > Environment variable > Config constant
        api_key = None
        if use_llm:
            if llm_provider == "huggingface":
                api_key = (
                    st.session_state.get('hf_api_key') or 
                    os.getenv("HUGGINGFACE_API_TOKEN") or
                    HUGGINGFACE_API_TOKEN if HUGGINGFACE_API_TOKEN != "YOUR_HUGGINGFACE_TOKEN_HERE" else None
                )
            elif llm_provider == "openai":
                api_key = (
                    st.session_state.get('openai_api_key') or 
                    os.getenv("OPENAI_API_KEY") or
                    OPENAI_API_KEY if OPENAI_API_KEY else None
                )
            elif llm_provider == "anthropic":
                api_key = (
                    st.session_state.get('anthropic_api_key') or 
                    os.getenv("ANTHROPIC_API_KEY") or
                    ANTHROPIC_API_KEY if ANTHROPIC_API_KEY else None
                )
        
        # Load or reload matcher if settings changed
        if (st.session_state.matcher is None or 
            st.session_state.get('last_use_llm') != use_llm or
            st.session_state.get('last_llm_provider') != llm_provider):
            st.session_state.last_use_llm = use_llm
            st.session_state.last_llm_provider = llm_provider
            st.session_state.matcher = load_matcher(
                use_llm=use_llm,
                llm_provider=llm_provider,
                api_key=api_key
            )
        
        # Check if we have JD and resumes
        if not st.session_state.jd_text:
            st.warning("⚠️ Please provide a job description in the 'Job Description' tab.")
            return
        
        # Prepare resumes
        resumes = []
        
        # Get uploaded resumes
        if uploaded_files:
            for file in uploaded_files:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Parse resume
                    text = parse_resume(tmp_path)
                    resumes.append((file.name, text))
                    
                    # Clean up
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
        
        # Get selected resumes
        if 'selected_resumes' in locals() and selected_resumes:
            for resume_name in selected_resumes:
                try:
                    resume_path = resume_options[resume_name]
                    text = parse_resume(resume_path)
                    resumes.append((resume_name, text))
                except Exception as e:
                    st.error(f"Error processing {resume_name}: {str(e)}")
        
        if not resumes:
            st.warning("⚠️ Please upload or select resumes in the 'Resumes' tab.")
            return
        
        # Match button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            match_button = st.button(
                "🔍 Start Matching",
                type="primary",
                use_container_width=True
            )
        
        # Handle button click - use session state to persist across re-runs
        if match_button:
            st.session_state.results = []
            st.session_state.llm_explanations = {}
            st.session_state.matching_in_progress = True
            st.session_state.should_run_matching = True  # Flag to run matching
        
        # Run matching if flag is set (persists across Streamlit re-runs)
        should_run = st.session_state.get('should_run_matching', False) and st.session_state.get('matching_in_progress', False)
        
        print(f"\n[DEBUG] Button clicked: {match_button}")
        print(f"[DEBUG] should_run_matching: {st.session_state.get('should_run_matching', False)}")
        print(f"[DEBUG] matching_in_progress: {st.session_state.get('matching_in_progress', False)}")
        print(f"[DEBUG] should_run: {should_run}")
        
        if should_run:
            print(f"[DEBUG] Entering matching block...")
            # Perform matching
            if use_llm:
                progress_text = "🤖 Running AI-powered matching (this may take 30-60 seconds)..."
            else:
                progress_text = "🤖 Analyzing resumes and computing matches..."
            
            # Create status container for progress updates
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Ensure matcher is loaded
                if st.session_state.matcher is None:
                    status_container.error("❌ Matcher not initialized. Please refresh the page.")
                    st.session_state.matching_in_progress = False
                    st.session_state.should_run_matching = False
                    return
                
                print(f"\n[DEBUG] ===== STARTING MATCHING PROCESS =====")
                print(f"[DEBUG] Job Description length: {len(st.session_state.jd_text)} chars")
                print(f"[DEBUG] Number of resumes: {len(resumes)}")
                print(f"[DEBUG] Matcher type: {type(st.session_state.matcher).__name__}")
                
                # Step 1: Vector similarity search
                status_container.info(f"⏳ Step 1/3: Generating embeddings for {len(resumes)} resume(s)... This may take a moment for large files.")
                progress_bar.progress(10)
                
                print(f"[DEBUG] Calling compute_match_scores...")
                results = st.session_state.matcher.compute_match_scores(
                    st.session_state.jd_text,
                    resumes
                )
                print(f"[DEBUG] Got {len(results)} results from matching")
                progress_bar.progress(40)
                status_container.success("✅ Step 1/3: Embeddings generated successfully!")
                
                if len(results) == 0:
                    print(f"[WARNING] No results returned from matching!")
                    status_container.warning("⚠️ No results from vector search. Check your resumes and job description.")
                    st.session_state.matching_in_progress = False
                    st.session_state.should_run_matching = False
                    return
                
                # Step 2: Apply filters
                status_container.info("⏳ Step 2/3: Applying filters and role matching...")
                progress_bar.progress(60)
                
                # Determine if we should use LLM for role detection
                use_llm_roles = (
                    st.session_state.get('use_llm_role_detection', False) and 
                    enable_role_filter and 
                    isinstance(st.session_state.matcher, LLMEnhancedMatcher)
                )
                
                filtered_results = st.session_state.matcher.filter_by_threshold(
                    results,
                    threshold=threshold,
                    jd_text=st.session_state.jd_text if enable_role_filter else None,
                    role_filter=enable_role_filter,
                    use_llm_role_detection=use_llm_roles,
                    llm_matcher=st.session_state.matcher if use_llm_roles else None
                )
                print(f"[DEBUG] After filtering: {len(filtered_results)} candidates above {threshold}% threshold")
                progress_bar.progress(80)
                status_container.success("✅ Step 2/3: Filters applied!")
                
                # Step 3: LLM reranking (if enabled)
                # Double-check LLM settings (use_llm might have changed after matcher was loaded)
                current_use_llm = st.session_state.get('use_llm', False)
                print(f"[DEBUG] === LLM CHECK ===")
                print(f"[DEBUG] use_llm variable: {use_llm}")
                print(f"[DEBUG] current_use_llm from session: {current_use_llm}")
                print(f"[DEBUG] matcher type: {type(st.session_state.matcher)}")
                print(f"[DEBUG] is LLMEnhancedMatcher: {isinstance(st.session_state.matcher, LLMEnhancedMatcher)}")
                
                if current_use_llm and isinstance(st.session_state.matcher, LLMEnhancedMatcher):
                    status_container.info("⏳ Step 3/3: LLM is analyzing and reranking candidates... This may take 30-60 seconds.")
                    progress_bar.progress(85)
                    try:
                        # Get top candidates for LLM analysis
                        num_llm_candidates = st.session_state.get('num_llm_candidates', 10)
                        top_for_llm = filtered_results[:num_llm_candidates]
                        
                        print(f"[DEBUG] ===== STARTING LLM RERANKING =====")
                        print(f"[DEBUG] Attempting LLM reranking for {len(top_for_llm)} candidates...")
                        print(f"[DEBUG] Matcher LLM provider: {st.session_state.matcher.llm_provider}")
                        print(f"[DEBUG] API key available: {bool(st.session_state.matcher.api_key)}")
                        
                        # LLM rerank
                        llm_results = st.session_state.matcher.llm_rerank_candidates(
                            st.session_state.jd_text,
                            top_for_llm,
                            top_k=min(len(top_for_llm), num_results)
                        )
                        
                        # Check if LLM actually succeeded or just returned fallback results with errors
                        has_errors = False
                        error_messages = []
                        for result in llm_results:
                            explanation = result[3] if len(result) > 3 else ""
                            if "LLM Error:" in explanation or "Error" in explanation or "error" in explanation.lower():
                                has_errors = True
                                if explanation not in error_messages:
                                    error_messages.append(explanation)
                        
                        if has_errors:
                            # LLM failed - treat as error
                            error_msg = error_messages[0] if error_messages else "LLM reranking failed"
                            print(f"[DEBUG] ✗ LLM reranking failed (detected error in results): {error_msg}")
                            progress_bar.progress(100)
                            status_container.warning(f"⚠️ LLM reranking failed: {error_msg[:150]}... Using vector search results.")
                            st.session_state.llm_explanations = {}
                            # Don't use LLM results - continue with vector search results
                        else:
                            # LLM succeeded
                            print(f"[DEBUG] ✓ LLM reranking successful! Got {len(llm_results)} results")
                            
                            # Store explanations
                            st.session_state.llm_explanations = {
                                result[0]: result[3] for result in llm_results  # filename: explanation
                            }
                            
                            # Convert LLM results back to format expected by rest of code
                            # LLM results: (filename, score, text, explanation)
                            # Expected: (filename, score, text)
                            filtered_results = [
                                (r[0], r[1], r[2]) for r in llm_results
                            ] + filtered_results[len(llm_results):]  # Add remaining non-LLM results
                            
                            progress_bar.progress(100)
                            status_container.success("✅ Step 3/3: LLM reranking completed!")
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[DEBUG] LLM reranking failed: {error_msg}")
                        progress_bar.progress(100)
                        # Show warning but continue with vector search results
                        status_container.warning(f"⚠️ LLM reranking failed: {error_msg[:100]}... Using vector search results.")
                        st.session_state.llm_explanations = {}
                        # Continue with filtered_results from vector search
                else:
                    if current_use_llm:
                        print(f"[DEBUG] ⚠ LLM checkbox is checked BUT:")
                        print(f"[DEBUG]   - Matcher type: {type(st.session_state.matcher)}")
                        print(f"[DEBUG]   - Is LLMEnhancedMatcher? {isinstance(st.session_state.matcher, LLMEnhancedMatcher)}")
                        print(f"[DEBUG]   - Expected: {LLMEnhancedMatcher}")
                        print(f"[DEBUG] 💡 TIP: The matcher might need to be reloaded. Try clicking 'Start Matching' again.")
                    else:
                        print(f"[DEBUG] LLM reranking is DISABLED (checkbox unchecked)")
                    progress_bar.progress(100)
                    status_container.success("✅ Matching completed!")
                    st.session_state.llm_explanations = {}
                
                st.session_state.results = filtered_results
                st.session_state.last_resume_count = len(resumes)  # Store for display
                st.session_state.matching_in_progress = False  # Mark matching as complete
                st.session_state.should_run_matching = False  # Clear flag
                
                print(f"[DEBUG] Matching completed! Stored {len(st.session_state.results)} results in session state")
                print(f"[DEBUG] Results will be displayed on next Streamlit re-run")
                
                # Clear status after a brief moment
                time.sleep(0.5)
                status_container.empty()
                progress_bar.empty()
                
                # Check if we have results after matching completes
                if not st.session_state.results or len(st.session_state.results) == 0:
                    print(f"[WARNING] No results after matching - threshold may be too high")
                    st.warning(f"⚠️ No candidates found above {threshold}% threshold. Try lowering the threshold.")
                    # Don't return - let the display code handle the empty state
                else:
                    # Results exist - continue to display them below
                    print(f"[DEBUG] Results ready to display: {len(st.session_state.results)} candidates")
                    # Results will be displayed by the code below - no need to return
                
            except Exception as e:
                status_container.error(f"❌ Error during matching: {str(e)}")
                st.error(f"**Error Details:** {str(e)}")
                st.exception(e)
                st.info("💡 **Tips:**\n- Make sure all resume files are valid (PDF/DOCX/TXT)\n- Check that job description is not empty\n- Try with fewer resumes if the issue persists")
                st.session_state.matching_in_progress = False  # Mark as complete even on error
                st.session_state.should_run_matching = False  # Clear flag
                return
        
        # Display existing results (if available and matching not in progress)
        # This runs both after matching completes AND on subsequent page loads
        has_results = st.session_state.results and len(st.session_state.results) > 0
        matching_done = not st.session_state.get('matching_in_progress', False)
        
        if has_results and matching_done:
            print(f"[DEBUG] Displaying results: {len(st.session_state.results)} candidates")
            # Results summary
            st.markdown("---")
            st.subheader("📊 Summary")
            
            # Get resume count from session state or from current resumes list
            total_resumes = len(resumes) if resumes else len(st.session_state.get('last_resume_count', 0))
            if hasattr(st.session_state, 'last_resume_count'):
                total_resumes = st.session_state.last_resume_count
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyzed", total_resumes)
            with col2:
                st.metric("Matches Found", len(st.session_state.results))
                if use_llm:
                    st.caption("🤖 AI-Enhanced")
            with col3:
                avg_score = sum(r[1] for r in st.session_state.results[:num_results]) / min(len(st.session_state.results), num_results)
                st.metric("Average Score", f"{avg_score:.1f}%")
            with col4:
                best_score = st.session_state.results[0][1] if st.session_state.results else 0
                st.metric("Best Match", f"{best_score:.1f}%")
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Match Score Distribution")
            
            top_results = st.session_state.results[:num_results]
            scores = [r[1] for r in top_results]
            names = [r[0] for r in top_results]
            
            # Bar chart
            fig = px.bar(
                x=scores,
                y=names,
                orientation='h',
                labels={'x': 'Match Score (%)', 'y': 'Candidate'},
                title='Match Scores by Candidate',
                color=scores,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(
                height=max(400, len(top_results) * 40),
                yaxis={'autorange': 'reversed'}  # Reverse y-axis to show highest scores at top
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.markdown("---")
            st.subheader("🎯 Ranked Candidates")
            
            # Create DataFrame for better display
            df_data = []
            for rank, (filename, score, text) in enumerate(top_results, 1):
                # Extract preview
                preview = text[:200].replace('\n', ' ')
                
                # Detect role
                roles = st.session_state.matcher.detect_role_category(text, strict=True)
                role_str = ", ".join(roles) if roles else "Unknown"
                
                df_data.append({
                    "Rank": rank,
                    "Candidate": filename,
                    "Match Score": f"{score:.2f}%",
                    "Score": score,
                    "Role": role_str,
                    "Preview": preview
                })
            
            df = pd.DataFrame(df_data)
            
            # Display table
            st.dataframe(
                df[["Rank", "Candidate", "Match Score", "Role", "Preview"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Detailed view
            st.markdown("---")
            st.subheader("📋 Detailed Candidate Profiles")
            
            for rank, (filename, score, text) in enumerate(top_results, 1):
                # Check if we have LLM explanation
                llm_explanation = st.session_state.llm_explanations.get(filename, None)
                has_llm = llm_explanation is not None and llm_explanation and "Error" not in llm_explanation
                
                expander_label = f"Rank {rank}: {filename} - {score:.2f}%"
                if has_llm:
                    expander_label += " 🤖 AI-Enhanced"
                
                with st.expander(expander_label, expanded=(rank == 1)):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        score_class = get_score_color(score)
                        st.markdown(f'<div class="match-score {score_class}">{score:.1f}%</div>', unsafe_allow_html=True)
                        
                        # LLM explanation box
                        if has_llm:
                            st.markdown("---")
                            st.markdown("**🤖 AI Explanation:**")
                            st.info(llm_explanation)
                        
                        # Role info
                        roles = st.session_state.matcher.detect_role_category(text, strict=True)
                        if roles:
                            st.write("**Detected Role:**")
                            for role in roles:
                                st.write(f"- {role.title()}")
                    
                    with col2:
                        if show_details:
                            st.text_area(
                                "Resume Content",
                                text,
                                height=300,
                                disabled=True,
                                key=f"resume_{filename}_{rank}"
                            )
                        else:
                            st.text_area(
                                "Resume Preview (First 500 chars)",
                                text[:500] + "..." if len(text) > 500 else text,
                                height=200,
                                disabled=True,
                                key=f"preview_{filename}_{rank}"
                            )
            
            # Download results
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="matching_results.csv",
                mime="text/csv"
            )
            
            # Text export
            results_text = "RESUME MATCHING RESULTS\n"
            results_text += "="*80 + "\n\n"
            results_text += f"Job Description: {len(st.session_state.jd_text)} characters\n"
            results_text += f"Total Resumes Analyzed: {len(resumes)}\n"
            results_text += f"Matches Found: {len(st.session_state.results)}\n"
            results_text += f"Threshold: {threshold}%\n\n"
            results_text += "="*80 + "\n\n"
            
            for rank, (filename, score, text) in enumerate(top_results, 1):
                results_text += f"Rank {rank}: {filename}\n"
                results_text += f"Match Score: {score:.2f}%\n"
                results_text += f"Preview: {text[:300]}...\n\n"
                results_text += "-"*80 + "\n\n"
            
            st.download_button(
                label="📄 Download Results as Text",
                data=results_text,
                file_name="matching_results.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()

