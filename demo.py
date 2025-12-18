"""
Main Demo Script - Resume Matching System
Demonstrates AI-powered resume filtering based on job description
"""
import sys
from pathlib import Path
from resume_matcher import ResumeMatcher


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print(" " * 20 + "RESUME MATCHING DEMO")
    print(" " * 15 + "AI-Powered Candidate Filtering System")
    print("="*80 + "\n")


def main():
    """Main demo function."""
    print_banner()
    
    # Initialize matcher
    matcher = ResumeMatcher()
    
    # Determine paths
    data_dir = Path("data")
    
    # Check for Software Developer JD, otherwise use default
    jd_files = list(data_dir.glob("jd*.txt"))
    if not jd_files:
        print("⚠ Warning: No job description file found in 'data' directory.")
        print("   Please ensure you have a 'jd*.txt' file in the data folder.")
        return
    
    # Use the first JD file found (prefer software developer one)
    preferred_jd = data_dir / "jd_software_developer.txt"
    jd_path = preferred_jd if preferred_jd.exists() else jd_files[0]
    
    print(f"📄 Job Description: {jd_path.name}")
    
    # Load job description
    try:
        jd_text = matcher.load_job_description(jd_path)
        print(f"   Length: {len(jd_text)} characters\n")
    except Exception as e:
        print(f"❌ Error loading job description: {str(e)}")
        return
    
    # Load resumes
    print(f"📁 Loading resumes from: {data_dir}/")
    try:
        resumes = matcher.load_resumes(resume_directory=str(data_dir))
        print(f"\n✓ Successfully loaded {len(resumes)} resume(s)\n")
    except Exception as e:
        print(f"❌ Error loading resumes: {str(e)}")
        return
    
    if not resumes:
        print("❌ No resumes found in the data directory.")
        print("   Please add resume files (.pdf, .docx, or .txt) to the 'data' folder.")
        return
    
    # Compute matches
    print("\n" + "-"*80)
    print("COMPUTING SIMILARITY SCORES...")
    print("-"*80)
    results = matcher.compute_match_scores(jd_text, resumes)
    
    # Display results
    matcher.display_results(results, show_details=False)
    
    # Filter results (optional threshold and role-based filtering)
    # Higher threshold for better filtering (cosine similarity typically gives 0-100% scores)
    threshold = 35.0  # Lower threshold since cosine similarity gives different score range
    # Use role filtering to prevent mismatches (e.g., data scientist matching sales job)
    filtered_results = matcher.filter_by_threshold(
        results, 
        threshold=threshold,
        jd_text=jd_text,
        role_filter=True  # Enable role-based filtering
    )
    
    print(f"\n📊 Filtering candidates above {threshold}% match threshold...")
    print(f"✓ {len(filtered_results)} candidate(s) passed the threshold")
    
    if filtered_results:
        print("\n✅ RECOMMENDED CANDIDATES:")
        print("-"*80)
        for rank, (filename, score, text) in enumerate(filtered_results[:3], start=1):
            print(f"{rank}. {filename} ({score:.2f}%)")
    
    # Save detailed results
    output_file = "matching_results.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RESUME MATCHING RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Job Description: {jd_path.name}\n")
            f.write(f"Total Resumes Analyzed: {len(resumes)}\n")
            f.write(f"Candidates Above {threshold}% Threshold: {len(filtered_results)}\n\n")
            f.write("="*80 + "\n\n")
            
            for rank, (filename, score, text) in enumerate(results, start=1):
                status = "✅ PASSED" if score >= threshold else "❌ BELOW THRESHOLD"
                f.write(f"Rank {rank}: {filename}\n")
                f.write(f"Match Score: {score:.2f}%\n")
                f.write(f"Status: {status}\n")
                f.write(f"\nResume Content:\n{text[:500]}...\n\n")
                f.write("-"*80 + "\n\n")
        
        print(f"\n✓ Detailed results saved to: {output_file}")
    except Exception as e:
        print(f"⚠ Warning: Could not save results to file: {str(e)}")
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

