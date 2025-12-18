"""
Resume Parser - Extracts text from PDF and DOCX resume files
"""
import os
from pathlib import Path


def extract_text_from_pdf(file_path):
    """Extract text from PDF file using pdfplumber."""
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except ImportError:
        raise ImportError("pdfplumber is required for PDF parsing. Install with: pip install pdfplumber")
    except Exception as e:
        raise Exception(f"Error reading PDF {file_path}: {str(e)}")


def extract_text_from_docx(file_path):
    """Extract text from DOCX file using python-docx."""
    try:
        from docx import Document
        
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx")
    except Exception as e:
        raise Exception(f"Error reading DOCX {file_path}: {str(e)}")


def parse_resume(file_path):
    """
    Parse resume file (PDF or DOCX) and extract text.
    
    Args:
        file_path: Path to resume file
        
    Returns:
        Extracted text content
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        # For text files, just read directly
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .pdf, .docx, .txt")


def load_resumes_from_directory(directory_path):
    """
    Load all resume files from a directory.
    
    Args:
        directory_path: Path to directory containing resume files
        
    Returns:
        List of tuples: (filename, text_content)
    """
    directory = Path(directory_path)
    resumes = []
    
    if not directory.exists():
        return resumes
    
    # Supported extensions
    resume_extensions = ['.pdf', '.docx', '.doc', '.txt']
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in resume_extensions:
            try:
                # Skip job description files
                if 'jd' in file_path.name.lower() or 'job' in file_path.name.lower():
                    continue
                    
                text = parse_resume(file_path)
                if text:
                    resumes.append((file_path.name, text))
                    print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {str(e)}")
    
    return resumes


if __name__ == "__main__":
    # Test the parser
    test_dir = "data"
    if os.path.exists(test_dir):
        resumes = load_resumes_from_directory(test_dir)
        print(f"\nLoaded {len(resumes)} resumes")

