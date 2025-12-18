"""
Generate synthetic job description and resumes for Software Developer role
"""
from pathlib import Path


def generate_software_developer_jd():
    """Generate a realistic Software Developer job description."""
    jd = """Software Developer

Job Description:

We are seeking a skilled Software Developer to join our dynamic development team. 
The ideal candidate will have strong programming skills and experience building 
scalable web applications.

Requirements:
• Bachelor's degree in Computer Science or related field
• 3+ years of professional software development experience
• Proficiency in Python, JavaScript, and modern web frameworks
• Experience with RESTful API development
• Strong knowledge of databases (SQL, PostgreSQL)
• Familiarity with version control systems (Git)
• Experience with cloud platforms (AWS, Azure, or GCP)
• Understanding of software testing and debugging
• Knowledge of agile development methodologies
• Strong problem-solving and analytical skills

Preferred Skills:
• Experience with React or Vue.js frontend frameworks
• Knowledge of Docker and containerization
• Experience with CI/CD pipelines
• Understanding of microservices architecture
• API design and documentation

We offer competitive compensation and opportunities for professional growth in a 
collaborative, innovative environment."""
    
    return jd.strip()


def generate_resumes():
    """Generate 5 synthetic resumes - mix of matching and non-matching candidates."""
    
    # Resume 1: Strong match - Experienced Software Developer
    resume1 = """John Smith
Software Developer

Email: john.smith@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

PROFESSIONAL SUMMARY:
Experienced Software Developer with 5 years of expertise in full-stack web 
development. Proficient in Python, JavaScript, React, and RESTful API design. 
Strong background in cloud platforms and agile methodologies.

TECHNICAL SKILLS:
• Programming Languages: Python, JavaScript, TypeScript, Java
• Web Frameworks: React, Node.js, Flask, Django
• Databases: PostgreSQL, MySQL, MongoDB
• Cloud Platforms: AWS, Azure
• Tools: Git, Docker, Jenkins, JIRA
• APIs: RESTful API design and development

PROFESSIONAL EXPERIENCE:

Senior Software Developer | Tech Solutions Inc. | 2020 - Present
• Developed and maintained scalable web applications using Python and React
• Designed and implemented RESTful APIs serving 100K+ daily requests
• Collaborated with cross-functional teams using agile methodologies
• Deployed applications to AWS cloud infrastructure
• Wrote unit and integration tests ensuring 85% code coverage
• Participated in code reviews and mentored junior developers

Software Developer | Digital Innovations LLC | 2018 - 2020
• Built responsive web applications using JavaScript and React
• Worked with PostgreSQL databases for data management
• Used Git for version control in collaborative environment
• Implemented CI/CD pipelines using Jenkins

EDUCATION:
Bachelor of Science in Computer Science
University of California, Berkeley | 2018

CERTIFICATIONS:
• AWS Certified Developer - Associate
• Certified Scrum Master"""
    
    # Resume 2: Good match - Junior Software Developer
    resume2 = """Sarah Johnson
Junior Software Developer

Email: sarah.j@email.com
Phone: (555) 234-5678
Location: Seattle, WA

PROFESSIONAL SUMMARY:
Motivated Software Developer with 2 years of experience in web development. 
Strong foundation in Python and JavaScript, with growing expertise in modern 
frameworks and cloud technologies.

TECHNICAL SKILLS:
• Programming Languages: Python, JavaScript
• Web Frameworks: Vue.js, Flask
• Databases: PostgreSQL, SQLite
• Cloud Platforms: AWS (beginner)
• Tools: Git, Docker basics
• APIs: REST API consumption and basic design

PROFESSIONAL EXPERIENCE:

Junior Software Developer | StartupTech Co. | 2022 - Present
• Developed web features using Python Flask backend
• Built interactive UI components using Vue.js
• Worked with PostgreSQL databases
• Collaborated using Git version control
• Participated in agile sprint planning and daily standups

Software Intern | CodeWorks | 2021 - 2022
• Assisted in building web applications
• Learned Python and JavaScript programming
• Participated in code reviews

EDUCATION:
Bachelor of Science in Computer Science
University of Washington | 2021"""
    
    # Resume 3: Partial match - Data Scientist (different but related)
    resume3 = """Michael Chen
Data Scientist

Email: michael.chen@email.com
Phone: (555) 345-6789
Location: New York, NY

PROFESSIONAL SUMMARY:
Data Scientist with 4 years of experience in machine learning, statistical 
analysis, and data engineering. Proficient in Python for data analysis and 
model development.

TECHNICAL SKILLS:
• Programming Languages: Python, R, SQL
• Libraries: Pandas, NumPy, Scikit-learn, TensorFlow
• Databases: PostgreSQL, MongoDB
• Tools: Git, Jupyter Notebooks
• Cloud Platforms: AWS (S3, EC2)
• Machine Learning: Model development and deployment

PROFESSIONAL EXPERIENCE:

Data Scientist | Analytics Corp | 2019 - Present
• Developed machine learning models using Python
• Analyzed large datasets using SQL and PostgreSQL
• Built data pipelines on AWS infrastructure
• Collaborated using Git for code versioning
• Presented findings to stakeholders

Data Analyst | Insights Ltd | 2017 - 2019
• Performed statistical analysis using Python and R
• Queried databases using SQL
• Created data visualizations and reports

EDUCATION:
Master of Science in Data Science
Columbia University | 2017"""
    
    # Resume 4: Weak match - IT Support Specialist
    resume4 = """Emily Rodriguez
IT Support Specialist

Email: emily.r@email.com
Phone: (555) 456-7890
Location: Austin, TX

PROFESSIONAL SUMMARY:
Dedicated IT Support Specialist with 6 years of experience providing technical 
support and maintaining IT infrastructure. Strong troubleshooting and customer 
service skills.

TECHNICAL SKILLS:
• Operating Systems: Windows, Linux, macOS
• Networking: TCP/IP, DNS, DHCP
• Hardware: Desktop and server maintenance
• Software: Active Directory, Office 365, VMware
• Ticketing Systems: ServiceNow, JIRA
• Basic Scripting: PowerShell, Bash

PROFESSIONAL EXPERIENCE:

Senior IT Support Specialist | Corporate Solutions | 2018 - Present
• Provided technical support to 500+ employees
• Managed Active Directory and user accounts
• Installed and configured hardware and software
• Troubleshot network connectivity issues
• Documented IT procedures and solutions

IT Support Technician | TechSupport Inc | 2017 - 2018
• Resolved help desk tickets
• Maintained computer systems and printers
• Assisted with software installations

EDUCATION:
Associate Degree in Information Technology
Austin Community College | 2017

CERTIFICATIONS:
• CompTIA A+
• Microsoft Certified: Azure Fundamentals"""
    
    # Resume 5: No match - Marketing Manager
    resume5 = """David Thompson
Marketing Manager

Email: david.thompson@email.com
Phone: (555) 567-8901
Location: Los Angeles, CA

PROFESSIONAL SUMMARY:
Creative Marketing Manager with 7 years of experience in digital marketing, 
brand management, and campaign development. Proven track record of driving 
growth through strategic marketing initiatives.

SKILLS:
• Digital Marketing: SEO, SEM, Social Media Marketing
• Tools: Google Analytics, HubSpot, Salesforce
• Content Creation: Copywriting, Graphic Design
• Project Management: Campaign planning and execution
• Analytics: Marketing metrics and ROI analysis

PROFESSIONAL EXPERIENCE:

Marketing Manager | Brand Strategies Inc | 2019 - Present
• Developed and executed comprehensive marketing campaigns
• Managed digital advertising budgets exceeding $500K annually
• Analyzed marketing performance using Google Analytics
• Collaborated with sales team using Salesforce CRM
• Created engaging content for social media platforms

Marketing Coordinator | Creative Agency | 2016 - 2019
• Assisted in campaign development and execution
• Managed social media accounts
• Created marketing materials and presentations

EDUCATION:
Bachelor of Arts in Marketing
University of California, Los Angeles | 2016

CERTIFICATIONS:
• Google Analytics Certified
• HubSpot Content Marketing Certified"""
    
    return [
        ("resume_software_dev_senior.txt", resume1),
        ("resume_software_dev_junior.txt", resume2),
        ("resume_data_scientist.txt", resume3),
        ("resume_it_support.txt", resume4),
        ("resume_marketing_manager.txt", resume5)
    ]


def save_sample_data(output_dir="data"):
    """Save generated job description and resumes to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save job description
    jd_path = output_path / "jd_software_developer.txt"
    jd_text = generate_software_developer_jd()
    with open(jd_path, 'w', encoding='utf-8') as f:
        f.write(jd_text)
    print(f"✓ Created: {jd_path}")
    
    # Save resumes
    resumes = generate_resumes()
    for filename, content in resumes:
        resume_path = output_path / filename
        with open(resume_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {resume_path}")
    
    print(f"\n✓ Generated {len(resumes) + 1} files in '{output_dir}' directory")
    print("\nResume breakdown:")
    print("  1. Strong match: resume_software_dev_senior.txt (5 years exp, all skills)")
    print("  2. Good match: resume_software_dev_junior.txt (2 years exp, some skills)")
    print("  3. Partial match: resume_data_scientist.txt (different role, Python/DB skills)")
    print("  4. Weak match: resume_it_support.txt (IT but not software dev)")
    print("  5. No match: resume_marketing_manager.txt (completely different field)")


if __name__ == "__main__":
    save_sample_data()

