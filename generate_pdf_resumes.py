"""
Generate 5 unique resumes and convert them to PDF format
"""
from pathlib import Path


def generate_unique_resumes():
    """Generate 5 unique resume profiles different from the DOCX ones."""
    
    # Resume 1: Frontend Developer
    resume1 = """Alexandra Martinez
Frontend Web Developer

Contact Information:
Email: alexandra.martinez@email.com
Phone: (555) 111-2222
Location: San Diego, CA
Portfolio: alexandramartinez.dev
LinkedIn: linkedin.com/in/alexandramartinez

PROFESSIONAL SUMMARY:
Creative Frontend Developer with 4 years of experience building responsive 
and interactive web applications. Expert in modern JavaScript frameworks 
and UI/UX design principles. Passionate about creating seamless user 
experiences and pixel-perfect interfaces.

TECHNICAL SKILLS:
• Frontend: HTML5, CSS3, JavaScript (ES6+), TypeScript
• Frameworks: React, Vue.js, Angular, Next.js
• Styling: Tailwind CSS, Sass, Styled Components, Bootstrap
• Tools: Webpack, Vite, npm, yarn, Git, GitHub
• Design: Figma, Adobe XD, Responsive Design
• Testing: Jest, React Testing Library, Cypress
• APIs: RESTful API integration, GraphQL

PROFESSIONAL EXPERIENCE:

Senior Frontend Developer | WebFlow Solutions | 2021 - Present
• Developed and maintained multiple React-based web applications serving 
  500K+ monthly users
• Implemented responsive designs using Tailwind CSS and CSS-in-JS solutions
• Collaborated with UX designers to translate mockups into pixel-perfect 
  components
• Optimized application performance, reducing load time by 40%
• Built reusable component library used across 10+ projects
• Mentored junior developers and conducted code reviews

Frontend Developer | Digital Creations Inc | 2019 - 2021
• Created interactive user interfaces using Vue.js and JavaScript
• Implemented responsive web designs for mobile and desktop platforms
• Integrated third-party APIs and payment gateways
• Participated in agile sprints and daily standups

EDUCATION:
Bachelor of Science in Computer Science
University of California, San Diego | 2019

PROJECTS:
• E-commerce Platform - Built with React and TypeScript, handling 1000+ 
  daily transactions
• Portfolio Website - Responsive design with smooth animations
• Task Management App - Vue.js application with real-time updates

CERTIFICATIONS:
• Google Frontend Developer Certification (2022)
• React Advanced Patterns Course (2021)"""
    
    # Resume 2: Backend Developer
    resume2 = """Robert Kim
Backend Software Engineer

Email: robert.kim@email.com
Phone: (555) 222-3333
Location: Seattle, WA
GitHub: github.com/robertkim

PROFESSIONAL SUMMARY:
Experienced Backend Engineer with 6 years specializing in scalable server-side 
development. Expert in microservices architecture, API design, and database 
optimization. Proven track record of building robust systems handling millions 
of requests daily.

TECHNICAL SKILLS:
• Languages: Python, Java, Node.js, Go, C++
• Frameworks: Django, Flask, Spring Boot, Express.js, FastAPI
• Databases: PostgreSQL, MongoDB, Redis, MySQL, DynamoDB
• Cloud: AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes
• APIs: REST, GraphQL, gRPC, WebSocket
• Tools: Git, Jenkins, Terraform, Kafka, RabbitMQ
• Testing: pytest, unittest, Postman, API testing

PROFESSIONAL EXPERIENCE:

Senior Backend Engineer | CloudTech Systems | 2020 - Present
• Designed and developed microservices architecture serving 5M+ API requests daily
• Built RESTful APIs using Python/Django and Node.js/Express
• Optimized database queries, improving response time by 60%
• Implemented caching strategies using Redis, reducing database load by 50%
• Deployed and managed services on AWS using Docker and Kubernetes
• Designed database schemas for high-traffic applications
• Implemented authentication and authorization systems
• Created comprehensive API documentation

Backend Developer | TechStartup Co | 2018 - 2020
• Developed backend services using Python Flask
• Built RESTful APIs for mobile and web applications
• Worked with PostgreSQL and MongoDB databases
• Implemented authentication using JWT tokens
• Participated in code reviews and agile development

EDUCATION:
Master of Science in Computer Science
University of Washington | 2018

Bachelor of Science in Software Engineering
University of Washington | 2016

PROJECTS:
• Real-time Chat Application - WebSocket-based backend with 10K+ concurrent users
• E-commerce Backend API - Handles inventory, orders, and payments
• Analytics Service - Processes and aggregates large datasets

CERTIFICATIONS:
• AWS Certified Solutions Architect (2021)
• MongoDB Certified Developer (2020)"""
    
    # Resume 3: DevOps Engineer
    resume3 = """Jennifer Lee
DevOps Engineer

Email: jennifer.lee@email.com
Phone: (555) 333-4444
Location: Austin, TX
LinkedIn: linkedin.com/in/jenniferlee-devops

PROFESSIONAL SUMMARY:
DevOps Engineer with 5 years of experience automating infrastructure, 
implementing CI/CD pipelines, and managing cloud environments. Expert in 
containerization, infrastructure as code, and monitoring systems. Strong 
background in Linux system administration and cloud architecture.

TECHNICAL SKILLS:
• Cloud Platforms: AWS, Azure, Google Cloud Platform
• Containerization: Docker, Kubernetes, ECS, EKS
• CI/CD: Jenkins, GitLab CI, GitHub Actions, CircleCI
• Infrastructure as Code: Terraform, CloudFormation, Ansible
• Monitoring: Prometheus, Grafana, CloudWatch, Datadog
• Scripting: Bash, Python, PowerShell
• Version Control: Git, GitHub, GitLab
• Operating Systems: Linux (Ubuntu, RHEL, CentOS), Windows Server

PROFESSIONAL EXPERIENCE:

Senior DevOps Engineer | CloudInfra Solutions | 2021 - Present
• Designed and implemented CI/CD pipelines reducing deployment time by 70%
• Managed Kubernetes clusters running 200+ microservices
• Automated infrastructure provisioning using Terraform
• Implemented monitoring and alerting using Prometheus and Grafana
• Reduced cloud costs by 30% through optimization strategies
• Managed AWS infrastructure (EC2, S3, RDS, Lambda, VPC)
• Implemented security best practices and compliance measures
• Mentored junior engineers and documented processes

DevOps Engineer | TechCorp | 2019 - 2021
• Maintained and optimized CI/CD pipelines using Jenkins
• Deployed applications using Docker containers
• Managed cloud infrastructure on AWS
• Automated routine tasks using Python and Bash scripts
• Monitored system health and performance

EDUCATION:
Bachelor of Science in Information Technology
Texas A&M University | 2019

CERTIFICATIONS:
• AWS Certified DevOps Engineer - Professional (2022)
• Certified Kubernetes Administrator (CKA) (2021)
• Docker Certified Associate (2020)
• Terraform Associate (2020)"""
    
    # Resume 4: Full Stack Developer
    resume4 = """Michael Anderson
Full Stack Developer

Email: michael.anderson@email.com
Phone: (555) 444-5555
Location: Denver, CO
Portfolio: michaelanderson.dev
GitHub: github.com/michaelanderson

PROFESSIONAL SUMMARY:
Versatile Full Stack Developer with 4 years of experience building end-to-end 
web applications. Skilled in both frontend and backend technologies with a 
passion for creating complete solutions. Experience in startup environments 
and agile development practices.

TECHNICAL SKILLS:
• Frontend: React, Next.js, TypeScript, HTML5, CSS3, Tailwind CSS
• Backend: Node.js, Express.js, Python, Django, FastAPI
• Databases: PostgreSQL, MongoDB, SQLite
• Cloud: AWS (EC2, S3, Lambda), Vercel, Heroku
• Tools: Git, Docker, npm, webpack
• APIs: RESTful API development, GraphQL
• Testing: Jest, React Testing Library, pytest

PROFESSIONAL EXPERIENCE:

Full Stack Developer | StartupHub Inc | 2020 - Present
• Developed complete web applications from design to deployment
• Built responsive frontend using React and Next.js
• Created RESTful APIs using Node.js and Express
• Designed and implemented PostgreSQL database schemas
• Deployed applications to AWS and Vercel
• Implemented authentication and user management systems
• Optimized application performance and SEO
• Collaborated with designers and product managers

Junior Full Stack Developer | WebDev Agency | 2018 - 2020
• Developed client websites using React and Node.js
• Built custom WordPress themes and plugins
• Integrated third-party APIs and services
• Maintained and updated existing web applications

EDUCATION:
Bachelor of Science in Computer Science
University of Colorado Boulder | 2018

PROJECTS:
• Social Media Platform - Full stack app with real-time features (React, Node.js)
• E-learning Platform - Course management system (Next.js, Django)
• Task Management Tool - Collaborative project management app
• Personal Blog - Static site with CMS integration

CERTIFICATIONS:
• Full Stack Web Development Bootcamp (2018)"""
    
    # Resume 5: Mobile App Developer
    resume5 = """Samantha Taylor
Mobile App Developer

Email: samantha.taylor@email.com
Phone: (555) 555-6666
Location: Boston, MA
Portfolio: samanthataylor.dev
GitHub: github.com/samanthataylor

PROFESSIONAL SUMMARY:
Mobile App Developer with 5 years of experience creating native and cross-platform 
mobile applications. Expert in iOS and Android development with a focus on 
performance, user experience, and modern design patterns. Experience in both 
startup and enterprise environments.

TECHNICAL SKILLS:
• Mobile: React Native, Flutter, Swift, Kotlin, Java
• iOS: Swift, SwiftUI, UIKit, Xcode, Core Data
• Android: Kotlin, Java, Jetpack Compose, Android Studio
• Backend: Node.js, Firebase, REST APIs
• Tools: Git, GitHub, Jira, Postman
• Design: Material Design, Human Interface Guidelines
• Testing: XCTest, Espresso, Jest

PROFESSIONAL EXPERIENCE:

Senior Mobile Developer | AppTech Solutions | 2021 - Present
• Developed cross-platform mobile apps using React Native and Flutter
• Built native iOS applications using Swift and SwiftUI
• Created Android applications using Kotlin and Jetpack Compose
• Published 10+ apps to App Store and Google Play Store
• Optimized app performance, reducing crash rate by 80%
• Integrated third-party APIs and SDKs
• Implemented push notifications and in-app purchases
• Collaborated with designers and backend developers
• Managed app releases and updates

Mobile Developer | MobileFirst Inc | 2019 - 2021
• Developed mobile applications using React Native
• Built iOS apps using Swift and Objective-C
• Created Android apps using Java and Kotlin
• Tested apps across multiple devices and OS versions
• Fixed bugs and improved app stability

EDUCATION:
Bachelor of Science in Computer Science
Massachusetts Institute of Technology | 2019

PROJECTS:
• Fitness Tracking App - React Native app with 50K+ downloads
• Food Delivery App - iOS and Android app with real-time tracking
• Social Networking App - Cross-platform app with chat features
• Productivity Suite - Task management app for iOS

CERTIFICATIONS:
• Apple iOS Development Certificate (2020)
• Google Android Developer Certification (2020)
• React Native Advanced Development (2021)"""
    
    return [
        ("resume_frontend_developer.pdf", resume1, "Frontend Developer"),
        ("resume_backend_engineer.pdf", resume2, "Backend Engineer"),
        ("resume_devops_engineer.pdf", resume3, "DevOps Engineer"),
        ("resume_fullstack_developer.pdf", resume4, "Full Stack Developer"),
        ("resume_mobile_developer.pdf", resume5, "Mobile Developer"),
    ]


def create_pdf_from_text(text_content, output_path):
    """Create a PDF file from text content."""
    # Try fpdf2 first (lighter weight)
    try:
        from fpdf import FPDF
        
        # Replace Unicode bullets with dashes for compatibility
        text_content = text_content.replace('•', '-')
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Try to use Unicode font if available, otherwise use default
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
            pdf.set_font('DejaVu', size=10)
        except:
            # Fallback to Helvetica and replace special chars
            pdf.set_font("Helvetica", size=10)
        
        # Split text into lines
        lines = text_content.split('\n')
        
        for line in lines:
            if not line.strip():
                pdf.ln(5)
                continue
            
            line = line.strip()
            
            # Check if it's a header
            is_header = (line.isupper() or 
                        (line and line[0].isupper() and len(line) < 80 and 
                         ':' not in line and not ',' in line[:30]))
            
            # Check if it's a major section (all caps or short uppercase lines)
            is_major_section = line.isupper() and len(line) < 60
            
            if is_major_section:
                pdf.ln(5)
                try:
                    pdf.set_font('DejaVu', style='B', size=14)
                except:
                    pdf.set_font("Helvetica", style="B", size=14)
                pdf.cell(0, 10, line, ln=1)
                try:
                    pdf.set_font('DejaVu', size=10)
                except:
                    pdf.set_font("Helvetica", size=10)
            elif is_header:
                try:
                    pdf.set_font('DejaVu', style='B', size=11)
                except:
                    pdf.set_font("Helvetica", style="B", size=11)
                pdf.cell(0, 7, line, ln=1)
                try:
                    pdf.set_font('DejaVu', size=10)
                except:
                    pdf.set_font("Helvetica", size=10)
            else:
                # Handle long lines by wrapping
                pdf.multi_cell(0, 5, line)
            
            pdf.ln(2)
        
        pdf.output(str(output_path))
        return True
        
    except ImportError:
        # Fallback to reportlab if fpdf2 not available
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_LEFT
            
            doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            elements = []
            styles = getSampleStyleSheet()
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=12,
                textColor='black',
                spaceAfter=6,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                textColor='black',
                spaceAfter=3,
                alignment=TA_LEFT,
                leading=12
            )
            
            lines = text_content.split('\n')
            
            for line in lines:
                if not line.strip():
                    elements.append(Spacer(1, 0.1*inch))
                    continue
                
                line = line.strip()
                
                if line.isupper() or (line and line[0].isupper() and len(line) < 80 and ':' not in line and not ',' in line[:30]):
                    elements.append(Paragraph(line, heading_style))
                else:
                    line_escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    elements.append(Paragraph(line_escaped, normal_style))
            
            doc.build(elements)
            return True
            
        except ImportError:
            print(f"⚠ PDF libraries not installed. Install with: pip install fpdf2 OR pip install reportlab")
            return False
    except Exception as e:
        print(f"✗ Error creating PDF: {str(e)}")
        return False


def generate_pdf_resumes(output_dir="data"):
    """Generate 5 unique PDF resumes."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creating 5 unique PDF resumes...\n")
    
    resumes = generate_unique_resumes()
    pdf_count = 0
    
    for filename, content, role in resumes:
        pdf_path = output_path / filename
        
        if create_pdf_from_text(content, pdf_path):
            pdf_count += 1
            print(f"✓ Created: {filename} ({role})")
        else:
            print(f"✗ Failed: {filename}")
    
    print("\n" + "="*60)
    print(f"✓ Successfully created {pdf_count} PDF resume(s)")
    print("="*60)
    
    if pdf_count < len(resumes):
        print("\n⚠ Note: Install PDF library to create PDFs:")
        print("   pip install fpdf2")
        print("   OR")
        print("   pip install reportlab")


if __name__ == "__main__":
    generate_pdf_resumes()

