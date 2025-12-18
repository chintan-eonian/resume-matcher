"""
Generate comprehensive test data for resume matching system
Creates multiple job descriptions and resumes across different roles
"""
import os
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# ============================================================================
# JOB DESCRIPTIONS - Different roles/industries
# ============================================================================

job_descriptions = {
    "jd_software_developer.txt": """
SOFTWARE DEVELOPER - Full Stack
================================

Job Description

We are seeking a talented Full Stack Software Developer to join our dynamic team. 
The ideal candidate will have experience in both frontend and backend development 
technologies.

Responsibilities:
- Design and develop scalable web applications using React, Node.js, and Python
- Write clean, maintainable, and efficient code
- Collaborate with cross-functional teams to define and implement new features
- Participate in code reviews and maintain high code quality standards
- Troubleshoot and debug applications
- Work with databases (PostgreSQL, MongoDB)
- Implement RESTful APIs and microservices architecture
- Use version control systems (Git, GitHub, GitLab)
- Deploy applications to cloud platforms (AWS, Azure, GCP)

Requirements:
- Bachelor's degree in Computer Science or related field
- 3+ years of experience in software development
- Proficiency in JavaScript, Python, Java, or similar languages
- Experience with React, Angular, or Vue.js frameworks
- Knowledge of databases (SQL and NoSQL)
- Understanding of Docker, Kubernetes, CI/CD pipelines
- Strong problem-solving and analytical skills
- Excellent communication and teamwork abilities

Nice to have:
- Experience with machine learning and AI
- Knowledge of DevOps practices
- Experience with test-driven development (TDD)
""",

    "jd_sales_manager.txt": """
SALES MANAGER - B2B Technology Solutions
========================================

Job Description

We are looking for an experienced Sales Manager to drive our business development 
and revenue growth in the technology solutions sector.

Responsibilities:
- Develop and execute sales strategies to achieve revenue targets
- Build and maintain relationships with key enterprise clients
- Lead a team of sales representatives and provide coaching
- Identify new business opportunities and market trends
- Prepare and deliver sales presentations and proposals
- Negotiate contracts and close deals with clients
- Track sales metrics and performance indicators
- Attend industry events and conferences for networking
- Collaborate with marketing team on lead generation campaigns
- Manage sales pipeline and CRM systems (Salesforce, HubSpot)

Requirements:
- Bachelor's degree in Business, Marketing, or related field
- 5+ years of experience in B2B sales, preferably in technology sector
- Proven track record of meeting or exceeding sales quotas
- Strong leadership and team management skills
- Excellent communication, negotiation, and presentation skills
- Proficiency in CRM software and sales tools
- Ability to travel as needed (up to 30%)
- Self-motivated with strong drive for results

Nice to have:
- MBA or advanced degree
- Experience in SaaS or cloud solutions sales
- Knowledge of data analytics and reporting tools
""",

    "jd_data_scientist.txt": """
DATA SCIENTIST - Machine Learning Specialist
============================================

Job Description

We are seeking a skilled Data Scientist with expertise in machine learning to join 
our analytics team and drive data-driven decision making.

Responsibilities:
- Analyze large datasets to extract meaningful insights
- Develop and deploy machine learning models for predictive analytics
- Build and maintain data pipelines and ETL processes
- Create data visualizations and dashboards (Tableau, Power BI)
- Collaborate with business stakeholders to understand requirements
- Conduct statistical analysis and A/B testing
- Work with cloud platforms (AWS SageMaker, Google Cloud ML)
- Write production-quality code in Python or R
- Document models, algorithms, and methodologies
- Stay updated with latest ML/AI research and techniques

Requirements:
- Master's or PhD in Data Science, Statistics, Computer Science, or related field
- 3+ years of experience in data science or machine learning
- Strong programming skills in Python (pandas, scikit-learn, TensorFlow, PyTorch)
- Experience with SQL and database systems
- Knowledge of statistical methods and machine learning algorithms
- Experience with data visualization tools
- Strong analytical and problem-solving skills
- Excellent communication skills to explain complex concepts

Nice to have:
- Experience with deep learning and neural networks
- Knowledge of natural language processing (NLP)
- Experience with big data technologies (Spark, Hadoop)
- Published research papers in ML/AI
""",

    "jd_marketing_manager.txt": """
MARKETING MANAGER - Digital Marketing & Brand Management
========================================================

Job Description

We are looking for a creative and strategic Marketing Manager to lead our digital 
marketing initiatives and brand management efforts.

Responsibilities:
- Develop and execute comprehensive marketing strategies
- Manage digital marketing campaigns across multiple channels (SEO, SEM, social media, email)
- Oversee content creation including blog posts, social media, and marketing materials
- Analyze marketing metrics and ROI using Google Analytics and other tools
- Manage marketing budget and allocate resources effectively
- Lead and mentor marketing team members
- Coordinate with external agencies and vendors
- Plan and execute events, trade shows, and product launches
- Monitor brand reputation and manage public relations
- Conduct market research and competitive analysis

Requirements:
- Bachelor's degree in Marketing, Communications, or related field
- 4+ years of experience in marketing, with focus on digital marketing
- Experience with Google Ads, Facebook Ads, LinkedIn Ads
- Proficiency in marketing automation tools (HubSpot, Mailchimp)
- Strong copywriting and content creation skills
- Knowledge of SEO and content marketing best practices
- Experience with analytics and reporting tools
- Creative thinking and problem-solving abilities
- Excellent project management and organizational skills

Nice to have:
- Marketing certifications (Google Ads, HubSpot, etc.)
- Experience with video marketing and production
- Graphic design skills (Adobe Creative Suite)
- Experience in B2B marketing
""",

    "jd_product_manager.txt": """
PRODUCT MANAGER - Software Products
===================================

Job Description

We are seeking an experienced Product Manager to drive product strategy and 
development for our software product portfolio.

Responsibilities:
- Define product vision, strategy, and roadmap
- Gather and analyze user requirements through research and feedback
- Work closely with engineering teams to deliver products on time
- Create product specifications, user stories, and acceptance criteria
- Prioritize features and manage product backlog
- Conduct competitive analysis and market research
- Collaborate with design, engineering, sales, and marketing teams
- Monitor product metrics and KPIs
- Plan product launches and go-to-market strategies
- Engage with customers and stakeholders for feedback
- Use agile methodologies (Scrum, Kanban) for product development

Requirements:
- Bachelor's degree in Business, Engineering, or related field
- 4+ years of experience in product management, preferably in software/tech
- Strong analytical and strategic thinking skills
- Experience with product management tools (Jira, Confluence, Aha!)
- Excellent communication and presentation skills
- Ability to work with cross-functional teams
- Understanding of software development lifecycle
- Customer-focused mindset
- Data-driven decision making approach

Nice to have:
- MBA or advanced degree
- Technical background (engineering, computer science)
- Experience with UX/UI design principles
- Product management certifications
""",

    "jd_accountant.txt": """
SENIOR ACCOUNTANT - Financial Reporting
=======================================

Job Description

We are looking for a detail-oriented Senior Accountant to manage our financial 
reporting and accounting operations.

Responsibilities:
- Prepare monthly, quarterly, and annual financial statements
- Perform general ledger reconciliations and month-end close procedures
- Ensure compliance with accounting standards (GAAP, IFRS)
- Manage accounts payable and accounts receivable processes
- Prepare and file tax returns and ensure tax compliance
- Conduct financial analysis and variance analysis
- Assist with budgeting and forecasting
- Coordinate with external auditors during audits
- Maintain accurate accounting records and documentation
- Use accounting software (QuickBooks, SAP, Oracle, NetSuite)

Requirements:
- Bachelor's degree in Accounting or Finance
- CPA (Certified Public Accountant) certification preferred
- 3+ years of experience in accounting or finance
- Strong knowledge of accounting principles and practices
- Proficiency in Excel and accounting software
- Attention to detail and accuracy
- Strong analytical and problem-solving skills
- Excellent organizational and time management skills
- Ability to meet deadlines and work under pressure

Nice to have:
- Experience with ERP systems
- Knowledge of tax regulations
- Experience in specific industry (retail, manufacturing, etc.)
"""
}

# ============================================================================
# RESUMES - Mix of matching and non-matching candidates
# ============================================================================

resumes = {
    "resume_software_dev_senior.txt": """
JOHN ANDERSON
Senior Full Stack Developer
Email: john.anderson@email.com | Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johnanderson

PROFESSIONAL SUMMARY
--------------------
Experienced Full Stack Developer with 6+ years of expertise in building scalable 
web applications. Proficient in React, Node.js, Python, and cloud technologies. 
Strong background in microservices architecture and DevOps practices.

TECHNICAL SKILLS
----------------
Programming Languages: JavaScript, Python, Java, TypeScript
Frontend: React, Angular, Vue.js, HTML5, CSS3
Backend: Node.js, Express, Django, Flask, Spring Boot
Databases: PostgreSQL, MongoDB, MySQL, Redis
Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD, Jenkins
Tools: Git, GitHub, Jira, Postman, VS Code

PROFESSIONAL EXPERIENCE

Senior Full Stack Developer | TechCorp Inc. | 2020 - Present
------------------------------------------------------------
- Led development of microservices-based e-commerce platform serving 1M+ users
- Built React frontend with Redux for state management
- Developed RESTful APIs using Node.js and Express
- Implemented CI/CD pipelines reducing deployment time by 60%
- Mentored junior developers and conducted code reviews
- Technologies: React, Node.js, PostgreSQL, AWS, Docker

Full Stack Developer | StartupXYZ | 2018 - 2020
------------------------------------------------
- Developed responsive web applications using React and Angular
- Created Python-based APIs and data processing services
- Worked with MongoDB for document storage
- Collaborated with cross-functional teams using Agile methodologies
- Technologies: React, Python, MongoDB, AWS

EDUCATION
---------
Bachelor of Science in Computer Science
University of Technology | 2018

PROJECTS
--------
- Open-source React component library (500+ GitHub stars)
- Real-time chat application using WebSockets
""",

    "resume_data_scientist.txt": """
DR. SARAH CHEN
Senior Data Scientist
Email: sarah.chen@email.com | Phone: (555) 234-5678
LinkedIn: linkedin.com/in/sarahchen

PROFESSIONAL SUMMARY
--------------------
PhD in Machine Learning with 5 years of experience in data science and predictive 
analytics. Expert in Python, TensorFlow, and building production ML systems. 
Published researcher with 10+ papers in top ML conferences.

TECHNICAL SKILLS
----------------
Languages: Python, R, SQL, Scala
ML/AI: TensorFlow, PyTorch, scikit-learn, XGBoost, Keras
Data Tools: pandas, NumPy, Spark, Hadoop
Visualization: Tableau, Power BI, Matplotlib, Seaborn
Cloud: AWS SageMaker, Google Cloud ML, Azure ML
Databases: PostgreSQL, MongoDB, Snowflake

PROFESSIONAL EXPERIENCE

Senior Data Scientist | DataAnalytics Corp | 2019 - Present
------------------------------------------------------------
- Built and deployed 15+ ML models for customer churn prediction and recommendation systems
- Reduced model training time by 70% using distributed computing with Spark
- Created automated ML pipelines using AWS SageMaker
- Developed A/B testing frameworks increasing conversion rates by 25%
- Presented insights to C-level executives using Tableau dashboards
- Published research on deep learning applications in NLP

Data Scientist | Analytics Solutions Inc. | 2017 - 2019
--------------------------------------------------------
- Analyzed large-scale datasets (100M+ records) to extract business insights
- Developed predictive models for sales forecasting and demand planning
- Built ETL pipelines for data preprocessing and feature engineering
- Created data visualizations and dashboards for stakeholders

EDUCATION
---------
PhD in Computer Science - Machine Learning Specialization
Stanford University | 2017

Master of Science in Statistics
UC Berkeley | 2014

PUBLICATIONS
------------
- "Deep Learning for Natural Language Processing" - NeurIPS 2020
- "Scalable Machine Learning Systems" - ICML 2019
""",

    "resume_sales_manager.txt": """
MICHAEL ROBERTSON
Sales Manager - B2B Technology
Email: michael.r@email.com | Phone: (555) 345-6789
LinkedIn: linkedin.com/in/michaelrobertson

PROFESSIONAL SUMMARY
--------------------
Results-driven Sales Manager with 8 years of experience in B2B technology sales. 
Proven track record of exceeding quotas and building high-performing sales teams. 
Expert in enterprise software and cloud solutions sales.

KEY ACHIEVEMENTS
----------------
- Exceeded annual sales targets by 35% for 3 consecutive years
- Closed $15M+ in enterprise deals over career
- Built and led team of 8 sales representatives
- Recognized as Top Performer 4 times

PROFESSIONAL EXPERIENCE

Sales Manager | CloudTech Solutions | 2019 - Present
-----------------------------------------------------
- Manage team of 8 sales representatives covering enterprise accounts
- Develop and execute sales strategies for SaaS and cloud solutions
- Achieved 125% of annual revenue target in 2023
- Closed 5 enterprise deals worth $2M+ each
- Coach and mentor sales team, improving average performance by 40%
- Use Salesforce CRM to manage pipeline of 200+ opportunities
- Collaborate with marketing on lead generation and campaigns
- Attend industry conferences (AWS re:Invent, Salesforce Dreamforce)

Senior Sales Representative | TechVendor Inc. | 2016 - 2019
------------------------------------------------------------
- Exceeded quarterly sales quotas consistently (110-140%)
- Generated $8M in new business revenue
- Managed 50+ enterprise accounts
- Built relationships with C-level executives
- Negotiated multi-year contracts with Fortune 500 companies
- Used HubSpot CRM for pipeline management

Sales Representative | Software Solutions Ltd. | 2015 - 2016
-------------------------------------------------------------
- Generated $2M in new business in first year
- Conducted product demonstrations and presentations
- Qualified leads and developed sales proposals

EDUCATION
---------
Bachelor of Business Administration - Marketing
State University | 2015

CERTIFICATIONS
--------------
- Salesforce Certified Administrator
- AWS Certified Cloud Practitioner
- HubSpot Sales Software Certification
""",

    "resume_marketing_specialist.txt": """
EMILY JOHNSON
Digital Marketing Manager
Email: emily.johnson@email.com | Phone: (555) 456-7890
LinkedIn: linkedin.com/in/emilyjohnson

PROFESSIONAL SUMMARY
--------------------
Creative and strategic Digital Marketing Manager with 5 years of experience 
driving brand awareness and lead generation through multi-channel marketing campaigns. 
Expert in SEO, SEM, social media, and content marketing.

KEY SKILLS
----------
Digital Marketing: SEO, SEM, Google Ads, Facebook Ads, LinkedIn Ads
Content Marketing: Blog writing, social media, email campaigns
Analytics: Google Analytics, Adobe Analytics, Facebook Insights
Tools: HubSpot, Mailchimp, Hootsuite, Canva, Adobe Creative Suite
Marketing Automation: Lead nurturing, email workflows, scoring

PROFESSIONAL EXPERIENCE

Digital Marketing Manager | Growth Marketing Agency | 2020 - Present
---------------------------------------------------------------------
- Developed and executed marketing strategies for 10+ B2B clients
- Increased website traffic by 150% through SEO and content marketing
- Managed $500K+ annual digital advertising budget across Google and social platforms
- Created and managed content calendar producing 50+ blog posts annually
- Improved email open rates by 35% through A/B testing and segmentation
- Led team of 3 marketing specialists
- Generated 500+ qualified leads monthly through multi-channel campaigns
- Analyzed marketing ROI and presented reports to clients

Marketing Specialist | TechStart Inc. | 2018 - 2020
----------------------------------------------------
- Managed social media presence across Facebook, LinkedIn, Twitter, Instagram
- Created engaging content including blog posts, infographics, and videos
- Launched email marketing campaigns reaching 50K+ subscribers
- Coordinated trade show participation and event marketing
- Assisted with product launch campaigns

Marketing Coordinator | Local Business Solutions | 2016 - 2018
----------------------------------------------------------------
- Managed company website and blog content
- Created marketing materials and collateral
- Assisted with event planning and coordination
- Managed social media accounts

EDUCATION
---------
Bachelor of Arts in Marketing and Communications
Marketing University | 2016

CERTIFICATIONS
--------------
- Google Ads Certified
- HubSpot Content Marketing Certified
- Google Analytics Certified
""",

    "resume_product_manager.txt": """
DAVID KIM
Product Manager - Software Products
Email: david.kim@email.com | Phone: (555) 567-8901
LinkedIn: linkedin.com/in/davidkim

PROFESSIONAL SUMMARY
--------------------
Strategic Product Manager with 6 years of experience in software product development. 
Expert in agile methodologies, user research, and cross-functional collaboration. 
Successfully launched 5+ products from concept to market.

KEY SKILLS
----------
Product Management: Roadmapping, user stories, backlog management, prioritization
Tools: Jira, Confluence, Aha!, Productboard, Figma
Methodologies: Agile, Scrum, Kanban, Lean Startup
Analytics: Product metrics, user analytics, A/B testing
Communication: Stakeholder management, presentations, documentation

PROFESSIONAL EXPERIENCE

Senior Product Manager | SoftwareCo | 2020 - Present
-----------------------------------------------------
- Lead product strategy for mobile app portfolio with 2M+ active users
- Define product roadmap and prioritize features based on user research and data
- Work with engineering teams using Agile/Scrum to deliver features on time
- Conduct user interviews and analyze feedback to inform product decisions
- Manage product backlog and write detailed user stories
- Collaborate with design team on UX/UI improvements
- Launched 3 major features increasing user engagement by 45%
- Monitor product KPIs and metrics using analytics tools

Product Manager | TechStartup | 2018 - 2020
---------------------------------------------
- Owned product vision and strategy for SaaS platform
- Gathered requirements from customers and internal stakeholders
- Created product specifications and worked with engineering on implementation
- Conducted competitive analysis and market research
- Planned product launches and go-to-market strategies
- Increased user retention by 30% through product improvements

Associate Product Manager | Enterprise Software Inc. | 2016 - 2018
--------------------------------------------------------------------
- Supported product managers on multiple product lines
- Created product documentation and user guides
- Assisted with user acceptance testing
- Analyzed user data and generated reports

EDUCATION
---------
Master of Business Administration (MBA)
Business School | 2016

Bachelor of Science in Computer Science
Tech University | 2014

CERTIFICATIONS
--------------
- Certified Scrum Product Owner (CSPO)
- Pragmatic Marketing Certified
""",

    "resume_accountant.txt": """
LISA MARTINEZ
Senior Accountant - CPA
Email: lisa.martinez@email.com | Phone: (555) 678-9012
LinkedIn: linkedin.com/in/lisamartinez

PROFESSIONAL SUMMARY
--------------------
Detail-oriented Senior Accountant with 7 years of experience in financial reporting, 
tax compliance, and accounting operations. CPA certified with expertise in GAAP 
and financial analysis.

KEY SKILLS
----------
Accounting: Financial reporting, month-end close, reconciliation, auditing
Software: QuickBooks, SAP, Oracle, NetSuite, Excel (advanced)
Tax: Tax preparation, compliance, planning
Standards: GAAP, IFRS
Analysis: Financial analysis, variance analysis, budgeting, forecasting

PROFESSIONAL EXPERIENCE

Senior Accountant | Manufacturing Corp | 2019 - Present
--------------------------------------------------------
- Prepare monthly, quarterly, and annual financial statements
- Perform general ledger reconciliations and month-end close procedures
- Ensure compliance with GAAP accounting standards
- Manage accounts payable and accounts receivable processes
- Prepare and file quarterly tax returns (sales tax, payroll tax)
- Conduct financial analysis and variance analysis for management
- Assist with annual budgeting and forecasting process
- Coordinate with external auditors during year-end audits
- Maintain accurate accounting records and documentation

Accountant | Retail Solutions Inc. | 2017 - 2019
--------------------------------------------------
- Processed accounts payable and accounts receivable transactions
- Reconciled bank accounts and credit card statements
- Prepared journal entries and maintained general ledger
- Assisted with month-end and year-end closing procedures
- Generated financial reports using QuickBooks

Junior Accountant | Small Business Services | 2016 - 2017
----------------------------------------------------------
- Processed invoices and payments
- Maintained accounting records
- Assisted with bookkeeping tasks
- Prepared basic financial reports

EDUCATION
---------
Bachelor of Science in Accounting
Accounting University | 2016

CERTIFICATIONS
--------------
- Certified Public Accountant (CPA) - 2018
- QuickBooks Certified ProAdvisor
""",

    "resume_software_junior.txt": """
ALEX THOMPSON
Junior Software Developer
Email: alex.thompson@email.com | Phone: (555) 789-0123
LinkedIn: linkedin.com/in/alexthompson

PROFESSIONAL SUMMARY
--------------------
Recent computer science graduate with strong foundation in software development. 
Eager to contribute to a development team and learn from experienced engineers.

TECHNICAL SKILLS
----------------
Programming: JavaScript, Python, Java, C++
Web: HTML5, CSS3, React (basic), Node.js (basic)
Databases: SQL, MongoDB (basic)
Tools: Git, GitHub, VS Code

PROFESSIONAL EXPERIENCE

Software Development Intern | TechCompany | Summer 2023
--------------------------------------------------------
- Developed features for internal web application using React
- Fixed bugs and improved code quality
- Participated in code reviews
- Learned agile development practices

EDUCATION
---------
Bachelor of Science in Computer Science
State University | 2024

Relevant Coursework:
- Data Structures and Algorithms
- Software Engineering
- Database Systems
- Web Development

PROJECTS
--------
- Personal portfolio website using React
- Todo application with Node.js and MongoDB
- Weather app using REST API
""",

    "resume_hybrid_data_engineer.txt": """
JENNIFER PARK
Data Engineer / Data Scientist
Email: jennifer.park@email.com | Phone: (555) 890-1234
LinkedIn: linkedin.com/in/jenniferpark

PROFESSIONAL SUMMARY
--------------------
Data Engineer with 4 years of experience building data pipelines and infrastructure. 
Also skilled in data analysis and basic machine learning. Strong Python and SQL skills.

TECHNICAL SKILLS
----------------
Languages: Python, SQL, Scala, Java
Data Engineering: Spark, Hadoop, Airflow, Kafka, ETL pipelines
Databases: PostgreSQL, MongoDB, Snowflake, Redshift
Cloud: AWS (S3, EC2, Lambda), Azure
Data Science: pandas, scikit-learn (basic), Jupyter Notebooks
Tools: Git, Docker, Linux

PROFESSIONAL EXPERIENCE

Data Engineer | BigData Corp | 2020 - Present
-----------------------------------------------
- Built and maintained ETL pipelines processing 100GB+ daily
- Designed data warehouse architecture using Snowflake
- Optimized Spark jobs reducing processing time by 50%
- Created automated data quality checks and monitoring
- Worked with data scientists to deploy ML models to production
- Built real-time data streaming pipelines using Kafka
- Developed data ingestion systems from various sources (APIs, databases, files)

EDUCATION
---------
Bachelor of Science in Computer Engineering
Engineering University | 2020

PROJECTS
--------
- Open-source ETL framework for Python
- Real-time analytics dashboard
"""
}

# ============================================================================
# Generate Files
# ============================================================================

print("Generating test data files...")

# Generate Job Descriptions
print("\n📋 Generating Job Descriptions:")
for filename, content in job_descriptions.items():
    filepath = data_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"  ✓ Created {filename}")

# Generate Resumes
print("\n📄 Generating Resumes:")
for filename, content in resumes.items():
    filepath = data_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"  ✓ Created {filename}")

print(f"\n✅ Test data generation complete!")
print(f"\n📊 Summary:")
print(f"  - Job Descriptions: {len(job_descriptions)}")
print(f"  - Resumes: {len(resumes)}")
print(f"\n💡 Testing Guide:")
print(f"  - Software Developer JD → Should match: resume_software_dev_senior.txt, resume_software_junior.txt")
print(f"  - Data Scientist JD → Should match: resume_data_scientist.txt, resume_hybrid_data_engineer.txt")
print(f"  - Sales Manager JD → Should match: resume_sales_manager.txt")
print(f"  - Marketing Manager JD → Should match: resume_marketing_specialist.txt")
print(f"  - Product Manager JD → Should match: resume_product_manager.txt")
print(f"  - Accountant JD → Should match: resume_accountant.txt")
print(f"\n📁 All files saved in: {data_dir.absolute()}")

