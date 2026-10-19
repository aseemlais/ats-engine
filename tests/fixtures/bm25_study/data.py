"""SYNTHETIC study data for the BM25 query-strategy comparison (no real candidate or employer is described).
JDs are written to contain realistic generic boilerplate; resumes are short templates. Used by
scripts/bm25_query_strategy_compare.py and tests/test_bm25_query_modes.py. Mechanical evidence only: there are NO human
relevance labels, so nothing here measures hiring accuracy."""

JD_SENIOR_DE = """Senior Data Engineer
About the role
We are a fast-paced, innovative company looking for a passionate and self-motivated Senior Data Engineer to join our team.
Responsibilities
- Design, build and optimise scalable ETL pipelines using Spark and Airflow
- Collaborate with cross-functional teams and stakeholders in a fast-paced environment
- Ensure data quality, monitoring and alerting for production pipelines on AWS
- Take ownership, demonstrate strong leadership and excellent communication skills
- Mentor junior engineers and drive best practices with a positive attitude
Requirements
- 5+ years of experience in data engineering
- Strong proficiency in Python and SQL (required)
- Hands-on experience with Apache Spark and Airflow
- Experience with AWS, Azure or GCP
Nice to have
- Kafka, Terraform, Docker
Compensation
- Salary 25-35 LPA, hybrid, 3 interview rounds, notice period 30 days
"""

JD_NURSE = """Registered Nurse - ICU
About the role
We are a dynamic, patient-focused hospital seeking a dedicated and motivated Registered Nurse for our intensive care unit.
Responsibilities
- Provide direct patient care in the intensive care unit and monitor vital signs
- Administer medications and maintain accurate patient charts and documentation
- Collaborate with doctors, stakeholders and the wider team in a fast-paced environment
- Respond to emergencies and perform life support procedures
- Demonstrate excellent communication, strong teamwork and a positive attitude
Requirements
- Bachelor's degree in Nursing and a valid RN licence
- 2+ years of ICU experience
- BLS and ACLS certification (required)
Benefits
- Night shift allowance, health insurance, relocation support, interview in 2 rounds
"""

JD_MARKETING = """Digital Marketing Manager
About the role
We are an innovative, growth-focused company looking for a passionate and results-oriented Digital Marketing Manager.
Responsibilities
- Plan and execute SEO and SEM campaigns and manage Google Ads budgets
- Analyse campaign performance with Google Analytics and report on conversion
- Own the content calendar and email marketing programme
- Collaborate with cross-functional teams and stakeholders in a fast-paced environment
- Demonstrate strong leadership, excellent communication and ownership
Requirements
- 4+ years of experience in digital marketing
- Proven experience with SEO, Google Ads and Google Analytics (required)
- Experience with HubSpot or similar marketing automation tools
Perks
- Salary 12-18 LPA, hybrid working, 2 interview rounds
"""

_GEN = ("Passionate, self-motivated team player with excellent communication skills and strong leadership. Collaborated with "
        "cross-functional stakeholders in a fast-paced environment. Proven ability to work independently, take ownership, "
        "demonstrate a positive attitude and deliver results. Dedicated, reliable, adaptable and eager to learn. "
        "Willing to contribute to team success in a dynamic organization.")

# id -> (domain, text)
SYN_RESUMES = {
    "SYN_DE_FRESHER_1": ("de_fresher", "Fresher. B.Tech Computer Science. Python, SQL, MySQL, Pandas. Learned PySpark and Databricks in a training programme. Built an ETL pipeline project loading CSV files into a MySQL database with Python. Certification: SQL for data science."),
    "SYN_DE_FRESHER_2": ("de_fresher", "Graduate trainee. Python programming, SQL queries with joins and window functions, DBMS fundamentals. Project: data pipeline using Apache Spark and cloud storage on AWS. Familiar with ETL concepts and Databricks notebooks."),
    "SYN_DE_SENIOR_1": ("de_senior", "Senior Data Engineer, 7 years. Designed and optimised ETL pipelines with Spark and Airflow on AWS (S3, Glue, Redshift). Python and SQL daily. Kafka streaming ingestion, Terraform and Docker for infrastructure. Mentored junior engineers. Data quality monitoring and alerting."),
    "SYN_DE_SENIOR_2": ("de_senior", "Lead data engineer with 9 years of experience. Built data platforms on Azure and GCP, Apache Spark batch jobs orchestrated by Airflow, Python services, advanced SQL, warehouse modelling. Production pipelines, monitoring, on-call ownership, Docker."),
    "SYN_NURSE_1": ("nurse", "Registered Nurse, 4 years ICU experience. Direct patient care, vital signs monitoring, medication administration, patient charts and documentation. BLS and ACLS certified. Responded to emergencies and life support events. Licensed RN, Bachelor of Science in Nursing."),
    "SYN_NURSE_2": ("nurse", "Staff nurse intensive care unit, 3 years. Patient monitoring, ventilator care, administered medications, accurate documentation of patient charts, coordinated with doctors. BLS, ACLS. RN licence. B.Sc Nursing."),
    "SYN_NURSE_3": ("nurse", "Nursing graduate, 1 year ward experience. Patient care, vital signs, medication rounds, charting. BLS certified. Registered nurse licence."),
    "SYN_MKT_1": ("marketing", "Digital marketing manager, 6 years. Planned and executed SEO and SEM campaigns, managed Google Ads budgets, analysed performance in Google Analytics, conversion reporting. Email marketing programme in HubSpot. Content calendar ownership."),
    "SYN_MKT_2": ("marketing", "Marketing specialist 4 years. SEO, Google Ads campaigns, Google Analytics dashboards, HubSpot marketing automation, email campaigns and content planning."),
    "SYN_MKT_3": ("marketing", "Social media and content marketer, 2 years. Content calendar, email marketing, basic SEO and Google Analytics reporting."),
    "SYN_ACC_1": ("accounting", "Accountant, 5 years. Accounts payable, general ledger, month end close, bank reconciliation, tax filing, GAAP reporting, Excel and Tally."),
    "SYN_ACC_2": ("accounting", "Finance executive. Invoicing, audit support, budgeting, reconciliations, financial statements, payroll processing."),
    "SYN_MECH_1": ("mechanical", "Mechanical engineer, 6 years. CAD design in SolidWorks, finite element analysis, manufacturing tolerances, CNC machining, quality inspection, ISO 9001."),
    "SYN_GENERIC_1": ("generic_only", "Customer service associate. " + _GEN),
    "SYN_GENERIC_2": ("generic_only", "Office assistant. " + _GEN + " " + _GEN),
    "SYN_GENERIC_3": ("generic_only", "Administrative coordinator. " + _GEN),
}

# JD key -> (file or text, positive domains by construction)
JD_SPECS = {
    "fresher_technical": ("FILE:tests/fixtures/jd_user_trainee_full.txt", {"de_fresher"}),
    "experienced_technical": (JD_SENIOR_DE, {"de_senior"}),
    "senior_data_analyst": ("FILE:data/resumes/JD_Senior_Data_Analyst.txt", set()),   # no synthetic positives defined
    "other_domain_nursing": (JD_NURSE, {"nurse"}),
    "other_domain_marketing": (JD_MARKETING, {"marketing"}),
}
