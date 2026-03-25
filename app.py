import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="AI Job Advisor 2030", layout="wide")

st.title("🔮 AI Job Impact & Career Intelligence Advisor (2030)")
st.caption("AI Skills • Risk Modeling • Salary Forecast • Career Roadmap • Domain Intelligence")

# ----------------------------------------------------
# 1. LOAD FULL FEATURE DATASET
# ----------------------------------------------------
try:
    df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")
    st.success("✔ Full Feature Dataset Loaded Successfully!")
except:
    st.error("❌ Cannot find dataset. Ensure the CSV is in the same folder as app.py.")
    st.stop()

job_titles = df["Job_Title"].tolist()

# ----------------------------------------------------
# 2. DOMAIN DEFINITIONS
# ----------------------------------------------------
DOMAINS = [
    "Software Development & Systems Engineering",
    "Artificial Intelligence, Machine Learning & Data Science",
    "Cloud Computing, DevOps & Site Reliability",
    "Cybersecurity & Digital Forensics",
    "Financial Services, Banking & Wealth Management",
    "Accounting, Audit & Corporate Finance",
    "Healthcare, Diagnostics & Medical Assistance",
    "Nursing, Patient Care & Clinical Services",
    "Pharmaceuticals, Biotech & Life Sciences",
    "Manufacturing, Industrial Engineering & Robotics",
    "Mechanical, Electrical & Civil Engineering",
    "Construction, Architecture & Building Technology",
    "Retail, eCommerce & Customer Experience",
    "Sales, Business Development & Account Management",
    "Marketing, Advertising & Digital Growth",
    "Customer Support, Service & Communication",
    "Logistics, Supply Chain & Transportation",
    "Government, Public Services & Administration",
    "Law Enforcement, Defense & Security",
    "Education, Training & Curriculum Development",
    "Media, Content, Advertising & Digital Creation",
    "Creative Arts, Design & UI/UX",
    "Research, Innovation & Scientific Exploration",
    "Energy, Oil & Utilities",
    "Renewable Energy & Sustainability",
    "Travel, Tourism & Hospitality",
    "Automotive Engineering & Vehicle Technology",
    "Aerospace, Aviation & Aeronautics",
    "Agriculture, Food Science & Crop Technology",
    "Telecommunications & Network Infrastructure",
    "Human Resources, Talent Development & People Operations",
    "Business Strategy, Consulting & Leadership",
    "Real Estate, Property Management & Urban Planning",
    "Legal, Compliance & Regulatory Affairs",
    "Insurance & Risk Management",
    "Data Engineering, Big Data & Analytics",
    "IT Support, Infrastructure & System Administration",
    "Gaming, Simulation & Interactive Media",
    "Fashion, Clothing & Textile Technology",
    "Sports, Fitness & Performance Coaching",
    "Social Work, Counseling & Community Services",
    "Environment, Ecology & Waste Management",
    "Mining, Metals & Natural Resources",
    "Marine, Shipping & Port Operations",
    "Cyber-Physical Systems & Internet of Things (IoT)",
    "Operations, Quality Control & Process Management",
    "Procurement, Vendor & Contract Management",
    "Drone Technology, UAV Systems & Autonomous Robotics",
]

# ----------------------------------------------------
# 3. SKILL GRAPH FOR ROADMAP
# ----------------------------------------------------
SKILL_GRAPH = {dom: [
    "AI Literacy", "Prompt Engineering", "Cloud Basics", "Data Analysis",
    "Automation Tools", "Digital Communication", "Problem Solving",
    "Critical Thinking", "AI-Assisted Workflows"
] for dom in DOMAINS}

SKILL_GRAPH["Artificial Intelligence, Machine Learning & Data Science"] += [
    "Deep Learning", "NLP", "TensorFlow", "PyTorch", "RAG Systems",
    "LLM Fine-Tuning", "MLOps", "Vector Databases", "Model Evaluation"
]

SKILL_GRAPH["Software Development & Systems Engineering"] += [
    "Python", "System Design", "APIs", "Distributed Systems",
    "Microservices", "DevOps", "CI/CD", "Kubernetes"
]

# ----------------------------------------------------
# 4. AI ASSIST CUSTOM TASKS (FOR POPULAR ROLES)
# ----------------------------------------------------
AI_ASSIST_TASKS = {
    "Software Engineer": [
        "Code generation & debugging",
        "Automated testing",
        "System design suggestions",
        "API documentation",
        "Refactoring optimization",
        "CI/CD automation",
        "Security scanning",
        "Performance profiling",
        "Architecture review",
        "Code quality enhancement"
    ],
    "Data Scientist": [
        "Data cleaning automation",
        "Feature engineering",
        "Model selection",
        "Auto ML tuning",
        "Visualization creation",
        "Insight summarization",
        "SQL query generation",
        "Pipeline automation",
        "Scenario simulation",
        "Report generation"
    ],
    "AI/ML Engineer": [
        "Model architecture creation",
        "Dataset augmentation",
        "Training pipeline setup",
        "LLM fine-tuning",
        "RAG workflows",
        "Experiment tracking",
        "Error analysis",
        "Model deployment",
        "GPU optimization",
        "Auto hyperparameter tuning"
    ]
}

# ----------------------------------------------------
# 5. AUTO-GENERATED TASKS FOR ALL OTHER JOBS
# ----------------------------------------------------
def generate_ai_tasks(job_title):
    return [
        f"AI-powered workflow automation for {job_title}",
        f"Smart productivity optimization for {job_title}",
        f"Predictive analytics to improve {job_title} tasks",
        f"AI-based error detection in {job_title} activities",
        f"Automated reporting & documentation support for {job_title}",
        f"AI-driven risk & safety analysis for {job_title}",
        f"Resource usage optimization for {job_title}",
        f"Real-time voice-guided assistance for {job_title}",
        f"Scheduling & planning automation for {job_title}",
        f"Quality inspection automation for {job_title}"
    ]

# ----------------------------------------------------
# 6. LOAD EMBEDDING MODEL
# ----------------------------------------------------
st.info("⚙ Loading AI Model…")
model = SentenceTransformer("all-MiniLM-L6-v2")
job_embeddings = model.encode(job_titles)
domain_embeddings = model.encode(DOMAINS)

job_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(job_embeddings)
domain_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(domain_embeddings)

# ----------------------------------------------------
# 7. CURRENCY SELECTION
# ----------------------------------------------------
currency_rates = {
    "INR (₹)": 1,
    "USD ($)": 1/83.21,
    "EUR (€)": 1/90,
    "GBP (£)": 1/105
}

currency = st.selectbox("💱 Select Salary Currency", list(currency_rates.keys()))
rate = currency_rates[currency]
symbol = currency.split()[1]

# ----------------------------------------------------
# 8. JOB SELECTION (TOP 10)
# ----------------------------------------------------
TOP_10_JOBS = df["Job_Title"].head(10).tolist()
job_input = st.selectbox("🔍 Select a Job Role", TOP_10_JOBS)

# ----------------------------------------------------
# 9. MAIN AI ENGINE
# ----------------------------------------------------
if job_input:
    user_emb = model.encode([job_input])
    _, idx = job_nn.kneighbors(user_emb)
    match = df.iloc[idx[0][0]]

    # Predict domain
    _, dom_idx = domain_nn.kneighbors(user_emb)
    domain = DOMAINS[dom_idx[0][0]]

    st.header(f"🎯 Closest Match: **{match['Job_Title']}**")
    st.subheader(f"🏢 Domain: {domain}")

    fresher_salary = match["Fresher_Salary_INR"] * rate

    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🤖 Automation Probability", f"{match['Automation_Probability_2030']:.2f}")
    col2.metric("💰 Fresher Salary", f"{symbol}{fresher_salary:,.0f}")
    col3.metric("📈 Tech Growth Factor", f"{match['Tech_Growth_Factor']:.2f}")
    col4.metric("🧠 AI Assist Boost", f"{match['AI_Assist_Boost']:.2f}")

    # ----------------------------------------------------
    # AI Assist Tasks (NOW WORKS FOR ALL JOBS)
    # ----------------------------------------------------
    st.subheader("🤖 How AI Will Assist You (Top 10 Tasks)")

    role = match["Job_Title"]

    if role in AI_ASSIST_TASKS:
        tasks = AI_ASSIST_TASKS[role]
    else:
        tasks = generate_ai_tasks(role)

    for t in tasks:
        st.write("✔", t)

    # SURVIVAL SCORE
    st.subheader("🛡 AI Job Survival Score")
    survival = (1 - match["Automation_Probability_2030"]) * 100
    final_survival = min(100, survival + match["Tech_Growth_Factor"] * 10)

    colA, colB = st.columns(2)
    colA.metric("🔺 Automation Risk", f"{100 - survival:.1f}/100")
    colB.metric("🛡 Survival Score", f"{final_survival:.1f}/100")

    # SALARY FORECAST
    st.subheader("📈 Salary Forecast (5-Year AI Projection)")
    growth_curve = np.linspace(
        fresher_salary,
        fresher_salary * (1 + match["Tech_Growth_Factor"] * 0.30),
        5
    )

    fig = px.line(pd.DataFrame({
        "Year": ["2025", "2026", "2027", "2028", "2029"],
        "Salary": growth_curve
    }), x="Year", y="Salary", markers=True)
    st.plotly_chart(fig)

    # RADAR CHART
    st.subheader("📊 Required Skills Difficulty Map")
    radar_df = pd.DataFrame({
        "Skill": ["AI Literacy", "Prompt Engineering", "Critical Thinking",
                  "Automation Tools", "Data Analysis", "Digital Communication"],
        "Difficulty": [7, 8, 6, 5, 7, 4]
    })

    fig2 = px.line_polar(radar_df, r="Difficulty", theta="Skill", line_close=True)
    st.plotly_chart(fig2)

    # CAREER PATH
    st.subheader("🚀 Career Growth Path (Salary Projection)")
    levels = ["Fresher", "Junior", "Mid-Level", "Senior", "Lead"]
    base = fresher_salary
    multipliers = [1, 1.6, 2.4, 3.6, 5.1]

    fig3 = px.line(pd.DataFrame({
        "Level": levels,
        "Salary": [base * m for m in multipliers]
    }), x="Level", y="Salary", markers=True)
    st.plotly_chart(fig3)

    # JOB INFORMATION PANEL
    st.subheader("📌 Job Information Overview")
    st.write(f"**Job Type:** {match['Job_Type']}")
    st.write(f"**Work Mode:** {match['Work_Mode']}")
    st.write(f"**Degree Needed:** {match['Degree_Needed']}")
    st.write(f"**Industry Demand (2025):** {match['Industry_Demand_2025']}/100")
    st.write(f"**Learning Difficulty:** {match['Learning_Difficulty']}/10")
    st.write(f"**Skill Gap Index:** {match['Skill_Gap_Index']}/100")
    st.write(f"**Stress Level:** {match['Stress_Level']}/10")
    st.write(f"**Work-Life Balance:** {match['WLB_Score']}/10")
    st.write(f"**Future Category:** {match['Future_Category']}")

    # TOP SKILLS REQUIRED
    st.subheader("🔥 Top Skills Required")
    for skill in match["Top_Skills"].strip("[]").replace("'", "").split(","):
        st.write("✔", skill.strip())

    # ROADMAP
    st.subheader("🛠 Roadmap to Become This Role")
    st.write("### 🧩 1️⃣ Foundation Skills")
    st.write("- Learn domain fundamentals")
    st.write("- Build basic digital literacy")
    st.write("- Improve problem solving")

    st.write("### 🧩 2️⃣ Core Technical Skills")
    for s in SKILL_GRAPH[domain][:8]:
        st.write("✔", s)

    st.write("### 🚀 3️⃣ Advanced AI Skills")
    for s in SKILL_GRAPH[domain][8:15]:
        st.write("🚀", s)

    st.write("### 🧩 4️⃣ Gain Experience")
    st.write("- Internships / Freelancing")
    st.write("- Real projects")
    st.write("- Open-source contributions")

    st.write("### 🧩 5️⃣ Become Industry Ready")
    st.write("- Certifications")
    st.write("- Portfolio website")
    st.write("- Hackathons & coding challenges")

    st.success("✔ Full AI Analysis Completed!")
