"""
MajorMatch AI — College Major Recommendation Engine
=====================================================
A Streamlit app that recommends college majors based on
interests, grades, personality, and career goals.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MajorMatch AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6e0;
}

.stApp {
    background: linear-gradient(135deg, #0d0f14 0%, #111420 50%, #0d0f14 100%);
}

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 6vw, 4.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a5b4fc 0%, #e879f9 50%, #fb923c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}

.hero p {
    font-size: 1.1rem;
    color: #9ca3af;
    max-width: 500px;
    margin: 0 auto;
    font-weight: 300;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.4rem;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f1f0ec;
    margin: 0 0 1.5rem;
    line-height: 1.2;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}

.card:hover { border-color: rgba(99,102,241,0.4); }

/* ── Result cards ── */
.result-card {
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}

.result-card.rank-1 {
    background: linear-gradient(135deg, rgba(99,102,241,0.22) 0%, rgba(168,85,247,0.14) 100%);
    border: 1px solid rgba(99,102,241,0.45);
}
.result-card.rank-2 {
    background: linear-gradient(135deg, rgba(20,184,166,0.18) 0%, rgba(99,102,241,0.10) 100%);
    border: 1px solid rgba(20,184,166,0.35);
}
.result-card.rank-3 {
    background: linear-gradient(135deg, rgba(251,146,60,0.15) 0%, rgba(251,191,36,0.08) 100%);
    border: 1px solid rgba(251,146,60,0.30);
}

.result-rank {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.35rem;
}

.result-major {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f0ec;
    margin-bottom: 0.3rem;
}

.result-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.8rem;
}

.career-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #d1d5db;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > label,
div[data-testid="stMultiSelect"] > label,
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label,
div[data-testid="stCheckbox"] > label { color: #c9c7c0 !important; font-size: 0.95rem !important; }

.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: #6366f1 !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] { background: rgba(255,255,255,0.1) !important; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e8e6e0 !important;
}

.stButton > button {
    width: 100%;
    padding: 0.85rem 2.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.50);
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    margin: 0.2rem 0;
    transition: border-color 0.15s;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border-color: rgba(99,102,241,0.5);
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 2rem 0;
}

/* ── Step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 50%;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
    margin-right: 0.6rem;
    vertical-align: middle;
}

/* ── Info box ── */
.info-box {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    color: #c7d2fe;
    margin: 1rem 0;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA: MAJOR PROFILES
# ─────────────────────────────────────────────
MAJOR_PROFILES = {
    "Computer Science": {
        "emoji": "💻",
        "interests":    [5, 1, 2, 1, 3, 4, 2, 2],  # tech, arts, science, social, business, math, health, environment
        "grades":       [3, 5, 2, 2],               # english, math, science, social_studies
        "personality":  [5, 2, 4, 3, 5],            # analytical, creative, social, leadership, detail
        "career_goals": [5, 2, 3, 1, 4, 2],         # innovation, helping, stability, outdoors, high_income, travel
        "careers": ["Software Engineer", "Data Scientist", "AI/ML Engineer", "Product Manager", "Cybersecurity Analyst"],
        "description": "Build the software and systems that power the modern world.",
    },
    "Psychology": {
        "emoji": "🧠",
        "interests":    [1, 2, 3, 5, 1, 1, 4, 2],
        "grades":       [4, 2, 4, 5],
        "personality":  [3, 3, 5, 3, 4],
        "career_goals": [2, 5, 3, 1, 3, 2],
        "careers": ["Clinical Psychologist", "Counselor", "HR Specialist", "UX Researcher", "Social Worker"],
        "description": "Understand the human mind and help people live better lives.",
    },
    "Business Administration": {
        "emoji": "📈",
        "interests":    [3, 2, 1, 4, 5, 3, 1, 1],
        "grades":       [4, 4, 2, 4],
        "personality":  [3, 3, 4, 5, 3],
        "career_goals": [3, 2, 4, 1, 5, 3],
        "careers": ["Management Consultant", "Marketing Manager", "Entrepreneur", "Financial Analyst", "Operations Director"],
        "description": "Lead organizations, drive strategy, and create economic value.",
    },
    "Biomedical Engineering": {
        "emoji": "🔬",
        "interests":    [4, 1, 5, 2, 2, 5, 4, 3],
        "grades":       [3, 5, 5, 2],
        "personality":  [5, 3, 3, 3, 5],
        "career_goals": [4, 4, 3, 1, 4, 2],
        "careers": ["Biomedical Engineer", "Medical Device Designer", "Research Scientist", "Regulatory Affairs Specialist", "Clinical Engineer"],
        "description": "Merge engineering and medicine to create life-saving technologies.",
    },
    "Environmental Science": {
        "emoji": "🌱",
        "interests":    [2, 3, 5, 3, 2, 3, 2, 5],
        "grades":       [3, 3, 5, 4],
        "personality":  [4, 4, 3, 3, 4],
        "career_goals": [4, 4, 2, 5, 2, 4],
        "careers": ["Environmental Consultant", "Conservation Scientist", "Climate Analyst", "Park Ranger", "Sustainability Manager"],
        "description": "Protect ecosystems and tackle the planet's most urgent challenges.",
    },
    "Fine Arts / Design": {
        "emoji": "🎨",
        "interests":    [2, 5, 2, 3, 2, 1, 1, 3],
        "grades":       [5, 1, 2, 3],
        "personality":  [2, 5, 3, 3, 5],
        "career_goals": [5, 2, 2, 3, 3, 3],
        "careers": ["Graphic Designer", "UX/UI Designer", "Art Director", "Illustrator", "Creative Director"],
        "description": "Shape culture and communication through visual creativity.",
    },
    "Nursing / Healthcare": {
        "emoji": "🏥",
        "interests":    [2, 1, 4, 5, 1, 2, 5, 2],
        "grades":       [4, 3, 5, 3],
        "personality":  [3, 3, 5, 3, 5],
        "career_goals": [2, 5, 4, 2, 4, 2],
        "careers": ["Registered Nurse", "Nurse Practitioner", "Healthcare Administrator", "Physical Therapist", "Physician Assistant"],
        "description": "Provide compassionate care and make a daily difference in people's lives.",
    },
    "Political Science / Law": {
        "emoji": "⚖️",
        "interests":    [1, 3, 2, 5, 3, 2, 2, 3],
        "grades":       [5, 2, 2, 5],
        "personality":  [4, 3, 4, 5, 3],
        "career_goals": [3, 4, 3, 2, 4, 3],
        "careers": ["Lawyer", "Policy Analyst", "Politician", "Diplomat", "Non-profit Director"],
        "description": "Shape policy, advocate for justice, and influence society.",
    },
    "Data Science / Statistics": {
        "emoji": "📊",
        "interests":    [5, 1, 4, 2, 3, 5, 2, 2],
        "grades":       [3, 5, 4, 3],
        "personality":  [5, 2, 2, 3, 5],
        "career_goals": [4, 2, 4, 1, 5, 2],
        "careers": ["Data Analyst", "Quantitative Researcher", "Business Intelligence Engineer", "Statistician", "Machine Learning Engineer"],
        "description": "Turn raw numbers into insights that drive decisions.",
    },
    "Education": {
        "emoji": "📚",
        "interests":    [2, 3, 3, 5, 2, 2, 3, 2],
        "grades":       [5, 3, 3, 4],
        "personality":  [3, 4, 5, 4, 3],
        "career_goals": [2, 5, 3, 3, 2, 2],
        "careers": ["Teacher", "School Counselor", "Curriculum Designer", "Educational Technologist", "Principal"],
        "description": "Inspire the next generation and shape the future of learning.",
    },
}

# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def build_user_vector(interests, grades, personality, career_goals):
    """Combine all user inputs into a single feature vector."""
    return np.array(interests + grades + personality + career_goals, dtype=float)

def build_major_matrix():
    """Build a matrix where each row is a major's profile vector."""
    rows, names = [], []
    for major, profile in MAJOR_PROFILES.items():
        vec = (
            profile["interests"]
            + profile["grades"]
            + profile["personality"]
            + profile["career_goals"]
        )
        rows.append(vec)
        names.append(major)
    return np.array(rows, dtype=float), names

def get_recommendations(user_vector, top_n=3):
    """Return top_n majors with match percentages."""
    major_matrix, names = build_major_matrix()
    sim = cosine_similarity([user_vector], major_matrix)[0]
    # Scale to 55–99 for a friendlier display
    sim_scaled = 55 + (sim - sim.min()) / (sim.max() - sim.min() + 1e-9) * 44
    top_idx = sim_scaled.argsort()[::-1][:top_n]
    return [(names[i], round(sim_scaled[i], 1)) for i in top_idx]

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
FONT_COLOR = "#9ca3af"

def make_bar_chart(recommendations):
    majors = [r[0] for r in recommendations]
    scores = [r[1] for r in recommendations]
    colors = ["#6366f1", "#14b8a6", "#fb923c"]

    fig = go.Figure(go.Bar(
        x=scores,
        y=majors,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{s}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#e8e6e0", size=14, family="Syne"),
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(l=0, r=60, t=10, b=10),
        height=200,
        xaxis=dict(
            range=[0, 110],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#e8e6e0", size=12, family="DM Sans"),
            autorange="reversed",
            showgrid=False,
            ticksuffix="  ",
        ),
        showlegend=False,
        font=dict(color=FONT_COLOR),
    )
    return fig


def make_radar_chart(recommendations):
    categories = [
        "Technology", "Arts", "Science", "Social",
        "Business", "Math", "Health", "Environment"
    ]

    fig = go.Figure()
    colors = [
        ("rgba(99,102,241,0.25)", "#6366f1"),
        ("rgba(20,184,166,0.20)", "#14b8a6"),
        ("rgba(251,146,60,0.18)", "#fb923c"),
    ]

    for (major, _), (fill, line) in zip(recommendations, colors):
        vals = MAJOR_PROFILES[major]["interests"]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=fill,
            line=dict(color=line, width=2),
            name=major,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                gridcolor=GRID_COLOR,
                tickfont=dict(color=FONT_COLOR, size=9),
                showline=False,
            ),
            angularaxis=dict(
                tickfont=dict(color="#d1d5db", size=11, family="DM Sans"),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
        ),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        showlegend=True,
        legend=dict(
            font=dict(color="#d1d5db", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=30, r=30, t=30, b=30),
        height=350,
        font=dict(color=FONT_COLOR),
    )
    return fig


def make_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"color": "#f1f0ec", "size": 32, "family": "Syne"}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=FONT_COLOR, tickfont=dict(color=FONT_COLOR)),
            bar=dict(color=color),
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.08)",
            steps=[dict(range=[0, 100], color="rgba(255,255,255,0.03)")],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=score),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        height=160,
        font=dict(color=FONT_COLOR),
    )
    return fig

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>MajorMatch AI</h1>
  <p>Answer a few questions — discover the college major that fits your mind, strengths, and ambitions.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────
with st.form("major_match_form"):

    # ── Step 1: Interests ──────────────────────
    st.markdown('<div class="section-label">Step 1 of 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 What are your interests?</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Rate each area from 1 (not interested) to 5 (very passionate).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    interest_labels = [
        ("Technology & Coding", "💻", col1),
        ("Arts & Creativity",   "🎨", col2),
        ("Science & Research",  "🔬", col1),
        ("People & Society",    "🤝", col2),
        ("Business & Finance",  "📈", col1),
        ("Math & Statistics",   "📐", col2),
        ("Health & Medicine",   "🏥", col1),
        ("Nature & Environment","🌱", col2),
    ]
    interest_scores = []
    for label, emoji, col in interest_labels:
        with col:
            val = st.slider(f"{emoji} {label}", 1, 5, 3, key=f"int_{label}")
            interest_scores.append(val)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Step 2: Grades ─────────────────────────
    st.markdown('<div class="section-label">Step 2 of 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 How are your grades?</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Rate your performance in each subject from 1 (struggling) to 5 (excellent).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    grade_labels = [
        ("English / Writing",  "✍️",  col1),
        ("Mathematics",        "🔢",  col2),
        ("Science",            "⚗️",  col1),
        ("Social Studies",     "🗺️",  col2),
    ]
    grade_scores = []
    for label, emoji, col in grade_labels:
        with col:
            val = st.slider(f"{emoji} {label}", 1, 5, 3, key=f"gr_{label}")
            grade_scores.append(val)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Step 3: Personality ────────────────────
    st.markdown('<div class="section-label">Step 3 of 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧩 What describes you?</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Rate how strongly each trait describes you from 1 (not at all) to 5 (very much).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    personality_labels = [
        ("Analytical thinker",        "🔍", col1),
        ("Creative & imaginative",     "✨", col2),
        ("Social & collaborative",     "👥", col1),
        ("Natural leader",             "🚀", col2),
        ("Detail-oriented",            "🎯", col1),
    ]
    personality_scores = []
    for label, emoji, col in personality_labels:
        with col:
            val = st.slider(f"{emoji} {label}", 1, 5, 3, key=f"per_{label}")
            personality_scores.append(val)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Step 4: Career Goals ───────────────────
    st.markdown('<div class="section-label">Step 4 of 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 What do you want from your career?</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Rate how important each goal is to you from 1 (not important) to 5 (essential).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    career_labels = [
        ("Innovation & cutting-edge work", "⚡", col1),
        ("Helping & supporting others",    "💛", col2),
        ("Job stability & security",       "🏛️", col1),
        ("Outdoor / field work",           "🌲", col2),
        ("High income potential",          "💰", col1),
        ("Travel & global exposure",       "✈️", col2),
    ]
    career_scores = []
    for label, emoji, col in career_labels:
        with col:
            val = st.slider(f"{emoji} {label}", 1, 5, 3, key=f"car_{label}")
            career_scores.append(val)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("✨ Find My Top Majors")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if submitted:
    user_vec = build_user_vector(interest_scores, grade_scores, personality_scores, career_scores)
    results = get_recommendations(user_vec, top_n=3)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎓 Your Top 3 Major Matches</div>', unsafe_allow_html=True)

    rank_classes = ["rank-1", "rank-2", "rank-3"]
    rank_labels  = ["#1 Best Match", "#2 Strong Match", "#3 Good Match"]
    gauge_colors = ["#6366f1", "#14b8a6", "#fb923c"]

    # ── Result cards ──────────────────────────
    for i, (major, score) in enumerate(results):
        profile = MAJOR_PROFILES[major]
        careers_html = "".join(
            f'<span class="career-pill">🔸 {c}</span>' for c in profile["careers"]
        )
        st.markdown(f"""
        <div class="result-card {rank_classes[i]}">
            <div class="result-rank">{rank_labels[i]}</div>
            <div class="result-major">{profile["emoji"]} {major}</div>
            <div class="result-pct">{score}% match</div>
            <p style="color:#9ca3af; font-size:0.92rem; margin:0 0 0.9rem;">{profile["description"]}</p>
            <div><strong style="color:#e8e6e0; font-size:0.82rem; font-family:'Syne',sans-serif;
                  letter-spacing:0.08em; text-transform:uppercase;">Career Paths</strong></div>
            <div style="margin-top:0.4rem">{careers_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────
    st.markdown('<div class="section-title">📊 Visual Breakdown</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Match Scores**")
        st.plotly_chart(make_bar_chart(results), use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("**Interest Profile Comparison**")
        st.plotly_chart(make_radar_chart(results), use_container_width=True, config={"displayModeBar": False})

    # ── Gauge row ─────────────────────────────
    st.markdown("**Match Confidence Gauges**")
    gcol1, gcol2, gcol3 = st.columns(3)
    for col, (major, score), gc in zip([gcol1, gcol2, gcol3], results, gauge_colors):
        with col:
            st.markdown(f"<div style='text-align:center;font-size:0.85rem;color:#9ca3af;margin-bottom:-0.5rem'>{major}</div>", unsafe_allow_html=True)
            st.plotly_chart(make_gauge(score, gc), use_container_width=True, config={"displayModeBar": False})

    # ── Detailed comparison table ─────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Side-by-Side Comparison</div>', unsafe_allow_html=True)

    comparison_data = {
        "Attribute": ["Match %", "Top Career", "Median Salary*", "Avg. GPA Needed", "Job Growth*"],
        results[0][0]: [
            f"{results[0][1]}%",
            MAJOR_PROFILES[results[0][0]]["careers"][0],
            _salary := {"Computer Science":"$110k","Psychology":"$55k","Business Administration":"$75k",
                        "Biomedical Engineering":"$95k","Environmental Science":"$65k","Fine Arts / Design":"$58k",
                        "Nursing / Healthcare":"$80k","Political Science / Law":"$72k",
                        "Data Science / Statistics":"$108k","Education":"$52k"}.get(results[0][0], "—"),
            "3.2–3.6",
            "High",
        ],
        results[1][0]: [
            f"{results[1][1]}%",
            MAJOR_PROFILES[results[1][0]]["careers"][0],
            {"Computer Science":"$110k","Psychology":"$55k","Business Administration":"$75k",
             "Biomedical Engineering":"$95k","Environmental Science":"$65k","Fine Arts / Design":"$58k",
             "Nursing / Healthcare":"$80k","Political Science / Law":"$72k",
             "Data Science / Statistics":"$108k","Education":"$52k"}.get(results[1][0], "—"),
            "2.8–3.4",
            "Moderate",
        ],
        results[2][0]: [
            f"{results[2][1]}%",
            MAJOR_PROFILES[results[2][0]]["careers"][0],
            {"Computer Science":"$110k","Psychology":"$55k","Business Administration":"$75k",
             "Biomedical Engineering":"$95k","Environmental Science":"$65k","Fine Arts / Design":"$58k",
             "Nursing / Healthcare":"$80k","Political Science / Law":"$72k",
             "Data Science / Statistics":"$108k","Education":"$52k"}.get(results[2][0], "—"),
            "2.8–3.3",
            "Moderate",
        ],
    }

    df = pd.DataFrame(comparison_data).set_index("Attribute")
    st.dataframe(df, use_container_width=True)
    st.markdown("<p style='color:#6b7280;font-size:0.78rem'>* Approximate US figures. Salaries and growth vary by location, experience, and specialization.</p>", unsafe_allow_html=True)

    # ── Restart hint ──────────────────────────
    st.markdown("""
    <div class="info-box" style="margin-top:2rem;text-align:center;">
        🔄 Want to try different answers? Adjust the sliders above and click <strong>Find My Top Majors</strong> again.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 1.5rem;color:#4b5563;font-size:0.82rem;border-top:1px solid rgba(255,255,255,0.05);margin-top:3rem;">
    MajorMatch AI &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; Powered by scikit-learn cosine similarity
</div>
""", unsafe_allow_html=True)
