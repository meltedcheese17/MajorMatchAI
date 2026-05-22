# 🎓 MajorMatch AI

A modern Streamlit web app that recommends college majors based on your
interests, grades, personality, and career goals — powered by scikit-learn
cosine similarity.

---

## ✨ Features

- **4-section quiz** — interests, grades, personality, career goals
- **AI-style matching** — cosine similarity via scikit-learn
- **Top 3 major recommendations** with match percentages
- **Career path suggestions** for each major
- **Interactive charts** — bar chart, radar chart, gauge meters
- **Comparison table** — salary, GPA, job growth side-by-side
- **Dark modern UI** — custom CSS with gradient accents

---

## 🚀 How to Run Locally (Step-by-Step)

### Step 1 — Make sure Python is installed

Open your terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
python --version
```

You need Python **3.9 or higher**. Download it from https://python.org if needed.

---

### Step 2 — Download the project files

Save both files into the same folder on your computer:

```
my-project/
├── app.py
└── requirements.txt
```

---

### Step 3 — Open your terminal in that folder

**Mac / Linux:**
```bash
cd path/to/my-project
```

**Windows:**
```cmd
cd C:\path\to\my-project
```

---

### Step 4 — (Recommended) Create a virtual environment

This keeps your project's packages separate from the rest of your system:

```bash
# Create the virtual environment
python -m venv venv

# Activate it — Mac/Linux:
source venv/bin/activate

# Activate it — Windows:
venv\Scripts\activate
```

You'll see `(venv)` appear in your terminal prompt when it's active.

---

### Step 5 — Install the dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `plotly`.

---

### Step 6 — Run the app

```bash
streamlit run app.py
```

Your browser will automatically open to **http://localhost:8501** 🎉

---

### Step 7 — Use the app

1. Use the sliders to rate your interests, grades, personality, and career goals
2. Click **✨ Find My Top Majors**
3. View your top 3 recommended majors with match percentages, career paths, and charts

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `streamlit: command not found` | Make sure your virtual env is activated, then re-run `pip install -r requirements.txt` |
| Browser doesn't open | Manually visit http://localhost:8501 |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## 🧠 How It Works

1. **User inputs** (24 slider values) are combined into a single feature vector
2. Each of the 10 college majors has a pre-defined profile vector
3. **Cosine similarity** (from scikit-learn) measures how closely the user vector matches each major's vector
4. Similarity scores are scaled to the 55–99% range for a friendly display
5. The top 3 matches are returned with career suggestions and visualizations

---

## 📦 Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI and interactivity |
| `pandas` | Comparison data table |
| `scikit-learn` | Cosine similarity engine |
| `numpy` | Vector math |
| `plotly` | Bar, radar, and gauge charts |

---

## 🗂️ Project Structure

```
app.py              ← All application code (self-contained)
requirements.txt    ← Python package dependencies
README.md           ← This file
```

---

Built with ❤️ using Streamlit.
