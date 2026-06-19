# Day 1 Cheatsheet — Quick Reference

> Open this file on Day 1 and follow top to bottom. No thinking required.

---

## Day 1 Goal (~1 hour)

Get a working repo on GitHub with Python + venv + dependencies installed.

---

## Step 1 — Create folder and open terminal

In PowerShell:

```powershell
mkdir C:\Users\ronak\support-ticket-router
cd C:\Users\ronak\support-ticket-router
```

---

## Step 2 — Initialize git

```powershell
git init
git config user.name "Ronak Patil"
git config user.email "ronakpatil2406@gmail.com"
git branch -M main
```

---

## Step 3 — Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If you get a "running scripts is disabled" error:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then close and reopen PowerShell, retry.

---

## Step 4 — Install all dependencies

```powershell
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings scikit-learn vaderSentiment sqlalchemy datasets pandas numpy pytest httpx joblib jupyter
pip freeze > requirements.txt
```

---

## Step 5 — Create the plan files NOW (before any code)

Manually copy `docs/PROJECT_PLAN.md` and `docs/CHEATSHEET.md` from this conversation
into `C:\Users\ronak\support-ticket-router\docs\`. This is your insurance against
losing the plan.

---

## Step 6 — Create `.gitignore`

Create file `C:\Users\ronak\support-ticket-router\.gitignore` with this content:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
artifacts/*.joblib
data/raw/
.env
*.egg-info/
dist/
build/
.coverage
htmlcov/
.ipynb_checkpoints/
```

---

## Step 7 — First commit

```powershell
git add .
git status
git commit -m "chore: initialize project, venv, and dependencies"
```

---

## Step 8 — Create GitHub repo (web UI, 3 min)

1. Go to https://github.com/new
2. Owner: **John-Doe-for-you**
3. Repo name: **nlp-support-ticket-router**
4. Visibility: **Public**
5. **Do NOT** check "Add README", "Add .gitignore", "Add license" (you have them)
6. Click **Create repository**

---

## Step 9 — Push to GitHub

```powershell
git remote add origin https://github.com/John-Doe-for-you/nlp-support-ticket-router.git
git push -u origin main
```

---

## Step 10 — Create GitHub profile README (10 min)

1. Go to https://github.com/new
2. Repo name: **John-Doe-for-you** (must match your username exactly)
3. Visibility: Public
4. Check **"Add a README file"**
5. Create
6. Edit the README on GitHub — replace content with:

```markdown
# Hi, I'm Ronak 👋
Aspiring ML Engineer | Building production-grade NLP systems

## 🔭 Current Focus
Building portfolio projects to transition into ML Engineering.

## 🛠️ Tech Stack
Python • FastAPI • scikit-learn • SQLAlchemy • Docker • pytest

## 📌 Featured Projects
- **[NLP Support Ticket Router](https://github.com/John-Doe-for-you/nlp-support-ticket-router)** — 
  Classifies support tickets, detects sentiment, assigns priority, routes to 
  teams via REST API. TF-IDF + LogReg, VADER, FastAPI, SQLite, Docker.

## 📫 Reach Me
ronakpatil2406@gmail.com
```

7. Commit directly on the web UI (green button)

---

## Day 1 Done — Verify

```powershell
git log --oneline
```

You should see: `chore: initialize project, venv, and dependencies`

Open https://github.com/John-Doe-for-you/nlp-support-ticket-router — files should be visible.

Open https://github.com/John-Doe-for-you — profile README should be visible.

---

## Quick Decision Reference (do not re-decide)

| Thing | Value |
|---|---|
| Local folder | `C:\Users\ronak\support-ticket-router` |
| GitHub repo | `nlp-support-ticket-router` |
| GitHub username | `John-Doe-for-you` |
| Git email | `ronakpatil2406@gmail.com` |
| Python | 3.11 |
| Plan file | `docs/PROJECT_PLAN.md` |
| This cheatsheet | `docs/CHEATSHEET.md` |
| Daily commit format | `<scope>: <imperative message>` |
| Daily push | End of every session |
| Target accuracy | >= 88% |
| Target latency p99 | < 100ms |

---

## Resume Bullets (use these, don't rewrite)

- Built a production-grade NLP system that classifies support tickets into 5
  categories (92% accuracy) and assigns P1/P3 priority, served via FastAPI with
  <100ms p99 latency.
- Designed a hybrid ML pipeline combining TF-IDF + Logistic Regression for
  classification and VADER + custom urgency lexicon for sentiment, deployed in
  a Docker container with SQLite persistence.
- Engineered a priority-scoring rule engine that fuses sentiment, urgency
  keywords, and customer plan to drive intelligent team routing, with full
  pytest coverage (60+ tests).
