# Public Figure Classifier — Flask + Claude Vision

Classifies photos of 11 public figures using the Claude Vision API as the backend.

## Local run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python server.py
# open http://localhost:5000
```

## Deploy to Render (free)

1. Push this folder to a GitHub repo.
2. Go to https://render.com → **New Web Service** → connect the repo.
3. Set **Start Command**: `gunicorn server:app`
4. Add environment variable: `ANTHROPIC_API_KEY` = your key.
5. Deploy — live URL in ~2 minutes.

## Deploy to Railway (free)

1. Push to GitHub.
2. Go to https://railway.app → **New Project** → **Deploy from GitHub repo**.
3. Add variable: `ANTHROPIC_API_KEY` = your key.
4. Railway auto-detects the Procfile. Done.

## Project structure

```
server.py          ← Flask backend calling Claude Vision API
app.html           ← Original frontend (fixed & updated)
app.css            ← Original styles
images/            ← Celebrity portrait photos
requirements.txt
Procfile           ← For Render/Railway
```
