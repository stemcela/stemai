"""
StemAI API Server
==================
Wraps the agent system in a FastAPI endpoint so your
Lovable frontend can trigger agents and stream results.

Requirements:
    pip install fastapi uvicorn crewai crewai-tools openai requests python-dotenv

Run locally:
    uvicorn stemai_api:app --reload --port 8000

Deploy to Railway:
    1. Push this folder to a GitHub repo
    2. Connect repo to Railway
    3. Set env var: ANTHROPIC_API_KEY
    4. Railway auto-detects uvicorn and deploys

Endpoints:
    POST /analyze        — run the full PARSE → MAP → VALIDATE pipeline
    GET  /health         — health check
    GET  /topics         — preset topic suggestions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
from stemai_agents import run_stemai

app = FastAPI(
    title="StemAI Agent API",
    description="AI agents cracking the stem cell code",
    version="1.0.0",
)

# Allow your Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
        # "https://stem-code-decoded.lovable.app",
        # "http://localhost:3000",   # local dev
        # "http://localhost:5173",   # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    topic: Optional[str] = "mesenchymal stem cell exosomes neuroinflammation aging"

class AnalyzeResponse(BaseModel):
    topic: str
    status: str
    report: str

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "online", "system": "StemAI Agent Network"}


@app.get("/topics")
def topics():
    """Preset research topics for the frontend dropdown."""
    return {
        "topics": [
            "mesenchymal stem cell exosomes neuroinflammation aging",
            "MSC exosomes skin aging wound healing",
            "stem cell secretome systemic aging senescence",
            "extracellular vesicles cognitive decline Alzheimer's",
            "exosome therapy mechanism of action clinical trial",
        ]
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Trigger the full agent pipeline:
      PARSE → MAP → VALIDATE
    Returns the final validation report.

    Note: This takes 60-120 seconds. For production,
    consider a background task + polling pattern (see below).
    """
    STEM_KEYWORDS = [
    "stem cell", "exosome", "extracellular vesicle", "EV", "MSC",
    "mesenchymal", "regenerative", "secretome", "iPSC", "progenitor",
    "cell therapy", "exosomal", "stromal", "hematopoietic", "neural stem",
    "tissue regeneration", "paracrine", "senescence", "inflammaging",
    "aging", "longevity", "neurodegeneration", "SASP", "mitochondria"
    ]   
    topic = request.topic.strip()
    topic_lower = topic.lower()
    if not any(kw.lower() in topic_lower for kw in STEM_KEYWORDS):
    raise HTTPException(
        status_code=400,
        detail="StemAI is focused on stem cell, exosome, and regenerative medicine research. Please enter a topic related to these fields."
    )
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    try:
        # Run the crew in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_stemai, topic)

        return AnalyzeResponse(
            topic=topic,
            status="complete",
            report=str(result),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ─────────────────────────────────────────────
# Background job pattern (for production)
# ─────────────────────────────────────────────
#
# For a better UX, use this pattern instead of blocking:
#
#   POST /analyze/start   → returns { job_id }
#   GET  /analyze/{job_id} → returns { status, report }
#
# Implement with:
#   - Redis + Celery (robust)
#   - Supabase edge functions + DB polling (simpler)
#   - Railway background workers
#
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stemai_api:app", host="0.0.0.0", port=8000, reload=True)
