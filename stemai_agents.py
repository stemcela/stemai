"""
StemAI Agent System
====================
Three AI agents that work together to crack the stem cell code:

  PARSE    — fetches & summarizes stem cell literature from PubMed
  MAP      — builds mechanistic models from parsed findings
  VALIDATE — scores confidence and flags gaps in the evidence

Requirements:
    pip install crewai crewai-tools anthropic requests python-dotenv

Usage:
    1. Add your Anthropic key to a .env file:  ANTHROPIC_API_KEY=sk-ant-...
    2. Run:  python stemai_agents.py
    3. Optionally pass a topic:  python stemai_agents.py "exosomes neuroinflammation"
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import Field

load_dotenv()

# ─────────────────────────────────────────────
# LLM — Claude Sonnet via Anthropic
# ─────────────────────────────────────────────

claude = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
)

# ─────────────────────────────────────────────
# TOOL: PubMed Search
# ─────────────────────────────────────────────

class PubMedSearchTool(BaseTool):
    name: str = "PubMed Search"
    description: str = (
        "Searches PubMed for peer-reviewed literature. "
        "Input: a search query string. "
        "Returns: titles, abstracts, and PMIDs of the top results."
    )

    def _run(self, query: str) -> str:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        # Search for IDs
        search_url = f"{base}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": 5,
            "retmode": "json",
            "sort": "relevance",
        }
        try:
            r = requests.get(search_url, params=search_params, timeout=10)
            ids = r.json()["esearchresult"]["idlist"]
        except Exception as e:
            return f"PubMed search error: {e}"

        if not ids:
            return "No results found for that query."

        # Fetch abstracts
        fetch_url = f"{base}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "text",
        }
        try:
            r2 = requests.get(fetch_url, params=fetch_params, timeout=15)
            return r2.text[:6000]  # cap to avoid token overflow
        except Exception as e:
            return f"PubMed fetch error: {e}"


class ClinicalTrialsSearchTool(BaseTool):
    name: str = "ClinicalTrials Search"
    description: str = (
        "Searches ClinicalTrials.gov for registered trials. "
        "Input: a condition or intervention query string. "
        "Returns: trial titles, status, and brief summaries."
    )

    def _run(self, query: str) -> str:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": query,
            "pageSize": 5,
            "format": "json",
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            studies = data.get("studies", [])
            if not studies:
                return "No trials found."
            results = []
            for s in studies:
                proto = s.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                desc_mod = proto.get("descriptionModule", {})
                results.append({
                    "nct_id": id_mod.get("nctId"),
                    "title": id_mod.get("briefTitle"),
                    "status": status_mod.get("overallStatus"),
                    "summary": desc_mod.get("briefSummary", "")[:300],
                })
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"ClinicalTrials error: {e}"


# ─────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────

pubmed_tool = PubMedSearchTool()
trials_tool = ClinicalTrialsSearchTool()

parse_agent = Agent(
    role="PARSE — Stem Cell Literature Analyst",
    goal=(
        "Search PubMed and ClinicalTrials.gov for the latest evidence on stem cell "
        "and exosome therapies. Extract key findings: what was tested, what outcomes "
        "were observed, what mechanisms were proposed."
    ),
    backstory=(
     "You are a specialist in mesenchymal stem cell (MSC) biology, "
    "exosome and extracellular vesicle (EV) research, and regenerative medicine. "
    "You are expert in MISEV guidelines, secretome characterization, and EV isolation methods. "
    "You only analyze topics related to stem cells, exosomes, EVs, or regenerative medicine."
),
        # "You are a meticulous biomedical literature analyst trained on regenerative "
        # "medicine. You cut through hype and extract signal: what the data actually "
        # "shows about why stem cell therapies produce the effects they do."
    ),
    tools=[pubmed_tool, trials_tool],
    llm=claude,
    verbose=True,
    max_iter=2,
)

map_agent = Agent(
    role="MAP — Mechanistic Model Builder",
    goal=(
        "Take the findings from PARSE and construct a mechanistic model: "
        "identify the biological pathways, signaling molecules, and cellular "
        "processes that best explain the observed therapeutic effects."
    ),
    backstory=(
    "You are a systems biologist specializing in stem cell signaling: "
    "Wnt, Notch, TGF-β, paracrine and juxtacrine pathways, miRNA cargo mechanisms, "
    "and MSC immunomodulation. You build mechanistic models exclusively for "
    "stem cell and exosome therapeutic contexts."
        # "You are a systems biologist who thinks in pathways and networks. "
        # "Given a set of experimental findings, you build coherent mechanistic "
        # "hypotheses that connect molecular events to clinical outcomes. "
        # "You distinguish between correlation and mechanism."
    ),
    tools=[],
    llm=claude,
    verbose=True,
    max_iter=2,
)

validate_agent = Agent(
    role="VALIDATE — Evidence Confidence Scorer",
    goal=(
        "Review the mechanistic model from MAP and assign an evidence confidence "
        "score (0-100) for each proposed mechanism. Flag gaps, contradictions, "
        "and what data would be needed to increase confidence."
    ),
    backstory=(
    "You are a clinical research methodologist specializing in stem cell and EV trials. "
    "You evaluate evidence using ISEV standards, MISEV2018 guidelines, and regenerative "
    "medicine trial frameworks. You only score evidence for stem cell and exosome research."
        # "You are a clinical research methodologist and biostatistician. "
        # "You evaluate the strength of biological evidence rigorously: "
        # "sample sizes, replication, human vs animal data, confounders. "
        # "Your output is a structured confidence assessment that tells researchers "
        # "exactly where the evidence is strong and where it breaks down."
    ),
    tools=[],
    llm=claude,
    verbose=True,
    max_iter=2,
)

# ─────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────

def build_tasks(topic: str):

    parse_task = Task(
        description=(
        f"Search PubMed for 3 key papers on: '{topic}'. Extract the top 3 findings and proposed mechanisms in bullet points. Be brief."
            # f"Search PubMed and ClinicalTrials.gov for recent evidence on: '{topic}'. "
            # "Extract: (1) key experimental findings, (2) proposed mechanisms mentioned "
            # "by authors, (3) clinical outcomes observed, (4) cell types or EV subtypes involved. "
            # "Summarize in structured bullet points."
        ),
        expected_output=(
            "A structured summary with sections: Key Findings, Proposed Mechanisms, "
            "Clinical Outcomes, Cell/EV Types Involved. 300-500 words."
        ),
        agent=parse_agent,
    )

    map_task = Task(
        description=(
            f"In 200 words max, build a simple mechanistic model for '{topic}' based on the literature. Give one core hypothesis sentence."

            # "Using the literature summary from PARSE, construct a mechanistic model "
            # f"explaining how stem cell / exosome therapies produce their effects in: '{topic}'. "
            # "Identify: (1) primary signaling pathways activated, (2) downstream cellular "
            # "effects, (3) the most likely causal chain from therapy administration to "
            # "clinical outcome. Present as a numbered mechanistic hypothesis."
        ),
        expected_output=(
            "A mechanistic model with: Primary Pathways, Downstream Effects, "
            "Proposed Causal Chain. Include a one-sentence 'core mechanism hypothesis'."
        ),
        agent=map_agent,
        context=[parse_task],
    )

    validate_task = Task(
        description=(
            # "Review the mechanistic model from MAP. For each proposed mechanism, assign "
            # "an evidence confidence score from 0-100 based on: quality of studies, "
            # "human vs animal data, replication across labs, effect sizes. "
            # "Flag the top 3 evidence gaps — what experiments or data would most "
            # "increase confidence in the model?"
            f"Give an overall confidence score 0-100 for the model. List the top 3 evidence gaps in 2 sentences each. Total response under 300 words."
        ),
        expected_output=(
            "A validation report with: (1) Confidence Scores per mechanism (table format), "
            "(2) Overall Model Confidence Score 0-100, (3) Top 3 Evidence Gaps, "
            "(4) Recommended next experiments or data sources."
        ),
        agent=validate_agent,
        context=[parse_task, map_task],
    )

    return [parse_task, map_task, validate_task]


# ─────────────────────────────────────────────
# CREW & RUNNER
# ─────────────────────────────────────────────

def run_stemai(topic: str = "mesenchymal stem cell exosomes neuroinflammation aging"):
    print(f"\n{'='*60}")
    print(f"  StemAI Agent System")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    tasks = build_tasks(topic)

    crew = Crew(
        agents=[parse_agent, map_agent, validate_agent],
        tasks=tasks,
        process=Process.sequential,  # PARSE → MAP → VALIDATE
        verbose=True,
    )

    result = crew.kickoff()

    print(f"\n{'='*60}")
    print("  FINAL REPORT")
    print(f"{'='*60}\n")
    print(result)
    return result


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "mesenchymal stem cell exosomes neuroinflammation aging"
    )
    run_stemai(topic)
