"""
StemAI Agent System - Stem Cell & Exosome Research
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

load_dotenv()

claude = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2048,
)

class PubMedSearchTool(BaseTool):
    name: str = "PubMed Search"
    description: str = "Searches PubMed for peer-reviewed literature. Input: a search query string."

    def _run(self, query: str) -> str:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_params = {"db": "pubmed", "term": query, "retmax": 3, "retmode": "json", "sort": "relevance"}
        try:
            r = requests.get(f"{base}/esearch.fcgi", params=search_params, timeout=10)
            ids = r.json()["esearchresult"]["idlist"]
        except Exception as e:
            return f"PubMed search error: {e}"
        if not ids:
            return "No results found."
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}
        try:
            r2 = requests.get(f"{base}/efetch.fcgi", params=fetch_params, timeout=15)
            return r2.text[:3000]
        except Exception as e:
            return f"PubMed fetch error: {e}"


class ClinicalTrialsSearchTool(BaseTool):
    name: str = "ClinicalTrials Search"
    description: str = "Searches ClinicalTrials.gov for registered trials."

    def _run(self, query: str) -> str:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {"query.term": query, "pageSize": 3, "format": "json"}
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
                results.append({
                    "nct_id": id_mod.get("nctId"),
                    "title": id_mod.get("briefTitle"),
                    "status": status_mod.get("overallStatus"),
                })
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"ClinicalTrials error: {e}"


pubmed_tool = PubMedSearchTool()
trials_tool = ClinicalTrialsSearchTool()

parse_agent = Agent(
    role="PARSE — Stem Cell Literature Analyst",
    goal="Search PubMed for stem cell and exosome evidence. Extract top 3 findings in bullet points.",
    backstory=(
        "You are a specialist in mesenchymal stem cell biology, exosome research, "
        "and regenerative medicine. Expert in MISEV guidelines and secretome characterization."
    ),
    tools=[pubmed_tool, trials_tool],
    llm=claude,
    verbose=False,
    max_iter=2,
)

map_agent = Agent(
    role="MAP — Mechanistic Model Builder",
    goal="Build a brief mechanistic model from the literature. One core hypothesis, under 100 words.",
    backstory=(
        "You are a systems biologist specializing in stem cell signaling: "
        "Wnt, Notch, TGF-beta, paracrine pathways, and MSC immunomodulation."
    ),
    tools=[],
    llm=claude,
    verbose=False,
    max_iter=2,
)

validate_agent = Agent(
    role="VALIDATE — Evidence Confidence Scorer",
    goal="Score the model confidence 0-100. List 3 evidence gaps. Under 150 words total.",
    backstory=(
        "You are a clinical research methodologist specializing in stem cell and EV trials. "
        "You evaluate evidence using ISEV standards and MISEV2018 guidelines."
    ),
    tools=[],
    llm=claude,
    verbose=False,
    max_iter=2,
)


def build_tasks(topic: str):
    parse_task = Task(
        description=f"Search PubMed for '{topic}'. Return exactly 3 bullet points: one key finding, one proposed mechanism, one clinical outcome. Under 100 words total.",
        expected_output="3 bullet points under 100 words total.",
        agent=parse_agent,
    )
    map_task = Task(
        description=f"In under 100 words, state the single most likely mechanism for '{topic}'. One paragraph only.",
        expected_output="One paragraph under 100 words with a core mechanism hypothesis.",
        agent=map_agent,
        context=[parse_task],
    )
    validate_task = Task(
        description=f"Give an overall confidence score 0-100 for the model. List 3 evidence gaps in one sentence each. Total under 150 words.",
        expected_output="Confidence score and 3 evidence gaps under 150 words.",
        agent=validate_agent,
        context=[parse_task, map_task],
    )
    return [parse_task, map_task, validate_task]


def run_stemai(topic: str = "mesenchymal stem cell exosomes neuroinflammation aging"):
    print(f"\n{'='*60}\n  StemAI Agent System\n  Topic: {topic}\n{'='*60}\n")
    tasks = build_tasks(topic)
    crew = Crew(
        agents=[parse_agent, map_agent, validate_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    print(f"\n{'='*60}\n  FINAL REPORT\n{'='*60}\n")
    print(result)
    return result


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "mesenchymal stem cell exosomes neuroinflammation aging"
    run_stemai(topic)
