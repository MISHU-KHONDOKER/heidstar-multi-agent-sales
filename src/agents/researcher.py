"""
researcher.py
The Researcher Agent — finds qualified leads for Heidstar products.

Given a search brief (e.g., "stem cell research labs in Europe"),
this agent:
  1. Generates multiple targeted search queries using the LLM
  2. Executes each search via the Web Search tool
  3. Deduplicates results across all searches
  4. Asks the LLM to extract structured lead data from raw results
  5. Returns a clean list of candidate leads ready for qualification

This is the first stage of the multi-agent sales pipeline.
"""

import json
from typing import Optional
from langchain_openai import ChatOpenAI

# Import our custom tools
# We use relative imports because we are inside the 'src' package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.web_search import WebSearchTool


class ResearcherAgent:
    """
    Finds qualified Heidstar leads from the live internet.

    Usage:
        researcher = ResearcherAgent()
        leads = researcher.find_leads(
            brief="Pharma companies in Boston doing high-content drug screening",
            target_count=5,
        )
    """

    def __init__(self):
        # Connect to local Ollama (Qwen 3 1.7B running on your laptop)
        self.llm = ChatOpenAI(
            model="qwen3:1.7b",
            api_key="ollama",  # placeholder — Ollama does not check this
            base_url="http://localhost:11434/v1",
            temperature=0.3,   # a little creativity for query variation
            timeout=120,
        )
        self.search_tool = WebSearchTool()

    # ─────────────────────────────────────────────────────
    # Step 1: Generate diverse search queries
    # ─────────────────────────────────────────────────────
    def _generate_search_queries(self, brief: str, num_queries: int = 3) -> list[str]:
        """
        Ask the LLM to generate diverse search queries from one brief.

        Why diverse queries? A single query like "stem cell labs Germany"
        will return similar results. Multiple angles surface more leads.
        """
        prompt = f"""You are a sales researcher for Heidstar Technology, a Chinese precision microscopy hardware manufacturer.

The sales team has given you this lead-finding brief:
"{brief}"

Generate exactly {num_queries} different web search queries that would find real companies, hospitals, or research labs matching this brief. Each query should approach the target from a different angle — different keywords, different geography, different specializations.

Return ONLY a JSON array of {num_queries} strings. No explanation, no markdown, just the JSON array.

Example format:
["query 1", "query 2", "query 3"]
"""

        response = self.llm.invoke(prompt)
        text = response.content.strip()

        # Sometimes the LLM wraps JSON in markdown code blocks. Strip those.
        if text.startswith("```"):
            # Remove first line (```json or ```) and last line (```)
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries
        except json.JSONDecodeError:
            pass

        # Fallback: if LLM output is malformed, use the brief itself as a single query
        print(f"⚠️  Could not parse query list, falling back to single query.")
        return [brief]

    # ─────────────────────────────────────────────────────
    # Step 2: Execute searches and collect raw results
    # ─────────────────────────────────────────────────────
    def _gather_search_results(
        self,
        queries: list[str],
        results_per_query: int = 4,
    ) -> list[dict]:
        """
        Run each query through the Web Search tool, collect all results,
        deduplicate by URL.
        """
        all_results = []
        seen_urls = set()

        for query in queries:
            print(f"   🔎 Searching: {query}")
            results = self.search_tool.search(
                query=query,
                max_results=results_per_query,
            )
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

        return all_results

    # ─────────────────────────────────────────────────────
    # Step 3: Extract structured lead data using the LLM
    # ─────────────────────────────────────────────────────
    def _extract_lead_info(self, raw_result: dict) -> Optional[dict]:
        """
        Given a raw search result, ask the LLM to extract structured lead info.
        Returns a clean lead dict, or None if the result is not a real lead.
        """
        prompt = f"""You are analyzing a web search result to determine if it represents a real organization (company, hospital, university lab, or research institute) that could be a sales lead for Heidstar microscopy equipment.

SEARCH RESULT:
Title: {raw_result.get('title', '')}
URL: {raw_result.get('url', '')}
Content: {raw_result.get('content', '')[:500]}

Extract the following info as a JSON object:
{{
  "is_lead": true or false (true only if this is a real organization, NOT a news article or product review),
  "organization_name": "name of the company/lab/hospital",
  "organization_type": "one of: pharma_company, hospital, research_lab, university, oem_manufacturer, biotech_startup, other",
  "country": "country if identifiable, else 'unknown'",
  "focus_area": "1-sentence description of what they do",
  "signals": ["keyword phrase 1", "keyword phrase 2", "..."]
}}

The "signals" field should contain 2-5 short phrases describing the organization's areas of work — these will be used to match against Heidstar products.

Return ONLY the JSON object, no markdown, no explanation.
"""

        response = self.llm.invoke(prompt)
        text = response.content.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            lead = json.loads(text)
            if lead.get("is_lead"):
                # Add the source URL for traceability
                lead["source_url"] = raw_result.get("url", "")
                return lead
        except json.JSONDecodeError:
            pass

        return None

    # ─────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────
    def find_leads(
        self,
        brief: str,
        target_count: int = 5,
    ) -> list[dict]:
        """
        Find qualified leads matching the given brief.

        Args:
            brief: Natural-language description of the target.
                   e.g., "Stem cell research labs in Europe"
            target_count: Stop trying once we have this many real leads.

        Returns:
            A list of structured lead dicts.
        """
        print(f"\n🎯 Researcher Agent starting")
        print(f"   Brief: {brief}")
        print(f"   Target: {target_count} qualified leads\n")

        # Step 1: Generate diverse queries
        print("📝 Step 1: Generating search queries...")
        queries = self._generate_search_queries(brief, num_queries=3)
        for q in queries:
            print(f"   • {q}")
        print()

        # Step 2: Run searches
        print("🌐 Step 2: Executing web searches...")
        raw_results = self._gather_search_results(
            queries,
            results_per_query=4,
        )
        print(f"   Got {len(raw_results)} unique results\n")

        # Step 3: Extract structured lead info
        print("🔬 Step 3: Extracting lead data from each result...")
        leads = []
        for i, raw in enumerate(raw_results, 1):
            print(f"   [{i}/{len(raw_results)}] Analyzing: {raw.get('title', '')[:60]}...")
            lead = self._extract_lead_info(raw)
            if lead:
                leads.append(lead)
                print(f"       ✅ Lead: {lead.get('organization_name', 'unknown')}")
            else:
                print(f"       ⏩ Skipped (not a real organization)")

            if len(leads) >= target_count:
                print(f"\n   Reached target of {target_count} leads, stopping.")
                break

        print(f"\n✅ Researcher finished. Found {len(leads)} leads.\n")
        return leads


# ─────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Researcher Agent")
    print("=" * 60)

    researcher = ResearcherAgent()
    leads = researcher.find_leads(
        brief="Hospital pathology departments in Germany using digital slide scanners",
        target_count=3,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for i, lead in enumerate(leads, 1):
        print(f"\n{i}. {lead.get('organization_name', 'Unknown')}")
        print(f"   Type: {lead.get('organization_type', 'unknown')}")
        print(f"   Country: {lead.get('country', 'unknown')}")
        print(f"   Focus: {lead.get('focus_area', 'unknown')}")
        print(f"   Signals: {', '.join(lead.get('signals', []))}")
        print(f"   Source: {lead.get('source_url', '')}")