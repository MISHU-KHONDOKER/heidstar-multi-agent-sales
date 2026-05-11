"""
qualifier.py
The Qualifier Agent — filters and scores leads from the Researcher.

Given a list of raw leads (output from Researcher Agent), this agent:
  1. Filters out false positives (market research firms, news articles, etc.)
  2. Matches each real lead to specific Heidstar products using the Product DB
  3. Scores each lead 0-100 based on multiple factors
  4. Returns a ranked list of qualified leads ready for proposal writing

This is stage 2 of the multi-agent sales pipeline.
"""

import json
from typing import Optional
from langchain_openai import ChatOpenAI

# Path setup so we can import from src/tools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.product_database import ProductDatabaseTool


class QualifierAgent:
    """
    Filters, scores, and ranks raw leads from the Researcher Agent.

    Usage:
        qualifier = QualifierAgent()
        qualified = qualifier.qualify_leads(raw_leads)
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen3:1.7b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            temperature=0.1,    # low — we want consistent scoring
            timeout=120,
        )
        self.product_db = ProductDatabaseTool()

    # ─────────────────────────────────────────────────────
    # Step 1: Hard filter — is this even a real organization?
    # ─────────────────────────────────────────────────────
    def _is_real_organization(self, lead: dict) -> bool:
        """
        Quick deterministic check before we use the LLM.
        Catches obvious false positives without burning LLM time.
        """
        name = (lead.get("organization_name") or "").lower()
        focus = (lead.get("focus_area") or "").lower()

        # Common red-flag substrings that suggest market research / media, not a customer
        false_positive_signals = [
            "market research",
            "research firm",
            "consulting",
            "consultancy",
            "advisory",
            "analyst report",
            "market report",
            "market size",
            "decibio",
            "omrglobal",
            "marketsandmarkets",
            "grand view research",
            "fortune business insights",
            "mordor intelligence",
        ]

        text = f"{name} {focus}"
        for signal in false_positive_signals:
            if signal in text:
                return False

        return True

    # ─────────────────────────────────────────────────────
    # Step 2: LLM-based qualification and scoring
    # ─────────────────────────────────────────────────────
    def _score_lead(self, lead: dict) -> Optional[dict]:
        """
        Use the LLM to evaluate one lead in depth.
        Returns the lead enriched with score, reasoning, and product matches.
        """
        # First, get product matches using the Product DB
        signals = lead.get("signals", [])
        matching_products = self.product_db.find_products_for_signals(signals)

        # Build a compact summary of matched products for the LLM
        if matching_products:
            products_summary = "\n".join([
                f"  - {p['id']} ({p['category']}): {p['description'][:100]}"
                for p in matching_products[:3]
            ])
        else:
            products_summary = "  (No direct product matches found by signal overlap.)"

        prompt = f"""You are a senior sales analyst at Heidstar Technology, a precision microscopy hardware manufacturer based in Xiamen, China. Your job is to evaluate sales leads.

LEAD INFORMATION:
- Organization: {lead.get('organization_name', 'unknown')}
- Type: {lead.get('organization_type', 'unknown')}
- Country: {lead.get('country', 'unknown')}
- Focus area: {lead.get('focus_area', 'unknown')}
- Signals: {', '.join(signals)}

HEIDSTAR PRODUCTS THAT MIGHT FIT THIS LEAD:
{products_summary}

Evaluate this lead against the following criteria and return a JSON object:

{{
  "is_qualified": true or false (true if this is a real organization that COULD buy microscopy hardware),
  "score": integer from 0 to 100,
  "best_product_match": "product ID like HDS-MSCAN-60F, or null if no good match",
  "reasoning": "2-3 sentence explanation of the score",
  "buying_signal_strength": "strong | moderate | weak | none"
}}

SCORING GUIDE:
- 80-100: Real lab/hospital/OEM with active need matching a Heidstar product
- 60-79: Real organization with possible fit but unclear need
- 40-59: Real org but weak product fit
- 20-39: Possibly real but signal is unclear
- 0-19: Likely not a real customer (article, market report, etc.)

Return ONLY the JSON, no markdown, no explanation outside the JSON.
"""

        response = self.llm.invoke(prompt)
        text = response.content.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            evaluation = json.loads(text)

            # Build enriched lead
            qualified = {
                **lead,  # keep all original fields
                "is_qualified": evaluation.get("is_qualified", False),
                "score": int(evaluation.get("score", 0)),
                "best_product_match": evaluation.get("best_product_match"),
                "reasoning": evaluation.get("reasoning", ""),
                "buying_signal_strength": evaluation.get("buying_signal_strength", "none"),
                "matching_products": [p["id"] for p in matching_products[:3]],
            }
            return qualified

        except (json.JSONDecodeError, ValueError):
            print(f"   ⚠️  Could not parse evaluation for {lead.get('organization_name')}")
            return None

    # ─────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────
    def qualify_leads(
        self,
        raw_leads: list[dict],
        min_score: int = 50,
    ) -> list[dict]:
        """
        Qualify and rank a list of raw leads.

        Args:
            raw_leads: List of lead dicts from the Researcher Agent.
            min_score: Minimum score to keep in the final output (default 50).

        Returns:
            List of qualified leads, sorted highest score first.
        """
        print(f"\n🎯 Qualifier Agent starting")
        print(f"   Input: {len(raw_leads)} raw leads")
        print(f"   Minimum score threshold: {min_score}\n")

        if not raw_leads:
            print("   No leads to qualify.")
            return []

        qualified = []

        for i, lead in enumerate(raw_leads, 1):
            name = lead.get("organization_name", "unknown")
            print(f"   [{i}/{len(raw_leads)}] Evaluating: {name}")

            # Step 1: hard filter (free, fast)
            if not self._is_real_organization(lead):
                print(f"       ⏩ Filtered (false positive heuristic)")
                continue

            # Step 2: LLM scoring (slower, uses Ollama)
            evaluated = self._score_lead(lead)

            if evaluated is None:
                print(f"       ⚠️  Could not evaluate — skipping")
                continue

            score = evaluated["score"]
            print(f"       Score: {score}/100   "
                  f"Match: {evaluated.get('best_product_match', 'none')}   "
                  f"Signal: {evaluated.get('buying_signal_strength', 'none')}")

            if score >= min_score and evaluated["is_qualified"]:
                qualified.append(evaluated)
                print(f"       ✅ Qualified")
            else:
                print(f"       ❌ Below threshold")

        # Sort by score, highest first
        qualified.sort(key=lambda L: L["score"], reverse=True)

        print(f"\n✅ Qualifier finished. {len(qualified)} qualified leads "
              f"out of {len(raw_leads)} input.\n")
        return qualified


# ─────────────────────────────────────────────────────
# Quick self-test using the Researcher's last output style
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qualifier Agent")
    print("=" * 60)

    # Realistic input matching what the Researcher produces
    sample_leads = [
        {
            "is_lead": True,
            "organization_name": "Charité — Universitätsmedizin Berlin",
            "organization_type": "hospital",
            "country": "Germany",
            "focus_area": "Major German university hospital with a digital pathology department running thousands of slides per week",
            "signals": [
                "hospital pathology department",
                "digital pathology adoption",
                "high slide volume",
                "cancer diagnostic services",
            ],
            "source_url": "https://example.com/charite-pathology",
        },
        {
            "is_lead": True,
            "organization_name": "Omrglobal",
            "organization_type": "research_lab",
            "country": "Germany",
            "focus_area": "Digital Pathology, Telepathology, market research analysis",
            "signals": [
                "Digital Pathology",
                "Market Research",
                "industry reports",
            ],
            "source_url": "https://www.omrglobal.com/blogs/germany-digital-pathology-market",
        },
        {
            "is_lead": True,
            "organization_name": "MIT Koch Institute Drug Screening Core",
            "organization_type": "research_lab",
            "country": "USA",
            "focus_area": "High-content fluorescence drug screening for cancer research, runs FISH assays and immunofluorescence",
            "signals": [
                "FISH testing services",
                "high-content fluorescence imaging",
                "drug screening operations",
                "cancer genetics research",
            ],
            "source_url": "https://example.com/mit-koch-drug-screening",
        },
    ]

    qualifier = QualifierAgent()
    qualified = qualifier.qualify_leads(sample_leads, min_score=50)

    print("=" * 60)
    print("FINAL QUALIFIED LEADS (sorted by score)")
    print("=" * 60)
    for i, lead in enumerate(qualified, 1):
        print(f"\n{i}. {lead['organization_name']}  ({lead['country']})")
        print(f"   Score: {lead['score']}/100")
        print(f"   Best fit: {lead.get('best_product_match', 'n/a')}")
        print(f"   Signal strength: {lead.get('buying_signal_strength', 'n/a')}")
        print(f"   Reasoning: {lead.get('reasoning', '')}")