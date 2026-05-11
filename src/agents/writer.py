"""
writer.py
The Writer Agent — drafts tailored sales proposals.

Two modes:
  - PRODUCT PITCH: a matched product exists. We pitch it with real specs.
  - SOFT INTRODUCTION: no product matched. We send a relationship-opening
    email — NO product names, NO specs, NO inventions.

Two layers of hallucination protection:
  1. PROMPT-LEVEL: soft intro never mentions products; product pitch is
     constrained to a real product's real specs.
  2. POST-GENERATION VALIDATION: after the LLM writes, we scan the text
     for any product name that is not in our real catalog. If we find
     a fake name, we strip the line containing it.

This is stage 3 of the multi-agent sales pipeline.
"""

import re
from langchain_openai import ChatOpenAI

# Path setup so we can import from src/tools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.product_database import ProductDatabaseTool


class WriterAgent:
    """
    Generates tailored sales proposals for qualified leads.

    Usage:
        writer = WriterAgent()
        proposal = writer.write_proposal(qualified_lead)
        print(proposal["markdown"])
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen3:1.7b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            temperature=0.4,
            timeout=180,
        )
        self.product_db = ProductDatabaseTool()

        # Cache the real product IDs once — used for hallucination scanning
        self._real_product_ids = {
            p["id"] for p in self.product_db.get_all_products()
        }

    # ─────────────────────────────────────────────────────
    # POST-GENERATION VALIDATOR
    # ─────────────────────────────────────────────────────
    def _strip_fake_products(self, text: str) -> str:
        """
        Scan the generated proposal for any 'Heidstar <something>' or
        'HDS-<something>' that is NOT in our real catalog. If a fake name
        is found, remove the entire line containing it.

        This is our last line of defense against hallucinated products.
        """
        # Pattern catches things like "Heidstar 3000 Series", "HDS-XYZ", etc.
        # We are looking for product-like phrases that follow brand keywords.
        suspicious_pattern = re.compile(
            r"\b(Heidstar\s+[A-Z0-9][\w\-]+(?:\s+Series)?|HDS-[A-Z0-9][\w\-]*)",
            re.IGNORECASE,
        )

        kept_lines = []
        removed_count = 0

        for line in text.splitlines():
            matches = suspicious_pattern.findall(line)
            line_has_fake = False

            for match in matches:
                # Normalize the match — uppercase and strip trailing words
                # Compare against real product IDs (HDS-MSCAN-60F, etc.)
                # OR allow "Heidstar Technology" (company name, not a product)
                normalized = match.strip().upper()

                # Allow legitimate company references
                if normalized.startswith("HEIDSTAR TECHNOLOGY"):
                    continue
                if normalized in ("HEIDSTAR", "HEIDSTAR TECHNOLOGY"):
                    continue

                # Check if it matches any real product ID (allowing partial match)
                is_real = any(
                    real_id.upper() in normalized or normalized in real_id.upper()
                    for real_id in self._real_product_ids
                )

                if not is_real:
                    line_has_fake = True
                    removed_count += 1
                    print(f"   ⚠️  Removed line with fake product: '{match.strip()}'")
                    break

            if not line_has_fake:
                kept_lines.append(line)

        if removed_count > 0:
            print(f"   🛡️  Hallucination guard: stripped {removed_count} fake mention(s)")

        return "\n".join(kept_lines)

    # ─────────────────────────────────────────────────────
    # MODE 1 — Product pitch
    # ─────────────────────────────────────────────────────
    def _draft_product_pitch(self, lead: dict, product: dict, company: dict) -> str:
        """Draft a proposal pitching a specific Heidstar product with real specs."""
        credentials = "\n".join([f"  - {c}" for c in company["credentials"]])

        spec_lines = []
        for key, value in product["specs"].items():
            spec_lines.append(f"  - {key}: {value}")
        spec_block = "\n".join(spec_lines)

        prompt = f"""You are a senior sales engineer at Heidstar Technology writing a tailored business proposal email.

STRICT RULES:
1. The ONLY Heidstar product you may mention in this email is: {product['id']}
2. Use ONLY the specifications listed below. Do not invent resolutions, lens magnifications, or features.
3. Do not invent customer logos, prices, or testimonials.
4. Sign off as "The Heidstar Technology Team" — no other contact info.
5. Do NOT add a postscript, footnote, or note explaining the email.

THE PROSPECT:
- Organization: {lead.get('organization_name', 'Unknown')}
- Country: {lead.get('country', 'unknown')}
- Type: {lead.get('organization_type', 'unknown')}
- Focus: {lead.get('focus_area', '')}
- Why they fit: {lead.get('reasoning', '')}

THE PRODUCT ({product['id']}):
DESCRIPTION: {product['description']}
REAL SPECS:
{spec_block}
TARGET CUSTOMERS: {', '.join(product['target_customers'][:3])}

HEIDSTAR CREDENTIALS (use 1-2 of these):
{credentials}

WRITING RULES:
- Professional warm English, 250-350 words
- Markdown format (bold, bullets allowed)
- Start with "Subject: <subject line>" on first line, blank line, then body
- Reference specific facts about the prospect's work
- Reference the product's REAL specs only
- End with a 30-minute call invitation
- Sign off as "The Heidstar Technology Team"

Write the complete email now.
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ─────────────────────────────────────────────────────
    # MODE 2 — Soft introduction (NO product mentions at all)
    # ─────────────────────────────────────────────────────
    def _draft_soft_introduction(self, lead: dict, company: dict) -> str:
        """
        Pure relationship-opening email. No products. No specs. No model numbers.
        Just: 'we saw your work, we make precision imaging, can we talk?'
        """
        credentials = "\n".join([f"  - {c}" for c in company["credentials"]])

        prompt = f"""You are a business development representative at Heidstar Technology, writing a SHORT introduction email to a potential research prospect.

CRITICAL RULES — VIOLATING ANY OF THESE RUINS THE EMAIL:
1. Do NOT mention any specific product name or model number anywhere.
2. Do NOT mention any specific technical specifications (no resolutions, no slide counts, no scan speeds, no lens magnifications).
3. Do NOT promise specific capabilities or features.
4. Do NOT use the words "Series", "HDS-", or any model-like terminology.
5. The email is purely to OPEN A CONVERSATION — not to pitch anything.

ABOUT HEIDSTAR (what you CAN say in general terms):
- We are a precision microscopy and imaging hardware company.
- We are based in Xiamen, China.
- We make digital pathology scanners, fluorescence imaging systems, and precision optical components.
- We are a Zeiss Class A Global Supplier with aerospace-grade certification.

THE PROSPECT:
- Organization: {lead.get('organization_name', 'Unknown')}
- Country: {lead.get('country', 'unknown')}
- Type: {lead.get('organization_type', 'unknown')}
- Focus area: {lead.get('focus_area', '')}

EMAIL STRUCTURE (180-250 words total):
1. Subject line on first line as "Subject: <line>", then blank line.
2. Greeting (use "Dear Research Team," — do not invent a personal name).
3. ONE sentence acknowledging the prospect's research focus.
4. ONE sentence introducing Heidstar in general terms (no products).
5. TWO open questions about their imaging or instrumentation needs.
6. Invitation to a brief 20-minute discovery call.
7. Sign off as "The Heidstar Technology Team".

Do NOT add bullet lists of features. Do NOT add a postscript. Just the email body.

Write the complete email now.
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ─────────────────────────────────────────────────────
    # Public method
    # ─────────────────────────────────────────────────────
    def write_proposal(self, qualified_lead: dict) -> dict:
        """
        Draft a proposal for one qualified lead.
        Picks mode automatically based on whether a product was matched.
        Applies post-generation hallucination scanning before returning.
        """
        org_name = qualified_lead.get("organization_name", "Unknown")
        product_id = qualified_lead.get("best_product_match")
        company = self.product_db.get_company_info()

        product = (
            self.product_db.get_product_by_id(product_id)
            if product_id else None
        )

        print(f"\n📝 Writer Agent drafting proposal")
        print(f"   For: {org_name}")

        if product is not None:
            print(f"   Mode: PRODUCT PITCH ({product['id']})")
            raw_markdown = self._draft_product_pitch(qualified_lead, product, company)
            mode = "product_pitch"
            pitched = product["id"]
        else:
            print(f"   Mode: SOFT INTRODUCTION (no product matched)")
            raw_markdown = self._draft_soft_introduction(qualified_lead, company)
            mode = "soft_introduction"
            pitched = None

        # Apply hallucination guard
        clean_markdown = self._strip_fake_products(raw_markdown)

        print(f"   ✅ Drafted ({len(clean_markdown)} characters)\n")

        return {
            "organization_name": org_name,
            "product_pitched": pitched,
            "mode": mode,
            "markdown": clean_markdown,
        }


# ─────────────────────────────────────────────────────
# Self-test — exercises both modes
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Writer Agent — Both Modes + Hallucination Guard")
    print("=" * 60)

    writer = WriterAgent()

    # TEST 1: Product pitch
    print("\n" + "▶" * 60)
    print("▶ TEST 1: PRODUCT PITCH MODE (matched HDS-MSCAN-60F)")
    print("▶" * 60)
    pitch_lead = {
        "organization_name": "European Bank for Induced Pluripotent Stem Cells",
        "organization_type": "research_lab",
        "country": "Europe",
        "focus_area": "iPSC banking for European research community",
        "is_qualified": True,
        "score": 85,
        "best_product_match": "HDS-MSCAN-60F",
        "reasoning": "Active iPSC banking aligns with high-content fluorescence imaging needs.",
    }
    r1 = writer.write_proposal(pitch_lead)
    print("─" * 60)
    print(r1["markdown"])
    print()

    # TEST 2: Soft intro
    print("\n" + "▶" * 60)
    print("▶ TEST 2: SOFT INTRODUCTION MODE (no product matched)")
    print("▶" * 60)
    intro_lead = {
        "organization_name": "CMMC: iPSC-Lab",
        "organization_type": "university_lab",
        "country": "Germany",
        "focus_area": "Stem cell innovation including iPSCs and organoid research",
        "is_qualified": True,
        "score": 65,
        "best_product_match": None,
        "reasoning": "Real lab with iPSC focus, but specific instrumentation needs unclear.",
    }
    r2 = writer.write_proposal(intro_lead)
    print("─" * 60)
    print(r2["markdown"])