"""
web_search.py
A clean wrapper around the Tavily web search API.

This tool is used by agents to discover information from the live internet —
finding companies, research labs, hospitals, technical news, etc.

By wrapping Tavily here, we hide the API details from the agents.
If we ever change search providers, only this file changes.
"""

import os
import time
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables once at module import
load_dotenv()


class WebSearchTool:
    """
    Wraps the Tavily search API in a clean, agent-friendly interface.
    Includes automatic retry on transient network errors.

    Usage:
        tool = WebSearchTool()
        results = tool.search("pharma companies Boston cancer drug discovery")
    """

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in .env file")
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        max_retries: int = 3,
    ) -> list[dict]:
        """
        Search the web for the given query.

        Args:
            query: The natural-language search query.
            max_results: How many top results to return (default 5).
            max_retries: How many times to retry on transient errors (default 3).

        Returns:
            A list of result dicts, each with keys:
              - 'title': page title
              - 'url': page URL
              - 'content': brief snippet of the page content
              - 'score': relevance score (0-1, higher is more relevant)
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.search(
                    query=query,
                    max_results=max_results,
                )
                return response.get("results", [])

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_seconds = attempt * 2  # 2s, 4s, 6s backoff
                    print(
                        f"⚠️  Search attempt {attempt} failed ({type(e).__name__}). "
                        f"Retrying in {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)
                else:
                    print(f"❌ All {max_retries} attempts failed. Last error: {e}")

        return []


# ─────────────────────────────────────────────────────
# Quick self-test — run this file directly to test it
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing WebSearchTool...")
    tool = WebSearchTool()
    results = tool.search(
        query="hospital pathology departments digital slide scanner Germany",
        max_results=3,
    )
    print(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"   {r['url']}")
        print(f"   Score: {r['score']:.2f}")
        print(f"   Snippet: {r['content'][:150]}...")
        print()