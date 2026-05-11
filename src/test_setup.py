"""
test_setup.py
A simple test to verify our environment is correctly configured.
This script:
  1. Loads API keys from .env file
  2. Tests connection to Ollama (local LLM running on your laptop)
  3. Tests connection to Tavily Search API
If both succeed, our foundation is ready.

NOTE: Ollama runs locally — no internet, no API key, no cost.
You only need internet for Tavily web search.
"""

import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────
# Step 1: Load environment variables from .env file
# ─────────────────────────────────────────────────────
load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY")

print("=" * 60)
print("Heidstar Multi-Agent System — Setup Verification")
print("=" * 60)

if not tavily_key:
    print("❌ TAVILY_API_KEY not found. Check your .env file.")
    exit(1)

print("✅ Tavily API key loaded")
print(f"   Key starts with: {tavily_key[:10]}...")
print()

# ─────────────────────────────────────────────────────
# Step 2: Test Ollama (local Qwen 3 1.7B model)
# ─────────────────────────────────────────────────────
print("Testing Ollama (local Qwen 3 1.7B)...")
try:
    from langchain_openai import ChatOpenAI

    # Ollama exposes an OpenAI-compatible API on localhost:11434
    # No real API key needed — we pass any string.
    llm = ChatOpenAI(
        model="qwen3:1.7b",
        api_key="ollama",  # placeholder — Ollama does not check this
        base_url="http://localhost:11434/v1",
        temperature=0,
        timeout=60,
    )

    response = llm.invoke("Say hello in exactly 5 words.")
    print(f"✅ Ollama works! Response: {response.content}")
except Exception as e:
    print(f"❌ Ollama failed: {e}")
    print("   Make sure Ollama is running. Try: ollama list")
    exit(1)
print()

# ─────────────────────────────────────────────────────
# Step 3: Test Tavily Search API
# ─────────────────────────────────────────────────────
print("Testing Tavily Search API connection...")
try:
    from tavily import TavilyClient

    tavily = TavilyClient(api_key=tavily_key)
    result = tavily.search(
        query="Heidstar Technology Xiamen microscope",
        max_results=2,
    )
    print(f"✅ Tavily works! Found {len(result['results'])} search results")
    print(f"   First result title: {result['results'][0]['title']}")
except Exception as e:
    print(f"❌ Tavily failed: {e}")
    exit(1)
print()

print("=" * 60)
print("🎉 All systems operational. Foundation is ready.")
print("=" * 60)