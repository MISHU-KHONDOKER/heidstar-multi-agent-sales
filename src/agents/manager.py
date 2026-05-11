"""
manager.py
The Manager Agent — orchestrates the full multi-agent sales pipeline.

Built with LangGraph. Defines a state graph where:
  - Each NODE is one of our agents (Researcher, Qualifier, Writer)
  - Each EDGE controls the flow between them
  - All agents share a single state dict (PipelineState)

The graph flow:
  START -> Researcher -> Qualifier -> Writer -> END

With conditional short-circuits: if any stage produces no output,
the graph ends gracefully instead of running empty stages.
"""

import sys
from pathlib import Path

# Path setup so we can import siblings
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, START, END

from state import PipelineState
from agents.researcher import ResearcherAgent
from agents.qualifier import QualifierAgent
from agents.writer import WriterAgent


# ─────────────────────────────────────────────────────
# NODE FUNCTIONS — each one calls one agent
# ─────────────────────────────────────────────────────

def researcher_node(state: PipelineState) -> dict:
    """
    Calls the Researcher Agent and returns its output as a state update.
    LangGraph automatically merges this dict into the shared state.
    """
    print("\n" + "█" * 60)
    print("█  NODE: RESEARCHER")
    print("█" * 60)

    try:
        agent = ResearcherAgent()
        leads = agent.find_leads(
            brief=state["brief"],
            target_count=state.get("target_count", 5),
        )
        return {"raw_leads": leads}
    except Exception as e:
        print(f"⚠️  Researcher failed: {e}")
        return {"raw_leads": [], "error": f"Researcher: {e}"}


def qualifier_node(state: PipelineState) -> dict:
    """Calls the Qualifier Agent on the raw leads."""
    print("\n" + "█" * 60)
    print("█  NODE: QUALIFIER")
    print("█" * 60)

    try:
        agent = QualifierAgent()
        qualified = agent.qualify_leads(
            raw_leads=state.get("raw_leads", []),
            min_score=50,
        )
        return {"qualified_leads": qualified}
    except Exception as e:
        print(f"⚠️  Qualifier failed: {e}")
        return {"qualified_leads": [], "error": f"Qualifier: {e}"}


def writer_node(state: PipelineState) -> dict:
    """Calls the Writer Agent on each qualified lead."""
    print("\n" + "█" * 60)
    print("█  NODE: WRITER")
    print("█" * 60)

    try:
        agent = WriterAgent()
        proposals = []
        qualified = state.get("qualified_leads", [])

        for i, lead in enumerate(qualified, 1):
            print(f"\n   Drafting proposal {i}/{len(qualified)}...")
            proposal = agent.write_proposal(lead)
            proposals.append(proposal)

        return {"proposals": proposals}
    except Exception as e:
        print(f"⚠️  Writer failed: {e}")
        return {"proposals": [], "error": f"Writer: {e}"}


# ─────────────────────────────────────────────────────
# CONDITIONAL EDGES — decide where to go after each node
# ─────────────────────────────────────────────────────

def after_researcher(state: PipelineState) -> str:
    """If no leads were found, skip the rest."""
    if not state.get("raw_leads"):
        print("\n⏩ No raw leads — ending pipeline early.")
        return END
    return "qualifier"


def after_qualifier(state: PipelineState) -> str:
    """If no leads survived qualification, skip the writer."""
    if not state.get("qualified_leads"):
        print("\n⏩ No qualified leads — skipping writer.")
        return END
    return "writer"


# ─────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────

def build_pipeline():
    """
    Construct the LangGraph state graph.
    Returns a compiled, runnable pipeline.
    """
    graph = StateGraph(PipelineState)

    # Register the three nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("qualifier", qualifier_node)
    graph.add_node("writer", writer_node)

    # Define the flow
    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", after_researcher,
                                 {"qualifier": "qualifier", END: END})
    graph.add_conditional_edges("qualifier", after_qualifier,
                                 {"writer": "writer", END: END})
    graph.add_edge("writer", END)

    return graph.compile()


class ManagerAgent:
    """
    Public interface for running the full multi-agent pipeline.

    Usage:
        manager = ManagerAgent()
        result = manager.run(
            brief="Stem cell research labs in Europe",
            target_count=3,
        )
        print(result["proposals"])
    """

    def __init__(self):
        self.pipeline = build_pipeline()

    def run(self, brief: str, target_count: int = 3) -> dict:
        """
        Run the full Researcher -> Qualifier -> Writer pipeline.

        Args:
            brief: Description of the leads to find.
            target_count: How many qualified leads to aim for.

        Returns:
            Final state dict with raw_leads, qualified_leads, proposals.
        """
        print("\n" + "═" * 60)
        print("  HEIDSTAR MULTI-AGENT SALES PIPELINE")
        print("═" * 60)
        print(f"  Brief: {brief}")
        print(f"  Target count: {target_count}")
        print("═" * 60)

        initial_state: PipelineState = {
            "brief": brief,
            "target_count": target_count,
        }

        final_state = self.pipeline.invoke(initial_state)
        return final_state


# ─────────────────────────────────────────────────────
# Quick self-test — runs the full pipeline end-to-end
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    manager = ManagerAgent()

    result = manager.run(
        brief="Stem cell research laboratories in Europe doing iPSC work",
        target_count=2,
    )

    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE — FINAL SUMMARY")
    print("═" * 60)
    print(f"  Raw leads found:        {len(result.get('raw_leads', []))}")
    print(f"  Qualified leads:        {len(result.get('qualified_leads', []))}")
    print(f"  Proposals drafted:      {len(result.get('proposals', []))}")
    if result.get("error"):
        print(f"  Error encountered:      {result['error']}")
    print("═" * 60)

    proposals = result.get("proposals", [])
    if proposals:
        print("\n  PROPOSALS:")
        for i, p in enumerate(proposals, 1):
            print(f"\n  {'─' * 56}")
            print(f"  PROPOSAL {i}: {p['organization_name']}")
            print(f"  Pitching: {p['product_pitched']}")
            print(f"  {'─' * 56}")
            print(p['markdown'])