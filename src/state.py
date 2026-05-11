"""
state.py
Defines the shared state that flows through the multi-agent graph.

In LangGraph, every node (agent) receives the current state, does work,
and returns updates to the state. The state is a TypedDict — a dictionary
with declared field types so the editor and LangGraph can validate it.
"""

from typing import TypedDict, Optional


class PipelineState(TypedDict, total=False):
    """
    The single shared memory passed between all agents in the graph.

    'total=False' means every field is optional — agents can fill in fields
    as the pipeline progresses. The Researcher fills 'raw_leads', the
    Qualifier fills 'qualified_leads', the Writer fills 'proposals'.

    Fields:
        brief: The user's input — what kind of leads to find.
        target_count: How many qualified leads we want at the end.
        raw_leads: List of leads from Researcher (unfiltered).
        qualified_leads: List of leads from Qualifier (filtered + scored).
        proposals: List of proposal dicts from Writer.
        error: If any agent fails, the error message lives here.
    """
    # Input
    brief: str
    target_count: int

    # Filled by Researcher
    raw_leads: list[dict]

    # Filled by Qualifier
    qualified_leads: list[dict]

    # Filled by Writer
    proposals: list[dict]

    # Error tracking
    error: Optional[str]