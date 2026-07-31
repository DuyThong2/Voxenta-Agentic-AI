"""Lightweight realtime error-analysis graph for practice (gói 11 mục 2.4c) --
fan-out Azure Pronunciation Assessment + a single-round LLM grammar/vocab/coherence pass,
fan-in to one TurnCorrection-shaped list. Runs concurrently with resolve-next-question, not
the full evalGraph (too heavy for a mid-session pause) and not part of followUpDecisionGraph
(different concern: this SINKS content, it doesn't decide should_continue).
"""
