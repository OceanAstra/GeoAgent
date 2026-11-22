# Scope Note: Paper P23 Retrospective Replay

The agent workflow evaluated in the manuscript is the **P23 historical feedback replay** implemented in `agent/feedback/p23_feedback_replay.py` (together with `p22_frozen_reference.py` and the conditional-memory variant in `p24_conditional_repair.py`). It operates on frozen six-class classifier probabilities, fixed auxiliary candidate heads, historical annual feedback windows, and a conservative KEEP-or-REPAIR gate.

Separate internal prototype branches (such as exploratory web/service wrappers) are intentionally excluded from this publication-oriented repository so that every included module maps directly to the reproducible experiments reported in the manuscript.
