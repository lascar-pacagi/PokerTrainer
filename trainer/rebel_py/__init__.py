"""ReBeL for HU NLHE — pure-Python, vectorized over the 1326 hole combos.

Reimplements the ReBeL algorithm (Brown et al. 2020) against this project's
engine, reusing the 11-action abstraction of the CFR trainer. The cloned
Facebook reference (Liar's Dice, C++) lives under ``trainer/rebel/rebel`` and is
the algorithmic spec; this package is the poker port.

Phase 1 (this commit): the depth-limited CFR-D subgame solver + showdown values
+ exploitability, with NO value network — validated against the existing
push/fold Nash oracle and exact river endgames before any cluster compute.
"""
