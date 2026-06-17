"""Pluribus for HU NLHE — tabular Linear-MCCFR blueprint + depth-limited search.

Reimplements the *method* of Pluribus (Brown & Sandholm, Science 2019) against
this project's engine, on 2-player NLHE (the engine is heads-up; ReBeL was
ported the same way). Pluribus's defining contrast with the repo's neural agents
(Deep CFR in ``trainer/cfr``, ReBeL in ``trainer/rebel_py``) is that it uses
**NO neural network**: the strategy lives in *regret tables* over an *abstracted*
game. It famously trained on a 64-core / 512 GB CPU server with no GPU in 8 days
— the scarce resource is RAM for the tables, not FLOPs.

═══════════════════════════════════════════════════════════════════════════════
GUIDED TOUR.  ``docs/pluribus_implementation/pluribus_implementation.pdf`` is the
high-level map: it walks every Pluribus idea (from the Pluribus section of
``docs/modern_poker_ai/modern_poker_ai_tutorial``) to the file and function that
realizes it. Each module docstring carries a ``Pluribus idea → … · doc §N`` line
so you can jump between code and that document in either direction.

MODULE MAP (bottom-up; each layer is validated before the one above is built):

  abstraction    abstraction   action grid (engine 11 slots) + info buckets
                               (169 lossless preflop classes + postflop
                               k-means on equity-distribution histograms)
  blueprint      infoset       abstract infoset key + regret/strategy tables
                 mccfr         external-sampling Linear MCCFR over the engine
                 blueprint     single-process trainer + queryable Blueprint
  online search  ranges        Bayes range from the betting line (blueprint)
                 continuations the 4 biased continuation strategies (Trick 1)
                 search        depth-limited re-solve over the WHOLE range
                               (Trick 2) + nested re-solving of off-tree sizes
                 play          the agent: blueprint preflop, search elsewhere
  cluster        blueprint_mp  CPU actor pool + additive table merge (no GPU)
                 validate      the Nash-oracle / exploitability gates

Phase 1: the tabular blueprint + abstraction, NO search — validated against the
existing 10bb push/fold Nash oracle. Phase 2: depth-limited online re-solving
with continuation strategies (reusing ``rebel_py`` subgame machinery). Phase 3:
the multi-core cluster blueprint trainer.

NOTE on BLAS threads (same as ``rebel_py.__init__``): the blueprint is a flood of
*tiny* numpy ops (regret matching over 11-vectors) — OpenBLAS' thread pool would
add ~33 ms of launch overhead per op, 30× the single-threaded cost. We pin BLAS
to one thread *before numpy loads* (importing this package runs before any
submodule imports numpy). The Phase-3 trainer is CPU-only and process-parallel,
so one thread per worker × many workers is exactly the right shape for a
64-core box; no GPU is involved.
"""
import os as _os  # noqa: E402
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
