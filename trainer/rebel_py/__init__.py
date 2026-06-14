"""ReBeL for HU NLHE — pure-Python, vectorized over the 1326 hole combos.

Reimplements the ReBeL algorithm (Brown et al. 2020) against this project's
engine, reusing the 11-action abstraction of the CFR trainer. The cloned
Facebook reference (Liar's Dice, C++) lives under ``trainer/rebel/rebel`` and is
the algorithmic spec; this package is the poker port.

═══════════════════════════════════════════════════════════════════════════════
GUIDED TOUR.  ``docs/rebel_implementation/rebel_implementation.pdf`` is the
high-level map: it walks every ReBeL idea (from ``docs/modern_poker_ai_tutorial``)
to the file and function that realizes it, with the architecture and value-net
diagrams. Each module docstring below carries a ``ReBeL idea → … · doc §N`` line
so you can jump between code and that document in either direction.

MODULE MAP (bottom-up; each layer is validated before the one above is built):

  game machinery     hand_index   1326 combos + card-removal masks
                     showdown      rank/incidence terminal values (no dense matrix)
                     public_tree   depth-limited subgames, the chance node, net leaves
                     subgame_solver  vectorized CFR-D (the search)
                     exploitability  best response = the correctness gate
  value-net loop     pbs           the public belief state and its query encoding
                     models        V(PBS), the value network
                     recursive     RlRunner self-play + recursive_strategy
                     replay, train, config   reservoir + the training loop
  cluster            rebel_mp_gpu  CPU actors + GPU learner;  validate  manual gates
═══════════════════════════════════════════════════════════════════════════════

Phase 1: the depth-limited CFR-D subgame solver + showdown values +
exploitability, with NO value network — validated against the existing push/fold
Nash oracle and exact river endgames. Phase 2: a value net + recursive self-play
(ReBeL proper). Phase 3: the cluster trainer.

NOTE on BLAS threads: the solver is a flood of *small* matrix-vector products
(per-node (1326, n_actions) over the hand vector). With OpenBLAS' default thread
pool, each tiny matvec pays ~33 ms of thread-launch overhead — 30× slower than
the ~1 ms single-threaded cost. We pin BLAS to one thread *before numpy loads*
(importing this package runs before any submodule imports numpy). A GPU learner
(Phase 3) is unaffected — its heavy work is on the GPU, numpy stays small.
"""
import os as _os  # noqa: E402
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
