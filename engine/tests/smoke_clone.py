"""Smoke test: Env.clone() is a true deep copy.

Run with:
    PYTHONPATH=engine/build python engine/tests/smoke_clone.py

Used as a regression guard for the shared_ptr<HandEvaluator> refactor that
makes CFR-style tree traversal possible (see trainer/cfr/traversal.py).
"""
import pokertrainer_engine as pte


def call_idx(obs):
    legal = list(obs.legal)
    if pte.ActionType.CHECK_CALL in legal:
        return legal.index(pte.ActionType.CHECK_CALL)
    return 0


def main() -> None:
    env = pte.Env(seed=2026)
    env.reset(42)

    # Advance two preflop check-calls so we clone from a non-trivial state.
    for _ in range(2):
        env.step(call_idx(env.observation()))
    assert not env.is_terminal()

    hist_before   = env.state().history_size
    pot_before    = env.state().pot_chips
    to_act_before = env.to_act()
    hole_before   = list(env.state().hole)
    board_before  = list(env.state().board)

    # Clone, drive to terminal: parent must be untouched.
    clone = env.clone()
    n_steps = 0
    while not clone.is_terminal():
        clone.step(call_idx(clone.observation()))
        n_steps += 1
    assert clone.is_terminal()

    assert env.state().history_size == hist_before
    assert env.state().pot_chips    == pot_before
    assert env.to_act()             == to_act_before
    assert not env.is_terminal()
    assert list(env.state().hole)  == hole_before
    assert list(env.state().board) == board_before

    # Two independent clones from the same parent → same payoffs under same script.
    c2, c3 = env.clone(), env.clone()
    for c in (c2, c3):
        while not c.is_terminal():
            c.step(call_idx(c.observation()))
    assert c2.payoffs_bb() == c3.payoffs_bb() == clone.payoffs_bb()

    # 2000 clones don't blow up memory (shared HandEvaluator).
    clones = [env.clone() for _ in range(2000)]
    del clones

    print(f"OK — clone played {n_steps} steps to terminal, "
          f"payoffs={clone.payoffs_bb()}, parent intact")


if __name__ == "__main__":
    main()
