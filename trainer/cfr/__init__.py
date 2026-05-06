"""Deep CFR trainer for HU NLHE.

External-sampling Monte-Carlo CFR with neural-network approximation of
cumulative regrets and the average strategy (Brown et al. 2019), with
Linear CFR weighting (Brown & Sandholm 2019).

Sits alongside `trainer/dmc/` (the DouZero-style DMC trainer). The two are
independent — different algorithms, different buffers, different nets.
"""
