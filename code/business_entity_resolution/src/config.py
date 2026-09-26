"""Central configuration. Every knob the ranking pipeline uses lives here (retrieval knobs: src/retrieval.py)."""
from dataclasses import dataclass, field, asdict


@dataclass
class BlockingConfig:
    # Retrieval (src/retrieval.py index) keeps k_fwd nearest records per S1 and k_rev nearest S1 per record.
    # The ranker can use fewer: a pair enters feature building if fwd_rank <= k_fwd or rev_rank <= k_rev.
    k_fwd: int = 50
    k_rev: int = 5
    chunk_entities: int = 100_000  # S1 entities per feature chunk (memory control)


@dataclass
class ModelConfig:
    # K-fold over the sampled "rank" entities (stratified by country x has-match). Every training pair gets an
    # out-of-fold prediction: that is what stage 2 stacks on and what the decision rule is tuned on.
    n_folds: int = 5
    seed: int = 42
    max_train_entities: int = 600_000  # of the ~1.76M "rank" entities; 0 = all of them
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "binary",      # plain log-loss keeps probabilities calibrated,
        "metric": "binary_logloss",  # which the expected-F0.5 decision layer relies on
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "verbose": -1,
        "num_threads": 0,
    })


@dataclass
class CascadeConfig:
    # Stage 1 prunes the retrieved pairs; the survivors are stage 2's input and the content of
    # candidate_pairs.tsv, whose size per S1 entity counts in the final ranking. Grid: p1 cutoff x cap on
    # pairs per entity (0 = no cap). Chosen: the smallest set whose oracle F0.5 drops by at most
    # max_oracle_drop, or -- if target_pairs_per_entity is set -- the best oracle F0.5 within that size.
    prune: bool = True
    max_oracle_drop: float = 0.002
    target_pairs_per_entity: float = 0.0
    caps: tuple = (0, 20, 15, 12, 10, 8, 7, 6, 5)
    cutoffs: tuple = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)


@dataclass
class DecisionConfig:
    # Grid searched on the out-of-fold predictions; the winner (by macro F0.5) is saved with the model.
    thresholds: tuple = tuple(round(0.05 * i, 2) for i in range(4, 20))  # 0.20 .. 0.95
    try_expected_f: bool = True
    try_one_to_one: tuple = (False, True)
    calib_gammas: tuple = (0.8, 1.0, 1.25)  # p -> p**gamma sharpening for expected-F
    max_items_expected_f: int = 15  # only the top-N probabilities per entity enter the DP
    max_tune_entities: int = 200_000  # expected-F rules are grid-searched on at most this many entities


@dataclass
class PipelineConfig:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self):
        return asdict(self)
