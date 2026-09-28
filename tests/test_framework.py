import numpy as np

from alg_similarity import (
    ALGFrameworkSpec,
    GlobalSpec,
    LocalSpec,
    alg_framework_scores,
    framework_grid_size,
    global_grid_specs,
    local_grid_specs,
)


def test_complete_grid_size() -> None:
    assert len(local_grid_specs()) == 64
    assert len(global_grid_specs()) == 240
    assert framework_grid_size() == 168_960


def test_alg_star_is_bounded_and_symmetric() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(12, 8))
    b = rng.normal(loc=0.2, size=(12, 8))
    spec = ALGFrameworkSpec(
        LocalSpec("local_scaling", "manhattan", k=3),
        GlobalSpec("centroid", "cosine", "exponential", 2.0),
        0.7,
    )

    ab, _ = alg_framework_scores(a, b, [spec])
    ba, _ = alg_framework_scores(b, a, [spec])
    score_ab = ab[spec.metric_name]
    score_ba = ba[spec.metric_name]

    assert 0.0 <= score_ab <= 1.0
    assert np.isclose(score_ab, score_ba)

