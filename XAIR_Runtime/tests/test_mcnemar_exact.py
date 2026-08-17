"""Exact McNemar helper (no GPU)."""

from experiments.run_b2_validity_frontier import mcnemar_exact_p, mcnemar_yates_p


def test_exact_p_no_discordant():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_exact_p_one_sided_discordant():
    p = mcnemar_exact_p(149, 0)
    assert 0 < p < 1e-40
    # Two-sided exact: 2 * 0.5^149
    assert abs(p - 2.0 * (0.5**149)) < 1e-50


def test_exact_p_balanced_is_one():
    assert mcnemar_exact_p(5, 5) == 1.0


def test_yates_still_defined():
    chi2, p = mcnemar_yates_p(149, 0)
    assert chi2 > 0
    assert 0 < p < 1e-20
