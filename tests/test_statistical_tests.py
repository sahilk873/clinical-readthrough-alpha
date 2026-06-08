"""Tests for statistical tests module."""

import numpy as np
import pandas as pd

from clinical_alpha.returns.statistical_tests import (
    adjust_pvalues_multiple_testing,
    boehmer_test,
    bootstrap_car_ci,
    corrado_rank_test,
    generalized_sign_test,
    mann_whitney_test,
    patell_test,
    permutation_test,
    wilcoxon_signed_rank_test,
)


class TestBootstrapCARCI:
    def test_returns_dict_with_keys(self):
        cars = pd.Series([0.01, 0.02, -0.01, 0.005, -0.005, 0.015])
        result = bootstrap_car_ci(cars, n_iterations=500)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "std_err" in result
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_short_series(self):
        cars = pd.Series([0.01])
        result = bootstrap_car_ci(cars)
        assert result["mean"] == 0.01

    def test_empty_series(self):
        result = bootstrap_car_ci(pd.Series([], dtype=float))
        assert result["mean"] == 0.0


class TestPermutationTest:
    def test_two_sided(self):
        peer = [0.02, 0.03, 0.01, 0.04, 0.025]
        control = [-0.01, 0.0, -0.02, 0.005, -0.015]
        result = permutation_test(peer, control, n_iterations=500)
        assert "observed_diff" in result
        assert 0 <= result["p_value"] <= 1

    def test_greater_alternative(self):
        peer = [0.05, 0.06, 0.04]
        control = [-0.01, -0.02, 0.0]
        result = permutation_test(peer, control, n_iterations=500, alternative="greater")
        assert result["observed_diff"] > 0

    def test_empty_inputs(self):
        result = permutation_test([], [0.01], n_iterations=100)
        assert result["p_value"] == 1.0


class TestMultipleTestingAdjustment:
    def test_bonferroni(self):
        pvals = [0.01, 0.04, 0.5, 0.8]
        adjusted = adjust_pvalues_multiple_testing(pvals, method="bonferroni")
        assert all(a >= b for a, b in zip(adjusted, pvals))
        assert all(a <= 1.0 for a in adjusted)

    def test_benjamini_hochberg(self):
        pvals = [0.001, 0.01, 0.03, 0.05, 0.2, 0.8]
        adjusted = adjust_pvalues_multiple_testing(pvals, method="benjamini_hochberg")
        assert all(a >= b for a, b in zip(adjusted, pvals))

    def test_holm(self):
        pvals = [0.01, 0.02, 0.03, 0.4]
        adjusted = adjust_pvalues_multiple_testing(pvals, method="holm")
        assert all(a >= b for a, b in zip(adjusted, pvals))

    def test_empty(self):
        assert adjust_pvalues_multiple_testing([]) == []


class TestWilcoxonSignedRank:
    def test_returns_dict(self):
        peer = [0.02, 0.03, 0.01, 0.04]
        control = [0.0, 0.01, -0.01, 0.02]
        result = wilcoxon_signed_rank_test(peer, control)
        assert "statistic" in result
        assert "p_value" in result

    def test_small_sample(self):
        result = wilcoxon_signed_rank_test([0.01], [0.0])
        assert result["p_value"] == 1.0


class TestMannWhitney:
    def test_returns_dict(self):
        peer = [0.02, 0.03, 0.01, 0.04]
        control = [0.0, -0.01, 0.005]
        result = mann_whitney_test(peer, control)
        assert "statistic" in result
        assert "p_value" in result

    def test_identical_groups(self):
        data = [0.01, 0.02, 0.03]
        result = mann_whitney_test(data, data)
        assert result["p_value"] >= 0


class TestCorradoRankTest:
    def test_returns_dict(self):
        np.random.seed(42)
        est = pd.DataFrame({"A": np.random.randn(100) * 0.01, "B": np.random.randn(100) * 0.01})
        event = pd.DataFrame({"A": np.random.randn(10) * 0.02, "B": np.random.randn(10) * 0.02})
        result = corrado_rank_test(event, est, event_indices=[0, 1])
        assert "rank_statistic" in result
        assert "p_value" in result

    def test_empty_inputs(self):
        result = corrado_rank_test(pd.DataFrame(), pd.DataFrame(), [0])
        assert result["p_value"] == 1.0


class TestBoehmerTest:
    def test_returns_dict(self):
        np.random.seed(42)
        est = pd.DataFrame({"A": np.random.randn(100) * 0.01, "B": np.random.randn(100) * 0.01})
        event = pd.DataFrame({"A": np.random.randn(10) * 0.02, "B": np.random.randn(10) * 0.02})
        result = boehmer_test(event, est, event_indices=[0])
        assert "boehmer_statistic" in result
        assert "p_value" in result

    def test_empty(self):
        result = boehmer_test(pd.DataFrame(), pd.DataFrame(), [0])
        assert result["p_value"] == 1.0


class TestPatellTest:
    def test_returns_dict(self):
        np.random.seed(42)
        est = pd.DataFrame({"A": np.random.randn(100) * 0.01, "B": np.random.randn(100) * 0.01})
        event = pd.DataFrame({"A": np.random.randn(10) * 0.02, "B": np.random.randn(10) * 0.02})
        result = patell_test(event, est, event_indices=[0])
        assert "patell_statistic" in result
        assert "p_value" in result

    def test_empty(self):
        result = patell_test(pd.DataFrame(), pd.DataFrame(), [0])
        assert result["p_value"] == 1.0


class TestGeneralizedSignTest:
    def test_returns_dict(self):
        peer = [0.02, 0.03, -0.01, 0.04, 0.01]
        control = [0.0, 0.01, 0.0, -0.01, 0.005]
        result = generalized_sign_test(peer, control)
        assert "n_positive" in result
        assert "p_value" in result

    def test_small_sample(self):
        result = generalized_sign_test([0.01], [0.0])
        assert result["p_value"] == 1.0
