"""Post-processing for the official metric keeps values in [0, 1] and does what it says."""
import numpy as np
import pytest

deep = pytest.importorskip("milk10k_pipeline.deep")


def _probs(n=50, k=11, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((n, k)) ** 3
    return P / P.sum(1, keepdims=True)


def test_argmax_rescale_gives_exactly_one_positive_per_row():
    P = _probs()
    Q = deep.rescale_argmax(P)
    assert ((Q >= 0) & (Q <= 1)).all()
    assert ((Q >= 0.5).sum(1) == 1).all()
    assert (Q.argmax(1) == P.argmax(1)).all()


def test_threshold_rescale_moves_each_threshold_to_half():
    P = _probs()
    t = np.linspace(0.1, 0.6, 11)
    Q = deep.rescale_thresholds(P, t)
    assert ((Q >= 0) & (Q <= 1)).all()
    assert ((Q >= 0.5) == (P >= t[None, :])).all()


def test_leaderboard_f1_perfect_and_raw():
    y = np.arange(11).repeat(3)
    P = np.eye(11)[y] * 0.9 + 0.01
    assert deep.leaderboard_f1(y, P) == pytest.approx(1.0)
