import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from autoencoder_novelty import Autoencoder, auc, novelty


def test_autoencoder_learns_low_rank_data():
    r = np.random.RandomState(1)
    X = r.randn(60, 3) @ r.randn(3, 20)          # data that truly lives in 3 dimensions
    ae = Autoencoder(20, hidden=4)
    before = ae.errors(X).mean()
    ae.fit(X, epochs=600)
    assert ae.errors(X).mean() < 0.2 * before


def test_outlier_gets_higher_error_than_typical():
    r = np.random.RandomState(2)
    X = r.randn(80, 3) @ r.randn(3, 20)
    ae = Autoencoder(20, hidden=4).fit(X, epochs=600)
    outlier = r.randn(1, 20) * 3
    assert ae.errors(outlier)[0] > ae.errors(X).mean()


def test_auc_and_novelty():
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1, 2], [1, 2]) == 0.5 and auc([1], [2]) == 0.0
    assert novelty(5, [1, 2, 3, 4]) == 1.0 and novelty(0, [1, 2]) == 0.0
