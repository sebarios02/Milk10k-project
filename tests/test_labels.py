import numpy as np

from milk10k_pipeline import labels


def test_label_map_covers_all_classes():
    lm = labels.build_label_map()
    assert set(lm["fine"]["merge"]) == {"AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                                         "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"}
    assert set(lm["fine"]["merge"].values()) == set(lm["fine"]["classes"])
    assert sorted(lm["primary"]["to_index"].values()) == [0, 1, 2]


def test_balanced_weights_formula():
    y = np.array([0] * 8 + [1] * 2)
    w = labels.balanced_class_weights(y, 2)
    assert np.allclose(w, [10 / (2 * 8), 10 / (2 * 2)])
    assert np.isclose((w[y]).sum(), len(y))          # weighted counts sum to N


def test_sampler_weights_equalise_classes():
    y = np.array([0] * 90 + [1] * 10)
    sw = labels.sample_weights_for_sampler(y)
    assert np.isclose(sw[y == 0].sum(), sw[y == 1].sum())
