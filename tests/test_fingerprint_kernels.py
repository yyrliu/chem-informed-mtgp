from __future__ import annotations

import numpy as np
import pytest

from chem_informed_mtgp.fingerprint_kernel import (
    FINGERPRINT_MODE_BITVECTOR,
    FINGERPRINT_MODE_INDEXED_TANIMOTO,
    FINGERPRINT_MODE_LKJ_CORR,
    LKJ_MASTER_NONE,
    FingerprintMetadata,
    joint_kernel_factory,
)
from chem_informed_mtgp.fingerprint_kernel_numba import joint_kernel_factory_numba

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
from helpers.gp_model import get_init_hyperparamter_and_bounds  # noqa: E402


def _make_metadata() -> FingerprintMetadata:
    a_sites = ("A", "B", "C")
    smiles = ("C", "CC", "CCC")
    fingerprints = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    tanimoto_correlations = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.6],
            [0.4, 0.6, 1.0],
        ],
        dtype=np.float64,
    )
    a_site_to_index = {name: idx for idx, name in enumerate(a_sites)}
    smiles_to_index = {name: idx for idx, name in enumerate(smiles)}
    return FingerprintMetadata(
        a_sites=a_sites,
        smiles=smiles,
        fingerprints=fingerprints,
        tanimoto_correlations=tanimoto_correlations,
        a_site_to_index=a_site_to_index,
        smiles_to_index=smiles_to_index,
    )


def test_joint_kernel_factories_match_for_bitvector_mode():
    kernel_np = joint_kernel_factory(
        cont_dims=2,
        fingerprint_bits=4,
        fingerprint_mode=FINGERPRINT_MODE_BITVECTOR,
    )
    kernel_nb = joint_kernel_factory_numba(
        cont_dims=2,
        fingerprint_bits=4,
        fingerprint_mode=FINGERPRINT_MODE_BITVECTOR,
    )

    x1 = np.array([[0.1, 0.2, 1.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    x2 = np.array(
        [[0.1, 0.2, 1.0, 0.0, 1.0, 0.0], [0.3, 0.4, 0.0, 1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    hps = np.array([1.7, 1.2, 0.8, 0.4], dtype=np.float64)

    out_np = kernel_np(x1, x2, hps)
    out_nb = kernel_nb(x1, x2, hps)

    assert out_np.shape == out_nb.shape == (1, 2)
    assert np.allclose(out_np, out_nb)


def test_joint_kernel_factories_match_for_indexed_tanimoto_with_sigma_scaling():
    metadata = _make_metadata()
    kernel_np = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
    )
    kernel_nb = joint_kernel_factory_numba(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
        tanimoto_correlations=metadata.tanimoto_correlations,
        sigma_per_chemical=True,
        sigma_n_categories=len(metadata.a_sites),
    )

    x1 = np.array([[0.1, 0.2, 0.0], [0.5, 0.6, 1.0]], dtype=np.float64)
    x2 = np.array([[0.1, 0.2, 1.0], [0.5, 0.6, 2.0]], dtype=np.float64)
    hps = np.array([2.0, 1.5, 0.7, 0.4, 1.0, 1.2, 0.8], dtype=np.float64)

    out_np = kernel_np(x1, x2, hps)
    out_nb = kernel_nb(x1, x2, hps)

    assert out_np.shape == out_nb.shape == (2, 2)
    assert np.allclose(out_np, out_nb)


def test_joint_kernel_factories_match_for_lkj_corr_mode():
    metadata = _make_metadata()
    kernel_np = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
    )
    kernel_nb = joint_kernel_factory_numba(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        sigma_n_categories=len(metadata.a_sites),
    )

    x1 = np.array([[0.1, 0.2, 0.0], [0.5, 0.6, 1.0]], dtype=np.float64)
    x2 = np.array([[0.1, 0.2, 1.0], [0.5, 0.6, 2.0]], dtype=np.float64)
    hps = np.array([2.0, 1.5, 0.7, 0.4, 1.0, 1.2, 0.8], dtype=np.float64)

    out_np = kernel_np(x1, x2, hps)
    out_nb = kernel_nb(x1, x2, hps)

    assert out_np.shape == out_nb.shape == (2, 2)
    assert np.allclose(out_np, out_nb)


def test_lkj_master_category_changes_which_category_is_symmetric():
    metadata = _make_metadata()  # a_sites = ("A", "B", "C") -> indices 0, 1, 2
    # Per-category sigma scales all set to 1.0 so only the LKJ correlation
    # structure shows through the kernel values.
    hps = np.array([2.0, 1.5, 0.7, 0.4, 1.0, 1.0, 1.0], dtype=np.float64)
    x = np.array([[0.1, 0.2, 0.0], [0.1, 0.2, 1.0], [0.1, 0.2, 2.0]], dtype=np.float64)

    kernel_default = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
    )
    K_default = kernel_default(x, x, hps)
    # Default master is category 0 ("A"): corr(A,B) == corr(A,C), corr(B,C) differs.
    assert np.isclose(K_default[0, 1], K_default[0, 2])
    assert not np.isclose(K_default[1, 2], K_default[0, 1])

    kernel_master_b = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        lkj_master_category="B",
    )
    K_master_b = kernel_master_b(x, x, hps)
    # With "B" as master: corr(B,A) == corr(B,C), corr(A,C) differs.
    assert np.isclose(K_master_b[1, 0], K_master_b[1, 2])
    assert not np.isclose(K_master_b[0, 2], K_master_b[1, 0])

    # Name and index resolve to the same result.
    kernel_master_1 = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        lkj_master_category=1,
    )
    assert np.allclose(kernel_master_1(x, x, hps), K_master_b)


def test_joint_kernel_factories_match_for_lkj_corr_mode_with_master_category():
    metadata = _make_metadata()
    kernel_np = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        lkj_master_category="C",
    )
    kernel_nb = joint_kernel_factory_numba(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        sigma_n_categories=len(metadata.a_sites),
        lkj_master_category="C",
    )

    x1 = np.array([[0.1, 0.2, 0.0], [0.5, 0.6, 1.0]], dtype=np.float64)
    x2 = np.array([[0.1, 0.2, 1.0], [0.5, 0.6, 2.0]], dtype=np.float64)
    hps = np.array([2.0, 1.5, 0.7, 0.4, 1.0, 1.2, 0.8], dtype=np.float64)

    out_np = kernel_np(x1, x2, hps)
    out_nb = kernel_nb(x1, x2, hps)

    assert np.allclose(out_np, out_nb)
    # sanity: differs from default (master=0/"A")
    kernel_nb_default = joint_kernel_factory_numba(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        sigma_n_categories=len(metadata.a_sites),
    )
    assert not np.allclose(out_nb, kernel_nb_default(x1, x2, hps))


def test_joint_kernel_factories_match_for_full_lkj_corr_mode():
    metadata = _make_metadata()
    kernel_np = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        lkj_master_category=LKJ_MASTER_NONE,
    )
    kernel_nb = joint_kernel_factory_numba(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        sigma_n_categories=len(metadata.a_sites),
        lkj_master_category=LKJ_MASTER_NONE,
    )

    x1 = np.array([[0.1, 0.2, 0.0], [0.5, 0.6, 1.0]], dtype=np.float64)
    x2 = np.array([[0.1, 0.2, 1.0], [0.5, 0.6, 2.0]], dtype=np.float64)
    # One extra hyperparameter versus master-category mode (3 LKJ params instead of 2).
    hps = np.array([2.0, 1.5, 0.7, 0.4, 0.9, 1.0, 1.2, 0.8], dtype=np.float64)

    out_np = kernel_np(x1, x2, hps)
    out_nb = kernel_nb(x1, x2, hps)

    assert out_np.shape == out_nb.shape == (2, 2)
    assert np.allclose(out_np, out_nb)


def test_full_lkj_corr_has_no_forced_equal_correlations():
    metadata = _make_metadata()  # a_sites = ("A", "B", "C") -> indices 0, 1, 2
    kernel = joint_kernel_factory(
        cont_dims=2,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        lkj_master_category=LKJ_MASTER_NONE,
    )
    # a, ls1, ls2, b, c, sigma0=sigma1=sigma2=1.0 (neutralize sigma scaling)
    hps = np.array([0.5, 1.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    x = np.array([[0.1, 0.2, 0.0], [0.1, 0.2, 1.0], [0.1, 0.2, 2.0]], dtype=np.float64)

    K = kernel(x, x, hps)

    # All three pairwise correlations should be free to differ (unlike
    # master-category mode, where two of them are forced equal).
    assert not np.isclose(K[0, 1], K[0, 2])
    assert not np.isclose(K[0, 1], K[1, 2])
    assert not np.isclose(K[0, 2], K[1, 2])


def test_get_init_hyperparamter_and_bounds_full_lkj_has_one_extra_hyperparameter():
    _, bounds_master = get_init_hyperparamter_and_bounds(
        "score",
        n_features=6,
        n_signal_var=3,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
    )
    init_full, bounds_full = get_init_hyperparamter_and_bounds(
        "score",
        n_features=6,
        n_signal_var=3,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        lkj_master_category=LKJ_MASTER_NONE,
    )

    assert bounds_full.shape[0] == bounds_master.shape[0] + 1
    n_features = 6
    fp_weight_idx = 1 + n_features  # signal_var/a slot, then lengthscales
    assert np.allclose(bounds_full[fp_weight_idx], [-3.0, 3.0])  # b (fp weight slot)
    assert np.allclose(
        bounds_full[fp_weight_idx + 1], [-3.0, 3.0]
    )  # c (inserted LKJ param)

    metadata = _make_metadata()
    kernel_nb = joint_kernel_factory_numba(
        cont_dims=6,
        fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
        fingerprint_metadata=metadata,
        sigma_per_chemical=True,
        sigma_n_categories=len(metadata.a_sites),
        lkj_master_category=LKJ_MASTER_NONE,
    )
    x = np.array([[0.1] * 6 + [0.0], [0.1] * 6 + [1.0]], dtype=np.float64)
    out = kernel_nb(x, x, init_full)
    assert out.shape == (2, 2)


@pytest.mark.parametrize("factory", [joint_kernel_factory, joint_kernel_factory_numba])
def test_lkj_corr_requires_sigma_per_chemical(factory):
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="requires sigma_per_chemical=True"):
        factory(
            cont_dims=2,
            fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
            fingerprint_metadata=metadata,
            sigma_per_chemical=False,
        )


def test_lkj_corr_requires_exactly_three_categories():
    a_sites = ("A", "B")
    smiles = ("C", "CC")
    fingerprints = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    tanimoto_correlations = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64)
    metadata = FingerprintMetadata(
        a_sites=a_sites,
        smiles=smiles,
        fingerprints=fingerprints,
        tanimoto_correlations=tanimoto_correlations,
        a_site_to_index={name: idx for idx, name in enumerate(a_sites)},
        smiles_to_index={name: idx for idx, name in enumerate(smiles)},
    )

    with pytest.raises(ValueError, match="only supports exactly"):
        joint_kernel_factory(
            cont_dims=2,
            fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
            fingerprint_metadata=metadata,
            sigma_per_chemical=True,
        )

    with pytest.raises(ValueError, match="only supports exactly"):
        joint_kernel_factory_numba(
            cont_dims=2,
            fingerprint_mode=FINGERPRINT_MODE_LKJ_CORR,
            sigma_per_chemical=True,
            sigma_n_categories=len(a_sites),
        )


@pytest.mark.parametrize("factory", [joint_kernel_factory, joint_kernel_factory_numba])
def test_sigma_per_chemical_requires_indexed_tanimoto(factory):
    with pytest.raises(ValueError, match="sigma_per_chemical=True is only supported"):
        factory(
            cont_dims=2,
            fingerprint_mode=FINGERPRINT_MODE_BITVECTOR,
            sigma_per_chemical=True,
        )


def test_sigma_per_chemical_requires_metadata_or_correlations():
    with pytest.raises(ValueError, match="fingerprint_metadata is required"):
        joint_kernel_factory(
            cont_dims=2,
            fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
            sigma_per_chemical=True,
        )

    with pytest.raises(ValueError, match="tanimoto_correlations must be provided"):
        joint_kernel_factory_numba(
            cont_dims=2,
            fingerprint_mode=FINGERPRINT_MODE_INDEXED_TANIMOTO,
            sigma_per_chemical=True,
        )
