"""Tests for candidate_pool module."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from chem_informed_mtgp.candidate_pool import (
    iter_candidate_joint_batches,
    snap_dataframe,
)


# ========================================================================
#  Shared fixtures
# ========================================================================
@pytest.fixture
def param_config():
    """Standard parameter configuration for testing."""
    concentration_steps = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    return OrderedDict(
        [
            ("annealing_temp", {"range": (40, 150), "step": 5}),
            ("annealing_time", {"range": (60, 1800), "step": 30}),
            ("dispense_volume", {"range": (25, 300), "step": 25}),
            ("spin_speed", {"range": (1000, 6000), "step": 100}),
            ("dispense_speed", {"range": (80, 240), "step": 20}),
            (
                "concentration",
                {"range": (0.1, 2.0), "step": 0.25, "values": concentration_steps},
            ),
        ]
    )


@pytest.fixture
def a_site_to_smiles():
    """Standard site-to-SMILES mapping for testing."""
    return {"A": "CCO", "B": "c1ccccc1"}


@pytest.fixture
def concentration_steps():
    """Standard concentration steps for testing."""
    return np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])


# ========================================================================
#  Tests for snap_dataframe
# ========================================================================
class TestSnapDataframe:
    """Tests for the snap_dataframe function."""

    def test_snap_dataframe_basic(self, param_config):
        """Test basic snapping with range/step and values columns."""
        raw = pd.DataFrame(
            {
                "annealing_temp": [42.3, 147.8],
                "annealing_time": [61.2, 1799.6],
                "dispense_volume": [48, 298],
                "spin_speed": [1015, 5989],
                "dispense_speed": [81.2, 239.9],
                "concentration": [0.26, 1.73],
            }
        )
        snapped = snap_dataframe(raw, param_config)

        assert snapped["annealing_temp"].tolist() == [40.0, 150.0]
        assert snapped["concentration"].tolist() == [0.25, 1.75]
        assert snapped["annealing_time"].tolist() == [60.0, 1800.0]

    def test_snap_dataframe_returns_copy(self, param_config):
        """Test that snap_dataframe returns a copy, not modifying original."""
        raw = pd.DataFrame(
            {
                "annealing_temp": [42.3, 147.8],
                "annealing_time": [61.2, 1799.6],
                "dispense_volume": [48, 298],
                "spin_speed": [1015, 5989],
                "dispense_speed": [81.2, 239.9],
                "concentration": [0.26, 1.73],
            }
        )
        original_values = raw.copy()
        snapped = snap_dataframe(raw, param_config)

        # Original should be unchanged
        pd.testing.assert_frame_equal(raw, original_values)
        # Snapped should be different
        assert not snapped.equals(raw)

    def test_snap_dataframe_preserves_other_columns(self, param_config):
        """Test that columns not in param_config are preserved."""
        raw = pd.DataFrame(
            {
                "annealing_temp": [42.3],
                "annealing_time": [61.2],
                "dispense_volume": [48],
                "spin_speed": [1015],
                "dispense_speed": [81.2],
                "concentration": [0.26],
                "extra_col": ["keep_me"],
                "another_col": [3.14],
            }
        )
        snapped = snap_dataframe(raw, param_config)

        assert "extra_col" in snapped.columns
        assert snapped["extra_col"].tolist() == ["keep_me"]
        assert snapped["another_col"].tolist() == [3.14]

    def test_snap_dataframe_missing_column_raises_error(self, param_config):
        """Test that missing required columns raise ValueError."""
        incomplete = pd.DataFrame(
            {
                "annealing_temp": [42.3],
                "annealing_time": [61.2],
                # missing other columns
            }
        )
        with pytest.raises(ValueError, match="missing from the input DataFrame"):
            snap_dataframe(incomplete, param_config)

    def test_snap_dataframe_invalid_config_missing_range_with_values(self):
        """Test that 'values' without 'range' raises ValueError."""
        df = pd.DataFrame({"col": [0.5]})
        bad_config = {"col": {"values": [0.1, 0.5, 1.0]}}  # missing 'range'

        with pytest.raises(ValueError, match="missing 'range'"):
            snap_dataframe(df, bad_config)

    def test_snap_dataframe_invalid_config_missing_step(self):
        """Test that missing 'step' raises ValueError."""
        df = pd.DataFrame({"col": [42.3]})
        bad_config = {"col": {"range": (40, 50)}}  # missing 'step'

        with pytest.raises(ValueError, match="must contain either"):
            snap_dataframe(df, bad_config)

    def test_snap_dataframe_step_snapping_precision(self, param_config):
        """Test that step snapping respects epsilon tolerance."""
        raw = pd.DataFrame(
            {
                "annealing_temp": [40.0 + 1e-10, 150.0 - 1e-10],  # within epsilon
                "annealing_time": [60.0, 1800.0],
                "dispense_volume": [25.0, 300.0],
                "spin_speed": [1000.0, 6000.0],
                "dispense_speed": [80.0, 240.0],
                "concentration": [0.25, 1.75],
            }
        )
        snapped = snap_dataframe(raw, param_config)

        # Should snap to exact grid points despite tiny epsilon
        assert (snapped["annealing_temp"] >= 40.0).all()
        assert (snapped["annealing_temp"] <= 150.0).all()

    def test_snap_dataframe_preserves_index(self, param_config):
        """Test that the DataFrame index is preserved."""
        raw = pd.DataFrame(
            {
                "annealing_temp": [42.3, 147.8],
                "annealing_time": [61.2, 1799.6],
                "dispense_volume": [48, 298],
                "spin_speed": [1015, 5989],
                "dispense_speed": [81.2, 239.9],
                "concentration": [0.26, 1.73],
            },
            index=["sample1", "sample2"],
        )
        snapped = snap_dataframe(raw, param_config)

        assert snapped.index.tolist() == ["sample1", "sample2"]

    def test_snap_dataframe_concentration_values(
        self, param_config, concentration_steps
    ):
        """Test that concentration snaps to allowed values."""
        raw = pd.DataFrame(
            {
                "annealing_temp": [40.0],
                "annealing_time": [60.0],
                "dispense_volume": [25.0],
                "spin_speed": [1000.0],
                "dispense_speed": [80.0],
                "concentration": [0.3],  # Between 0.25 and 0.5
            }
        )
        snapped = snap_dataframe(raw, param_config)

        # Should snap to 0.25 (closest to 0.3)
        assert snapped["concentration"].iloc[0] == 0.25

        # Test another value
        raw.iloc[0, 5] = 0.6  # Between 0.5 and 0.75
        snapped = snap_dataframe(raw, param_config)
        assert snapped["concentration"].iloc[0] == 0.5


# ========================================================================
#  Tests for iter_candidate_joint_batches
# ========================================================================
class TestIterCandidateJointBatches:
    """Tests for the iter_candidate_joint_batches function."""

    def test_full_enumeration_bounds(self, param_config, a_site_to_smiles):
        """Test bounds_array and norm_bounds for full enumeration."""
        batch_iter, (bounds, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=10_000
        )

        assert bounds.shape == (6, 2)
        assert norm_bounds.shape == (6, 2)

        # bounds_array must reflect the declared range for every column
        expected_bounds = np.array(
            [[40, 150], [60, 1800], [25, 300], [1000, 6000], [80, 240], [0.1, 2.0]],
            dtype=float,
        )
        assert np.allclose(bounds, expected_bounds)

    def test_full_enumeration_norm_bounds_step_only_columns(
        self, param_config, a_site_to_smiles
    ):
        """Test that norm_bounds are [0, 1] for step-only columns."""
        batch_iter, (bounds, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=10_000
        )

        # For step-only columns the grid spans the full declared range → norm = [0, 1]
        assert np.allclose(norm_bounds[:5], [[0.0, 1.0]] * 5)

    def test_full_enumeration_norm_bounds_values_column(
        self, param_config, a_site_to_smiles
    ):
        """Test that norm_bounds reflect grid endpoints for 'values' columns."""
        batch_iter, (bounds, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=10_000
        )

        # For the 'values' column (concentration), grid endpoints [0.25, 1.75]
        # are a strict subset of [0.1, 2.0].
        conc_norm_min = (0.25 - 0.1) / (2.0 - 0.1)
        conc_norm_max = (1.75 - 0.1) / (2.0 - 0.1)

        assert np.isclose(norm_bounds[5, 0], conc_norm_min)
        assert np.isclose(norm_bounds[5, 1], conc_norm_max)

    def test_full_enumeration_grid_values(
        self, param_config, a_site_to_smiles, concentration_steps
    ):
        """Test that full enumeration produces grid values."""
        batch_iter, (bounds, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=10_000
        )

        allowed_conc = set(concentration_steps)

        for df, X in batch_iter:
            # annealing_temp should be on the step-5 grid
            assert ((df["annealing_temp"] - 40) % 5 < 1e-9).all()

            # concentration should be in allowed values
            assert (
                df["concentration"]
                .apply(lambda x: any(abs(x - a) < 1e-9 for a in allowed_conc))
                .all()
            )

            # X shape should match df
            assert X.ndim == 2 and X.shape[0] == len(df)
            break

    def test_snap_true_ignored_during_enumeration(self, param_config, a_site_to_smiles):
        """Test that snap=True is ignored during full enumeration."""
        batch_iter_false, (bounds_false, norm_bounds_false) = (
            iter_candidate_joint_batches(
                param_config, a_site_to_smiles, batch_size=10_000, snap=False
            )
        )
        batch_iter_true, (bounds_true, norm_bounds_true) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=10_000, snap=True
        )

        assert np.array_equal(bounds_false, bounds_true)
        assert np.array_equal(norm_bounds_false, norm_bounds_true)

    def test_random_sampling_snap_false_norm_bounds_all_zero_one(
        self, param_config, a_site_to_smiles
    ):
        """Test that snap=False random sampling has norm_bounds [0, 1]."""
        batch_iter, (_, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, n=500, random_state=42, snap=False
        )

        assert np.allclose(norm_bounds, [[0.0, 1.0]] * 6)

    def test_random_sampling_snap_false_continuous_values(
        self, param_config, a_site_to_smiles
    ):
        """Test that snap=False produces continuous (off-grid) values."""
        batch_iter, (_, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, n=500, random_state=42, snap=False
        )

        for df, X in batch_iter:
            # annealing_temp should NOT all be multiples of 5
            not_all_on_grid = not ((df["annealing_temp"] - 40) % 5 < 1e-9).all()
            assert not_all_on_grid

            # concentration should have many unique values (not just 7)
            assert df["concentration"].nunique() > 7

            # X shape should match df
            assert X.shape[0] == len(df)
            break

    def test_random_sampling_snap_true_grid_values(
        self, param_config, a_site_to_smiles, concentration_steps
    ):
        """Test that snap=True produces on-grid values."""
        batch_iter, (_, norm_bounds) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, n=500, random_state=42, snap=True
        )

        allowed_conc = set(concentration_steps)

        for df, X in batch_iter:
            # annealing_temp should be on the step-5 grid
            assert ((df["annealing_temp"] - 40) % 5 < 1e-9).all()

            # concentration should be in allowed values
            assert (
                df["concentration"]
                .apply(lambda x: any(abs(x - a) < 1e-9 for a in allowed_conc))
                .all()
            )

            # X shape should match df
            assert X.shape[0] == len(df)

            # Edge points must appear (uniform grid sampling)
            assert (df["annealing_temp"] == 40.0).any()
            assert (df["annealing_temp"] == 150.0).any()
            break

    def test_random_state_int_vs_generator(self, param_config, a_site_to_smiles):
        """Test that int seed and Generator produce same output."""
        it_int, (_, nb_int) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, n=100, snap=False, random_state=7
        )
        it_gen, (_, nb_gen) = iter_candidate_joint_batches(
            param_config,
            a_site_to_smiles,
            n=100,
            snap=False,
            random_state=np.random.default_rng(7),
        )

        for (df_int, _), (df_gen, _) in zip(it_int, it_gen):
            assert np.allclose(
                df_int[["annealing_temp", "concentration"]].values,
                df_gen[["annealing_temp", "concentration"]].values,
            )
            break

        assert np.array_equal(nb_int, nb_gen)

    def test_batch_size_respected(self, param_config, a_site_to_smiles):
        """Test that batch_size parameter is respected."""
        batch_size = 100
        batch_iter, _ = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=batch_size, n=500, snap=False
        )

        for df, X in batch_iter:
            assert len(df) <= batch_size
            assert X.shape[0] == len(df)

    def test_empty_continuous_columns_defaults_to_all(
        self, param_config, a_site_to_smiles
    ):
        """Test that empty continuous_columns defaults to all param_config keys."""
        # Empty list is falsy, so it defaults to param_config.keys()
        batch_iter, (bounds, _) = iter_candidate_joint_batches(
            param_config, a_site_to_smiles, batch_size=10_000, continuous_columns=[]
        )
        # Should work fine and produce the expected number of parameters
        assert bounds.shape[0] == len(param_config)

    def test_invalid_batch_size_raises_error(self, param_config, a_site_to_smiles):
        """Test that non-positive batch_size raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            iter_candidate_joint_batches(param_config, a_site_to_smiles, batch_size=0)
