"""
test_train_checkpoint.py
=========================
Unit tests for TrainingFeatureCheckpointManager (atomic chunk persistence for ML training).
"""

import numpy as np
import pytest
from pathlib import Path
from src.xgb_matcher.checkpoint import TrainingFeatureCheckpointManager


def test_training_feature_checkpoint_lifecycle(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints_test"
    feature_names = ["feat1", "feat2", "feat3"]
    manager = TrainingFeatureCheckpointManager(
        tag="test_tag",
        checkpoint_dir=ckpt_dir,
        sample_size=10,
        random_seed=42,
        feature_names=feature_names,
        chunk_size=3,
    )

    assert not manager.can_resume()

    # Add 5 samples (3 train, 2 dev)
    # Sample 0 (Train, 2 candidates)
    f0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y0 = np.array([1.0, 0.0], dtype=np.float32)
    manager.add_sample(0, "train", f0, y0, 2)

    # Sample 1 (Dev, 1 candidate)
    f1 = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    y1 = np.array([1.0], dtype=np.float32)
    manager.add_sample(1, "dev", f1, y1, 1)

    # Sample 2 (Train, 3 candidates) -> triggers chunk flush (size 3)
    f2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    y2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    manager.add_sample(2, "train", f2, y2, 3)

    # Verify chunk was written
    assert (ckpt_dir / "manifest.json").exists()
    assert manager.can_resume()

    # Add sample 3 (Train, 1 candidate)
    f3 = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)
    y3 = np.array([1.0], dtype=np.float32)
    manager.add_sample(3, "train", f3, y3, 1)

    # Finalize at index 3
    manager.finalize(last_idx=3)

    # Now simulate a crash/restart by creating a new manager with the same parameters
    resume_manager = TrainingFeatureCheckpointManager(
        tag="test_tag",
        checkpoint_dir=ckpt_dir,
        sample_size=10,
        random_seed=42,
        feature_names=feature_names,
        chunk_size=3,
    )

    assert resume_manager.can_resume()

    (
        last_idx,
        train_X_list,
        train_y_list,
        train_groups,
        dev_X_list,
        dev_y_list,
        dev_groups,
    ) = resume_manager.load_checkpoint()

    assert last_idx == 3
    # Total train queries: sample 0 (2 cands), sample 2 (3 cands), sample 3 (1 cand) -> 3 queries, 6 candidates total
    assert len(train_groups) == 3
    assert train_groups == [2, 3, 1]
    X_train = np.vstack(train_X_list)
    assert X_train.shape == (6, 3)
    y_train = np.concatenate(train_y_list)
    assert len(y_train) == 6

    # Total dev queries: sample 1 (1 cand)
    assert len(dev_groups) == 1
    assert dev_groups == [1]
    X_dev = np.vstack(dev_X_list)
    assert X_dev.shape == (1, 3)
    assert np.allclose(X_dev, f1)


def test_training_feature_checkpoint_mismatch_protection(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints_mismatch"
    feature_names = ["feat1", "feat2"]
    manager = TrainingFeatureCheckpointManager(
        tag="test_mismatch",
        checkpoint_dir=ckpt_dir,
        sample_size=5,
        random_seed=42,
        feature_names=feature_names,
        chunk_size=2,
    )

    f0 = np.array([[1.0, 2.0]], dtype=np.float32)
    y0 = np.array([1.0], dtype=np.float32)
    manager.add_sample(0, "train", f0, y0, 1)
    manager.finalize(0)

    # Manager with different seed should reject resume
    diff_seed_manager = TrainingFeatureCheckpointManager(
        tag="test_mismatch",
        checkpoint_dir=ckpt_dir,
        sample_size=5,
        random_seed=999,
        feature_names=feature_names,
    )
    assert not diff_seed_manager.can_resume()

    # Manager with different features should reject resume
    diff_feat_manager = TrainingFeatureCheckpointManager(
        tag="test_mismatch",
        checkpoint_dir=ckpt_dir,
        sample_size=5,
        random_seed=42,
        feature_names=["feat1", "different_feat"],
    )
    assert not diff_feat_manager.can_resume()
