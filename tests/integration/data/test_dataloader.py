"""Test the HeliosDataloader class."""

from pathlib import Path

import numpy as np
import pytest

from helios.data.constants import Modality
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset, collate_helios


def test_helios_dataloader(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
    )

    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1

    state_dict = dataloader.state_dict()
    dataloader.reset()
    dataloader.load_state_dict(state_dict)
    assert dataloader.batches_processed == batches_processed

    assert batches_processed == 1


def test_helios_dataloader_dataset_percentage(
    tmp_path: Path, setup_h5py_dir_20_samples: Path
) -> None:
    """Test the HeliosDataloader class."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir_20_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    len_dataset = len(dataset)
    assert len_dataset == 20
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=1,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
        dataset_percentage=0.5,
    )
    len_dataloader = len(dataloader)
    assert len_dataloader == 10

    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1

    assert dataloader.batches_processed == batches_processed


@pytest.mark.parametrize("dp_world_size", [2, 8])
def test_helios_dataloader_dataset_percentage_bigger_world_size(
    tmp_path: Path, setup_h5py_dir_100_samples: Path, dp_world_size: int
) -> None:
    """Test the HeliosDataloader class with different world sizes."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]
    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()
    len_dataset = len(dataset)
    assert len_dataset == 100
    assert isinstance(dataset, HeliosDataset)
    dataloader = HeliosDataLoader(
        dataset=dataset,
        work_dir=tmp_path,
        global_batch_size=16,
        dp_world_size=dp_world_size,
        dp_rank=0,
        fs_local_rank=0,
        seed=0,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
        dataset_percentage=0.5,
    )
    len_dataloader = len(dataloader)
    assert len_dataloader == 3

    dataloader.reshuffle()
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1
    assert dataloader.batches_processed == batches_processed


def test_dataset_percentage_consistent_across_epochs(
    tmp_path: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """Test that different epochs with same dataset percentage yield same unique indices."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]

    # Create first dataloader for epoch 1
    dataset1 = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    dataset1.prepare()

    # Helper to create dataset and dataloader with shared config
    def make_dataset_and_dataloader(work_dir):
        dataset = HeliosDataset(
            h5py_dir=setup_h5py_dir_100_samples,
            training_modalities=training_modalities,
            dtype=np.float32,
        )
        dataset.prepare()
        dataloader = HeliosDataLoader(
            dataset=dataset,
            work_dir=work_dir,
            global_batch_size=4,
            dp_world_size=1,
            dp_rank=0,
            fs_local_rank=0,
            seed=42,
            shuffle=True,
            num_workers=0,
            collator=collate_helios,
            target_device_type="cpu",
            token_budget=1000000,
            min_patch_size=1,
            max_patch_size=1,
            sampled_hw_p_list=[6],
            dataset_percentage=0.5,
        )
        return dataset, dataloader

    dataset1, dataloader1 = make_dataset_and_dataloader(tmp_path / "epoch1")
    dataset2, dataloader2 = make_dataset_and_dataloader(tmp_path / "epoch2")

    # Reshuffle for different epochs
    dataloader1.reshuffle(epoch=1)
    dataloader2.reshuffle(epoch=2)

    # The underlying dataset should have the same sample_indices since same seed was used for filtering
    assert np.array_equal(dataset1.sample_indices, dataset2.sample_indices), (
        "Same dataset_percentage with same seed should yield identical sample_indices across epochs"
    )


def test_concat_dataset_percentage_filtering(
    tmp_path: Path, setup_h5py_dir_20_samples: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """Test that dataset percentage filtering works with HeliosConcatDataset."""
    from helios.data.concat import HeliosConcatDataset

    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]

    # Create individual datasets
    dataset1 = HeliosDataset(
        h5py_dir=setup_h5py_dir_20_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    dataset2 = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    # Prepare individual datasets
    dataset1.prepare()
    dataset2.prepare()

    # Store original lengths
    original_len1 = len(dataset1)
    original_len2 = len(dataset2)

    # Create concat dataset
    concat_dataset = HeliosConcatDataset([dataset1, dataset2])
    concat_dataset.prepare()

    # Check original total length
    assert len(concat_dataset) == original_len1 + original_len2 == 120

    # Create dataloader with dataset percentage
    dataloader = HeliosDataLoader(
        dataset=concat_dataset,
        work_dir=tmp_path,
        global_batch_size=4,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        seed=42,
        shuffle=True,
        num_workers=0,
        collator=collate_helios,
        target_device_type="cpu",
        token_budget=1000000,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[6],
        dataset_percentage=0.5,
    )

    dataloader.reshuffle(epoch=1)

    # Each subdataset should be filtered to 50%
    assert len(dataset1) == int(original_len1 * 0.5) == 10
    assert len(dataset2) == int(original_len2 * 0.5) == 50

    # Total concat dataset should also be filtered
    assert len(concat_dataset) == 60  # 10 + 50

    # Test that we can iterate through the dataloader
    batches_processed = 0
    for batch in dataloader:
        batches_processed += 1
        assert batch is not None

    assert batches_processed > 0


def test_latlon_distribution_filtering(
    tmp_path: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """Test that latlon distribution is correctly updated before and after filtering."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]

    dataset = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )

    dataset.prepare()

    # Store original latlon distribution
    original_latlon = dataset.latlon_distribution.copy()
    original_sample_indices = dataset.sample_indices.copy()

    # Verify original state
    assert len(original_latlon) == 100
    assert len(original_sample_indices) == 100
    assert np.array_equal(original_sample_indices, np.arange(100))

    # Apply filtering
    dataset.filter_dataset_by_percentage(percentage=0.5, seed=42)

    # Check that dataset was filtered
    assert len(dataset.sample_indices) == 50
    assert len(dataset.latlon_distribution) == 50

    # Check that latlon_distribution corresponds to the filtered sample_indices
    expected_filtered_latlon = original_latlon[dataset.sample_indices]
    assert np.array_equal(dataset.latlon_distribution, expected_filtered_latlon), (
        "latlon_distribution should match the original distribution indexed by sample_indices"
    )

    # Test with concat dataset as well
    from helios.data.concat import HeliosConcatDataset

    # Create a second dataset for concat test
    dataset2 = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    dataset2.prepare()

    concat_dataset = HeliosConcatDataset([dataset, dataset2])
    concat_dataset.prepare()

    # Store original concat latlon distribution
    original_concat_latlon = concat_dataset.latlon_distribution.copy()
    assert len(original_concat_latlon) == 150  # 50 (filtered) + 100 (unfiltered)

    # Apply filtering to concat dataset
    concat_dataset.filter_dataset_by_percentage(percentage=0.5, seed=123)

    # Check that both subdatasets were filtered
    assert len(dataset.sample_indices) == 25  # 50 * 0.5
    assert len(dataset2.sample_indices) == 50  # 100 * 0.5
    assert len(concat_dataset.latlon_distribution) == 75  # 25 + 50

    # Verify latlon distribution is properly updated for concat dataset
    expected_concat_latlon = np.concatenate(
        [dataset.latlon_distribution, dataset2.latlon_distribution]
    )
    assert np.array_equal(concat_dataset.latlon_distribution, expected_concat_latlon), (
        "Concat dataset latlon_distribution should be concatenation of subdataset distributions"
    )


def test_dataset_percentage_deterministic_with_different_seeds(
    tmp_path: Path, setup_h5py_dir_100_samples: Path
) -> None:
    """Test that different seeds produce different but deterministic filtering results."""
    training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.OPENSTREETMAP_RASTER.name,
    ]

    # Create datasets with different seeds
    dataset1 = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    dataset1.prepare()

    dataset2 = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    dataset2.prepare()

    dataset3 = HeliosDataset(
        h5py_dir=setup_h5py_dir_100_samples,
        training_modalities=training_modalities,
        dtype=np.float32,
    )
    dataset3.prepare()

    # Filter with different seeds
    dataset1.filter_dataset_by_percentage(percentage=0.5, seed=42)
    dataset2.filter_dataset_by_percentage(percentage=0.5, seed=123)
    dataset3.filter_dataset_by_percentage(percentage=0.5, seed=42)  # Same as dataset1

    # Same seed should produce same results
    assert np.array_equal(dataset1.sample_indices, dataset3.sample_indices), (
        "Same seed should produce identical sample_indices"
    )
    assert np.array_equal(dataset1.latlon_distribution, dataset3.latlon_distribution), (
        "Same seed should produce identical latlon_distribution"
    )

    # Different seeds should produce different results (with high probability)
    assert not np.array_equal(dataset1.sample_indices, dataset2.sample_indices), (
        "Different seeds should produce different sample_indices"
    )

    # But all should have same length
    assert (
        len(dataset1.sample_indices)
        == len(dataset2.sample_indices)
        == len(dataset3.sample_indices)
        == 50
    )
