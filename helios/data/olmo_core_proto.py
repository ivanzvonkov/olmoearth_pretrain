"""Trying to prototype fitting everything into olmo core."""
# ruff: noqa
# mypy: ignore-errors

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import islice
from typing import Any, cast

import numpy as np
import rioxarray
import torch
import xarray as xr
from einops import rearrange
from olmo_core.data.data_loader import NumpyDataLoaderBase
from olmo_core.data.numpy_dataset import NumpyDatasetBase
from olmo_core.data.utils import get_rng
from olmo_core.utils import roundrobin, threaded_generator
from torch.utils.data import Dataset
from upath import UPath

from helios.data.dataset import DATA_SOURCE_TO_VARIATION_TYPE, S2_BANDS
from helios.data.index import DatasetIndexParser

# We need to be able to either subclass or replace for this to work

# Can we write our dataset as a subclass of olmo core's dataset?
# Can our collator replace olmo core's collator?


# Eventually we don't want a dataset index to be forced we want to be able to pass anything like a dataset index specific to this
# and have it be able to be used by the dataset


# I want a class that parses dataset structure and just returns a list of dicts or namedtuples that are samples


# the dataset index Parser outputs a list of dicts with all the info needed to load the sample plus sample_metadata
class HeliosDataset(NumpyDatasetBase, Dataset):
    """Helios dataset."""

    def __init__(self, *samples: dict[str, Any], dtype: np.dtype):
        """ "init

        Things that would need to be optional or should be forgotten about, or changed
        - paths would need to ba dictionary or collection of paths for this to work
        - pad_token_id: int,
        - eos_token_id: int,
        - vocab_size: int,
        """
        super().__init__(
            *samples,
            dtype=dtype,
            pad_token_id=-1,  # Not needed only LM
            eos_token_id=-1,  # Not needed only LM
            vocab_size=-1,  # Not needed only LM
        )
        #
        # What does it look like for me to access paths?

        # After init we have
        # paths to samples
        # numpy data type
        pass

    @property
    def max_sequence_length(self) -> int:
        """Max sequence length."""
        # NOT SUPER needed
        return max(item["num_timesteps"] for item in self.paths)

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the dataset."""
        # LM specific
        raise NotImplementedError("Fingerprint not implemented")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.paths)

    def _tif_to_array(
        self, tif_path: UPath | str, data_source: str
    ) -> tuple[np.ndarray, float]:
        """Convert a tif file to an array.

        Args:
            tif_path: The path to the tif file.
            data_source: The data source string to load the correct datasource
        Returns:
            The array from the tif file.
        """
        if data_source == "sentinel2":
            space_bands = S2_BANDS
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        # We will need different ingestion logic for different data sources at this point

        variation_type = DATA_SOURCE_TO_VARIATION_TYPE[data_source]
        if variation_type == "space_time_varying":
            with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
                # [all_combined_bands, H, W]
                # all_combined_bands includes all dynamic-in-time bands
                # interleaved for all timesteps
                # followed by the static-in-time bands
                values = cast(np.ndarray, data.values)
                # lon = np.mean(cast(np.ndarray, data.x)).item()
                # lat = np.mean(cast(np.ndarray, data.y)).item()

            num_timesteps = values.shape[0] / len(space_bands)
            assert (
                num_timesteps % 1 == 0
            ), f"{tif_path} has incorrect number of channels {space_bands} \
                {values.shape[0]=} {len(space_bands)=}"
            space_time_x = rearrange(
                values, "(t c) h w -> h w t c", c=len(space_bands), t=int(num_timesteps)
            )
            return space_time_x, num_timesteps
        else:
            raise NotImplementedError(f"Unknown variation type: {variation_type}")

    def _tif_to_array_with_checks(
        self, tif_path: UPath | str, data_source: str
    ) -> tuple[np.ndarray, float]:
        """Load the tif file and return the array.

        Args:
            tif_path: The path to the tif file.
            data_source: The data source.

        Returns:
            The array from the tif file.
        """
        try:
            output = self._tif_to_array(tif_path, data_source)
            return output
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            raise e

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the item at the given index."""
        sample = self.paths[index]  # THis really is the dict of all the sample info
        data_source_paths = sample["data_source_paths"]
        data_arrays = {}
        for data_source, tif_path in data_source_paths.items():
            data_source_array, num_timesteps = self._tif_to_array_with_checks(
                tif_path, data_source
            )
            data_arrays[data_source] = data_source_array
        output_dict: dict[str, Any] = {"data_arrays": data_arrays}
        output_dict["sample_metadata"] = sample["sample_metadata"]
        output_dict["num_timesteps"] = num_timesteps
        output_dict["data_source_metadata"] = sample["data_source_metadata"]
        return output_dict


def iter_batched_helios(
    iterable: Iterable[dict[str, Any]], local_batch_size: int
) -> Iterable[tuple[dict[str, Any], ...]]:
    batch: list[dict[str, Any]] = []
    instances = 0
    # shape: Optional[tuple[int, ...]] = None
    for x in iterable:
        if instances > local_batch_size:
            yield tuple(batch)
            batch.clear()
            instances = 0
            # shape = None

        batch.append(x)
        instances += 1

        # TODO:  Our shape checking is more complex we likely should do this later
        # if shape is not None and shape != x["input_ids"].shape:
        #     raise RuntimeError(
        #         f"Items in batch don't have the same shape! Expected {shape}, "
        #         f"got {tuple(x['input_ids'].shape)}"
        #     )
        # shape = tuple(x["input_ids"].shape)

    if batch:
        yield tuple(batch)


## IterableDataset wrapper extension
from olmo_core.data.data_loader import _IterableDatasetWrapper


class HeliosIterableDatasetWrapper(_IterableDatasetWrapper):
    """Helios iterable dataset wrapper."""

    def __init__(self, data_loader: NumpyDataLoaderBase):
        super().__init__(data_loader)

    def __iter__(self) -> Iterator[dict]:
        """Iterate over the dataset."""
        global_indices = self.data_loader.get_global_indices()

        num_threads = self.data_loader.num_threads
        if self.worker_info is None and self.data_loader.num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll
            # try to guess a good number of threads.
            num_threads = 4

        # Potentially slice by threads.
        instance_iterator: Iterator[dict[str, Any]]
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the maximum number of instances per batch locally.
            max_instances_per_rank: int
            if isinstance(self.dataset, HeliosDataset):
                max_instances_per_rank = self.data_loader.rank_batch_size
            else:
                raise NotImplementedError

            queue_size = math.ceil(max_instances_per_rank * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                # NOTE: `_get_local_instance_indices` might return an iterator, so we have to
                # create a unique one for each thread otherwise it would be exhausted prematurely
                # and give the wrong order.
                indices = self.data_loader._get_local_instance_indices(global_indices)
                generator = (
                    self.data_loader._get_dataset_item(int(idx))
                    for idx in islice(indices, i, None, num_threads)
                )
                thread_generators.append(
                    threaded_generator(
                        generator, maxsize=queue_size, thread_name=f"data thread {i}"
                    )
                )

            instance_iterator = roundrobin(*thread_generators)
        else:
            indices = self.data_loader._get_local_instance_indices(global_indices)
            instance_iterator = (
                self.data_loader._get_dataset_item(int(idx)) for idx in indices
            )
        print(f"data loader rank batch size {self.data_loader.rank_batch_size}")
        return (
            self.data_loader.collator(batch)
            for batch in iter_batched_helios(
                instance_iterator, self.data_loader.rank_batch_size
            )
        )


class HeliosDataLoader(NumpyDataLoaderBase):
    """Helios data loader."""

    def __init__(
        self,
        dataset: NumpyDatasetBase,
        *,
        collator: Callable[
            [Sequence[dict]], dict
        ],  # Shouls probl make the data collator into a class at some point to match
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            collator=collator,
            **kwargs,
        )

    @classmethod
    def wrap_numpy_dataset(
        cls,
        dataset: NumpyDatasetBase,
        *,
        global_batch_size: int,
        collator: Callable[[Sequence[dict]], dict],
        work_dir: UPath | None = None,
        seed: int = 0,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        num_threads: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        target_device_type: str = "cpu",
    ) -> "NumpyDataLoaderBase":
        """Construct the corresponding :class:`NumpyDataLoaderBase` instance for the given :class:`NumpyDatasetBase`.

        :param dataset: The dataset to wrap.
        """
        kwargs = dict(
            global_batch_size=global_batch_size,
            collator=collator,
            work_dir=work_dir or dataset.work_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
            seed=seed,
            num_threads=num_threads,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            target_device_type=target_device_type,
        )
        if isinstance(dataset, HeliosDataset):
            data_loader = HeliosDataLoader(
                dataset,
                **kwargs,  # type: ignore
            )
        else:
            raise NotImplementedError

        return data_loader

    @property
    def _global_indices_file(self) -> UPath:
        """Global indices file."""
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,
            size=self.total_size,
        )
        return UPath(self.work_dir) / f"{global_indices_fname}.npy"

    def _build_global_indices(self) -> np.ndarray:
        """Build global indices.

        This an array of all the training indices that wil be used globally in the distributed
        setup.
        """
        rng: np.random.Generator | None = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = get_rng(self.seed + self.epoch)
        indices = np.arange(len(self.dataset), dtype=np.uint32)
        if rng is not None:
            rng.shuffle(indices)
        # what shape should this be?
        # TODO:Remove tail of data to make it evenly divisible, not sure yet if we need this
        print(f"indices shape before removing tail {indices.shape}")
        indices = indices[: self.total_size]
        print(f"indices shape after removing tail {indices.shape}")
        return indices

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """Get local instance indices."""
        assert isinstance(self.dataset, HeliosDataset)
        # Slice up by batch.
        instances_per_batch = self.global_batch_size
        # shape: (global num batches, global num instances per batch)
        print(f"indices shape before reshape {indices.shape}")
        indices = indices.reshape(-1, instances_per_batch)
        print(
            f"indices shape after reshape to (global num batches, global num instances per batch) {indices.shape}"
        )

        # Offset by the number of batches already processed.
        if self.batches_processed > 0:
            indices = indices[self.batches_processed :]
        print(f"indices shape after offset {indices.shape}")

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we give worker 0 the first batch, worker 1 the second batch, etc.
            indices = indices[worker_info.id :: worker_info.num_workers]
            print(f"indices shape after slicing by worker rank {indices.shape}")
        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        print(f"indices shape after slicing by local DP rank {indices.shape}")
        return indices

    @property
    def total_size(self) -> int:
        """Total size."""
        return self.total_batches * self.global_batch_size

    @property
    def total_batches(self) -> int:
        """Total batches."""
        return len(self.dataset) // (self.global_batch_size)

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        return torch.utils.data.DataLoader(
            HeliosIterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False,
            timeout=0,
        )


## Config does not yet support our new dataset type so we will construct manually for now
if __name__ == "__main__":
    from helios.data.collator import olmo_compatible_variable_time_collate_fn

    index_path = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples
    workdir = UPath("/Users/henryh/Desktop/eai-repos/helios-repos/helios/workdir")
    dataloader = HeliosDataLoader.wrap_numpy_dataset(
        dataset=HeliosDataset(*samples, dtype=np.dtype("float32")),
        global_batch_size=4,
        collator=olmo_compatible_variable_time_collate_fn,
        work_dir=workdir,
        num_threads=4,
    )
    import time

    # Things needed?
    # Global indices is a file of all the indices in the dataset
    # then the local indices provides a way to get the indices for the current rank
    # - A way to get the global indices which is an array
    # - A way to get the local indices
    # - We want to be able to do muulti threaded and not multithreadd way
    # potentially missing dataset prepare
    for epoch in range(1, 3):
        dataloader.reshuffle(epoch=epoch)
        batch_iterator = dataloader._iter_batches()
        # Need to call reshuffle
        batches_found = 0
        batch_start = time.time()
        for batch in batch_iterator:
            batch_end = time.time()
            if batches_found > 0:
                print(f"batch time {batch_end - batch_start}")
            batches_found += 1
            time.sleep(10)
            batch_start = time.time()
            print("batch found")
        dataloader.reset()

    # need to call reset after the epich
