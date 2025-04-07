"""Compute the normalization stats for a given dataset."""

import argparse
import json
import logging
import multiprocessing as mp
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from upath import UPath

from helios.data.constants import IMAGE_TILE_SIZE, Modality, MISSING_VALUE
from helios.data.dataset import GetItemArgs, HeliosDataset, HeliosDatasetConfig
from helios.data.utils import update_streaming_stats
from olmo_core.utils import prepare_cli_environment

logger = logging.getLogger(__name__)


def process_sample(args: Tuple[HeliosDataset, int]) -> Dict[str, Any]:
    """Process a single sample to compute its normalization stats.

    Args:
        args: Tuple containing the dataset and the index to process

    Returns:
        Dictionary containing normalization values for this sample
    """
    dataset, i = args
    local_norm_dict: Dict[str, Any] = {}

    try:
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]

        if "timestamps" not in sample.modalities:
            # Returning info as part of the result rather than logging
            return {"error": f"Skipping {i} because it has no timestamps"}

        for modality in sample.modalities:
            if modality == "timestamps" or modality == "latlon":
                continue

            modality_data = sample.as_dict(ignore_nones=True)[modality]
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order

            if modality_data is None:
                continue

            if (modality_data == MISSING_VALUE).all():
                continue

            if modality not in local_norm_dict:
                local_norm_dict[modality] = {}

            for idx, band in enumerate(modality_bands):
                modality_band_data = modality_data[:, :, :, idx]  # (H, W, T, C)

                # For each band, store sum, sum_squared, and count
                if band not in local_norm_dict.get(modality, {}):
                    local_norm_dict.setdefault(modality, {})[band] = {
                        "sum": modality_band_data.sum(),
                        "sum_squared": (modality_band_data**2).sum(),
                        "count": modality_band_data.size,
                    }
                else:
                    local_norm_dict[modality][band]["sum"] += modality_band_data.sum()
                    local_norm_dict[modality][band]["sum_squared"] += (modality_band_data**2).sum()
                    local_norm_dict[modality][band]["count"] += modality_band_data.size

    except Exception as e:
        # Return errors rather than logging them
        return {"error": f"Error processing sample {i}: {str(e)}"}

    return local_norm_dict


def merge_norm_dicts(norm_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple normalization dictionaries.

    Args:
        norm_dicts: List of normalization dictionaries to merge

    Returns:
        Merged normalization dictionary
    """
    merged_dict: Dict[str, Any] = {}

    for norm_dict in norm_dicts:
        for modality in norm_dict:
            if modality not in merged_dict:
                merged_dict[modality] = {}

            for band in norm_dict[modality]:
                if band not in merged_dict[modality]:
                    merged_dict[modality][band] = {
                        "sum": norm_dict[modality][band]["sum"],
                        "sum_squared": norm_dict[modality][band]["sum_squared"],
                        "count": norm_dict[modality][band]["count"],
                    }
                else:
                    merged_dict[modality][band]["sum"] += norm_dict[modality][band]["sum"]
                    merged_dict[modality][band]["sum_squared"] += norm_dict[modality][band]["sum_squared"]
                    merged_dict[modality][band]["count"] += norm_dict[modality][band]["count"]

    # Calculate mean, variance, and std from the merged sums
    for modality in merged_dict:
        logger.info(f"Processing modality: {modality}")
        for band in merged_dict[modality]:
            logger.info(f"Processing band: {band}")
            count = merged_dict[modality][band]["count"]
            sum_val = merged_dict[modality][band]["sum"]
            sum_squared = merged_dict[modality][band]["sum_squared"]
            logger.info(f"Sum: {sum_val}, Sum squared: {sum_squared}, Count: {count}")

            mean = sum_val / count
            # Var = E[X²] - E[X]²
            var = (sum_squared / count) - (mean**2)

            merged_dict[modality][band]["mean"] = float(mean)
            merged_dict[modality][band]["var"] = float(var)
            merged_dict[modality][band]["std"] = float(np.sqrt(var))
            merged_dict[modality][band]["count"] = int(count)

            # Clean up intermediate values
            del merged_dict[modality][band]["sum"]
            del merged_dict[modality][band]["sum_squared"]

    return merged_dict


def compute_normalization_values(
    dataset: HeliosDataset,
    estimate_from: int | None = None,
    num_workers: int = mp.cpu_count(),
) -> dict[str, Any]:
    """Compute the normalization values for the dataset using parallelism.

    Args:
        dataset: The dataset to compute the normalization values for.
        estimate_from: The number of samples to estimate the normalization values from.
        num_workers: Number of worker processes to use.

    Returns:
        dict: A dictionary containing the normalization values for the dataset.
    """
    dataset_len = len(dataset)
    if estimate_from is not None:
        indices_to_sample = random.sample(list(range(dataset_len)), k=estimate_from)
    else:
        indices_to_sample = list(range(dataset_len))

    # Process samples in parallel
    logger.info(f"Processing {len(indices_to_sample)} samples with {num_workers} workers")
    with mp.Pool(num_workers) as pool:
        args_list = [(dataset, i) for i in indices_to_sample]
        results = []
        errors = []

        for result in tqdm(
            pool.imap(process_sample, args_list),
            total=len(indices_to_sample),
            desc="Computing normalization stats"
        ):
            if result is None:
                continue
            elif "error" in result:
                errors.append(result["error"])
                # Log errors from the worker processes
                logger.info(result["error"])
            else:
                results.append(result)

        # Log a summary of errors
        if errors:
            logger.info(f"Encountered {len(errors)} errors during processing")

    # Merge results from all processes (filter out None results and error messages)
    valid_results = [r for r in results if r and "error" not in r]
    if not valid_results:
        raise ValueError("No valid samples were processed, cannot compute normalization")
    logger.info(f"Merging {len(valid_results)} valid results")
    norm_dict = merge_norm_dicts(valid_results)

    norm_dict["total_n"] = dataset_len
    norm_dict["sampled_n"] = len(indices_to_sample)
    norm_dict["tile_path"] = str(dataset.tile_path)

    return norm_dict

if __name__ == "__main__":
    prepare_cli_environment()
    args = argparse.ArgumentParser()

    args.add_argument("--tile_path", type=str, required=True)
    args.add_argument("--supported_modalities", type=str, required=True)
    args.add_argument("--estimate_from", type=int, required=False, default=None)
    args.add_argument("--output_path", type=str, required=True)
    args_dict = args.parse_args().__dict__  # type: ignore

    logger.info(
        f"Computing normalization stats for {args_dict['tile_path']} with modalities {args_dict['supported_modalities']}"
    )


    def parse_supported_modalities(supported_modalities: str) -> list[str]:
        """Parse the supported modalities from a string."""
        return supported_modalities.split(",")


    # Use the config to build the dataset
    dataset_config = HeliosDatasetConfig(
        # tile_path=UPath(args_dict["tile_path"]),
        h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data/landsat_naip_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/102695",
        supported_modality_names=parse_supported_modalities(
            args_dict["supported_modalities"]
        ),
        normalize=False,
    )
    dataset = dataset_config.build()
    dataset.prepare()
    logger.info(f"Dataset: {dataset.normalize}")

    norm_dict = compute_normalization_values(
        dataset=dataset,
        estimate_from=args_dict["estimate_from"],
    )
    logger.info(f"Normalization stats: {norm_dict}")

    with open(args_dict["output_path"], "w") as f:
        json.dump(norm_dict, f)


    # Example usage:
    # 20250304 run:
    # python3 compute_norm.py --tile_path "/weka/dfive-default/helios/dataset/presto" --supported_modalities "sentinel2_l2a,sentinel1,worldcover" --output_path "/weka/dfive-default/yawenz/helios/data/norm_configs/computed_20250304.json"
