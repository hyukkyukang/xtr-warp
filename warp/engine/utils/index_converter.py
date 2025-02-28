import gc
import json
import os
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from warp.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN


def segmented_index_cumsum(input_tensor, offsets):
    values, indices = input_tensor.sort(stable=True)
    unique_values, inverse_indices, counts_values = torch.unique(
        values, return_inverse=True, return_counts=True
    )
    offset_arange = torch.arange(1, len(unique_values) + 1)
    offset_count_indices = offset_arange[inverse_indices]

    offset_counts = torch.zeros(counts_values.shape[0] + 1, dtype=torch.long)
    offset_counts[1:] = torch.cumsum(counts_values, dim=0)

    counts = torch.zeros_like(input_tensor, dtype=torch.long)
    counts[indices] = (
        torch.arange(0, input_tensor.shape[0]) - offset_counts[offset_count_indices - 1]
    )

    return counts + offsets[input_tensor.long()], offsets + torch.bincount(
        input_tensor, minlength=offsets.shape[0]
    )


def convert_index(index_path, destination_path=None):
    if destination_path is None:
        destination_path = index_path
    print(f"Compacting index at '{index_path}' to '{destination_path}'")
    os.makedirs(destination_path, exist_ok=True)
    with open(os.path.join(index_path, "plan.json"), "r") as file:
        plan = json.load(file)

    config = plan["config"]

    checkpoint = config["checkpoint"]
    assert checkpoint == "google/xtr-base-en"

    dim = config["dim"]
    nbits = config["nbits"]

    query_maxlen = config["query_maxlen"]
    doc_maxlen = config["doc_maxlen"]

    assert query_maxlen == QUERY_MAXLEN
    assert doc_maxlen == DOC_MAXLEN

    num_chunks = plan["num_chunks"]
    num_partitions = plan["num_partitions"]  # i.e., num_centroids

    centroids = torch.load(os.path.join(index_path, "centroids.pt"), map_location="cpu")
    assert centroids.shape == (num_partitions, dim)

    # TODO(jlscheerer) Perhaps do this per centroid instead of globally.
    bucket_cutoffs, bucket_weights = torch.load(
        os.path.join(index_path, "buckets.pt"), map_location="cpu"
    )

    np.save(
        os.path.join(destination_path, "bucket_cutoffs.npy"),
        bucket_cutoffs.float().numpy(force=True),
    )
    np.save(
        os.path.join(destination_path, "bucket_weights.npy"),
        bucket_weights.float().numpy(force=True),
    )

    print("#> centroids.dtype =", centroids.dtype)

    centroids = centroids.float()
    np.save(
        os.path.join(destination_path, "centroids.npy"),
        centroids.numpy(force=True).astype(np.float32),
    )

    ivf, ivf_lengths = torch.load(os.path.join(index_path, "ivf.pid.pt"))
    assert ivf_lengths.shape == (num_partitions,)
    assert ivf.shape == (ivf_lengths.sum(),)

    print("> Loading centroid information")
    centroid_sizes = torch.zeros((num_partitions,), dtype=torch.int64)
    for chunk in tqdm(range(num_chunks)):
        # NOTE codes describe the corresponding centroid for each embedding
        codes = torch.load(os.path.join(index_path, f"{chunk}.codes.pt"))
        # residuals = torch.load(os.path.join(index_path, f"{chunk}.residuals.pt"))
        centroid_sizes += torch.bincount(codes, minlength=num_partitions)
    num_residuals = centroid_sizes.sum().item()

    offsets = torch.zeros((num_partitions,), dtype=torch.int64)
    offsets[1:] = torch.cumsum(centroid_sizes[:-1], dim=0)

    residual_dim = (dim * nbits) // 8  # residuals are stored as uint8

    tensor_offsets = torch.zeros((num_partitions,), dtype=torch.int64)
    tensor_offsets[1:] = torch.cumsum(centroid_sizes[:-1], dim=0)

    tensor_compacted_residuals = torch.zeros(
        (num_residuals, residual_dim), dtype=torch.uint8
    )
    tensor_compacted_codes = torch.zeros((num_residuals,), dtype=torch.int32)

    print("> Compacting index")
    passage_id = 0
    for chunk in tqdm(range(num_chunks)):
        with open(os.path.join(index_path, f"doclens.{chunk}.json"), "r") as file:
            doclens = json.load(file)
        codes = torch.load(os.path.join(index_path, f"{chunk}.codes.pt"))
        residuals = torch.load(os.path.join(index_path, f"{chunk}.residuals.pt"))

        doclens = torch.tensor(doclens)
        assert doclens.sum() == residuals.shape[0]

        passage_ids = (
            torch.repeat_interleave(torch.arange(doclens.shape[0]), doclens).int()
            + passage_id
        )

        tensor_idx, tensor_offsets = segmented_index_cumsum(codes, tensor_offsets)

        tensor_compacted_residuals[tensor_idx] = residuals
        tensor_compacted_codes[tensor_idx] = passage_ids

        passage_id += doclens.shape[0]

    print("> Saving compacted index")
    torch.save(
        centroid_sizes,
        os.path.join(destination_path, "sizes.compacted.pt"),
    )
    torch.save(
        tensor_compacted_residuals,
        os.path.join(destination_path, "residuals.compacted.pt"),
    )
    torch.save(
        tensor_compacted_codes,
        os.path.join(destination_path, "codes.compacted.pt"),
    )

    print("> Repacking residuals")

    # Move tensors to CPU for processing if they're on GPU
    tensor_compacted_residuals_cpu = tensor_compacted_residuals.cpu()

    # Process in smaller batches to reduce peak memory usage
    def convert_in_batches(tensor_compacted_residuals, bucket_weights, nbits):
        batch_size = 10_000_000  # Adjust based on your available memory
        num_batches = (
            tensor_compacted_residuals.shape[0] + batch_size - 1
        ) // batch_size

        # Pre-compute lookup tables more efficiently
        keys_per_byte = 8 // nbits
        num_weights = len(bucket_weights)

        # Generate lookup table more efficiently
        if keys_per_byte <= 4:  # For nbits=2 or nbits=4
            # Generate indices directly without using product()
            indices = torch.arange(num_weights**keys_per_byte, dtype=torch.long)
            decompression_lookup_table = torch.zeros(
                (num_weights**keys_per_byte, keys_per_byte), dtype=torch.uint8
            )

            for i in range(keys_per_byte):
                divisor = num_weights ** (keys_per_byte - i - 1)
                decompression_lookup_table[:, i] = (indices // divisor) % num_weights
        else:
            # Fall back to original method for other cases
            decompression_lookup_table = torch.tensor(
                list(product(list(range(num_weights)), repeat=keys_per_byte))
            ).to(torch.uint8)

        # Process in batches
        results = []
        for i in tqdm(
            range(num_batches),
            desc="Repacking residuals",
            total=num_batches,
        ):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, tensor_compacted_residuals.shape[0])

            # Process this batch
            batch = tensor_compacted_residuals[start_idx:end_idx]

            # Apply the operations to the batch
            residuals_repacked_batch = decompression_lookup_table[batch.long()]

            # Apply the final transformation based on nbits
            if nbits == 4:
                residuals_repacked_batch_df = (
                    2**4 * residuals_repacked_batch[:, :, 0]
                    + residuals_repacked_batch[:, :, 1]
                )
            elif nbits == 2:
                residuals_repacked_batch_df = (
                    2**6 * residuals_repacked_batch[:, :, 0]
                    + 2**4 * residuals_repacked_batch[:, :, 1]
                    + 2**2 * residuals_repacked_batch[:, :, 2]
                    + residuals_repacked_batch[:, :, 3]
                )
            else:
                assert False

            # Store the result for this batch
            results.append(residuals_repacked_batch_df)

            # Force cleanup
            del residuals_repacked_batch

            gc.collect()

        # Combine the results
        return torch.cat(results, dim=0)

    # Replace the original code with a call to the batched version
    residuals_repacked_compacted_df = convert_in_batches(
        tensor_compacted_residuals_cpu, bucket_weights, nbits
    )

    # Move back to GPU only if needed for subsequent operations
    print(
        f"> Moving repacked residuals from {residuals_repacked_compacted_df.device} to {tensor_compacted_residuals.device}"
    )
    residuals_repacked_compacted_df = residuals_repacked_compacted_df.to(
        tensor_compacted_residuals.device
    )
    repacked_path = os.path.join(destination_path, "residuals.repacked.compacted.pt")
    print(f"> Saving repacked residuals to {repacked_path}")
    torch.save(residuals_repacked_compacted_df, repacked_path)
    print("> Done!")
