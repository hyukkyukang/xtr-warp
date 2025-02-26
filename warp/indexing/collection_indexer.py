import gc
import logging
import os
import random

import torch
import tqdm
import ujson

try:
    import faiss
except ImportError as e:
    print("WARNING: faiss must be imported for indexing")
from functools import partial
from typing import *

import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import warp.utils.distributed as distributed
from configs import (
    CHUNK_LENGTH,
    COLLECTION_SAMPLE_RATE,
    INPUT_LENGTH,
    KMEANS_SAMPLE_RATE,
)
from warp.data.collection import (
    Collection,
    SampledCollection,
    collate_fn_with_worker_tokenizers,
    worker_init_fn,
)
from warp.indexing.codecs.residual import ResidualCodec
from warp.indexing.collection_encoder import CollectionEncoder
from warp.indexing.index_saver import IndexSaver
from warp.indexing.utils import optimize_ivf
from warp.infra.config.config import ColBERTConfig
from warp.infra.launcher import print_memory_stats
from warp.infra.run import Run
from warp.modeling.checkpoint import Checkpoint
from warp.utils.utils import print_message

logger = logging.getLogger("CollectionIndexer")


def encode(config, collection, shared_lists, shared_queues, verbose: int = 3):
    encoder = CollectionIndexer(config=config, collection=collection, verbose=verbose)
    encoder.run(shared_lists)


class CollectionIndexer:
    """
    Given a collection and config, encode collection into index and
    stores the index on the disk in chunks.
    """

    def __init__(self, config: ColBERTConfig, collection, verbose=2):
        self.verbose = verbose
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        self.use_gpu = self.config.total_visible_gpus > 0

        if self.config.rank == 0 and self.verbose > 1:
            self.config.help()

        self.collection = Collection.cast(collection)
        self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config)
        if self.use_gpu:
            self.checkpoint = self.checkpoint.cuda()

        self.encoder = CollectionEncoder(config, self.checkpoint)
        self.saver = IndexSaver(config)

        # Create SampledCollection
        self.sampled_collection = self._create_sampled_collection()

        print_memory_stats(f"RANK:{self.rank}")

    def _create_sampled_collection(self) -> SampledCollection:
        sampled_collection = SampledCollection(
            dataset=self.collection._data,
            sample_pids=range(
                int(len(self.collection._data) * COLLECTION_SAMPLE_RATE)
            ),  # Use all pids
            rank=self.rank,
            nranks=self.nranks,
            batch_size_to_consider=self.encoder.batch_size,
        )
        assert sampled_collection.is_valid_batch_size(
            self.encoder.batch_size
        ), f"Batch size {self.encoder.batch_size} is not valid for the sampled collection. disk chunk size = {sampled_collection.get_disk_chunk_size()}"

        return sampled_collection

    def run(self, shared_lists):
        with torch.inference_mode():
            self.setup()  # Computes and saves plan for whole collection
            distributed.barrier(self.rank)
            print_memory_stats(f"RANK:{self.rank}")

            if not self.config.resume or not self.saver.try_load_codec():
                self.train(shared_lists)  # Trains centroids from selected passages
            distributed.barrier(self.rank)
            print_memory_stats(f"RANK:{self.rank}")

            self.index()  # Encodes and saves all tokens into residuals
            distributed.barrier(self.rank)
            print_memory_stats(f"RANK:{self.rank}")

            self.finalize()  # Builds metadata and centroid to passage mapping
            distributed.barrier(self.rank)
            print_memory_stats(f"RANK:{self.rank}")

    def setup(self):
        """
        Calculates and saves plan.json for the whole collection.

        plan.json { config, num_chunks, num_partitions, num_embeddings_est, avg_doclen_est}
        num_partitions is the number of centroids to be generated.
        """
        self.num_disk_chunks = self.sampled_collection.total_num_disk_chunks

        # Select the number of partitions
        num_passages: int = len(self.sampled_collection.global_sample_pids)
        num_chunks_per_pid: int = INPUT_LENGTH // CHUNK_LENGTH
        num_chunks: int = num_passages * num_chunks_per_pid
        self.num_embeddings_est: int = num_chunks * CHUNK_LENGTH
        self.num_partitions = int(
            int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))
        )
        self.avg_doclen_est = CHUNK_LENGTH

        if self.verbose > 0:
            Run().print_main(f"Creating {self.num_partitions:,} partitions.")
            Run().print_main(
                f"*Estimated* {int(self.num_embeddings_est):,} embeddings."
            )
        # If resume, try to load the plan and check if it's valid with the current config.
        # Otherwise, we can not resume
        if self.config.resume:
            loaded_plan: Dict[str, Any] = self._load_plan()
            # Check if the plan is valid
            assert (
                loaded_plan["num_disk_chunks"] == self.num_disk_chunks
            ), f"num_disk_chunks = {loaded_plan['num_disk_chunks']} != {self.num_disk_chunks}"
            assert (
                loaded_plan["num_partitions"] == self.num_partitions
            ), f"num_partitions = {loaded_plan['num_partitions']} != {self.num_partitions}"
            assert (
                loaded_plan["num_embeddings_est"] == self.num_embeddings_est
            ), f"num_embeddings_est = {loaded_plan['num_embeddings_est']} != {self.num_embeddings_est}"
            assert (
                loaded_plan["avg_doclen_est"] == self.avg_doclen_est
            ), f"avg_doclen_est = {loaded_plan['avg_doclen_est']} != {self.avg_doclen_est}"
            assert (
                loaded_plan["num_global_sample_pids"] == len(
                    self.sampled_collection.global_sample_pids
                )
            ), f"num_global_sample_pids = {loaded_plan['num_global_sample_pids']} != {len(self.sampled_collection.global_sample_pids)}"

        else:
            # Saves sampled passages and embeddings for training k-means centroids later
            sampled_pids: List[int] = self._sample_pids()
            self._sample_embeddings(sampled_pids)

            self._save_plan()

    def _sample_pids(self):
        num_chunks = self.sampled_collection.total_num_chunks
        num_passages = len(self.sampled_collection.global_sample_pids)
        num_chunks_per_pid = INPUT_LENGTH // CHUNK_LENGTH

        # Simple alternative: < 100k: 100%, < 1M: 15%, < 10M: 7%, < 100M: 3%, > 100M: 1%
        # Keep in mind that, say, 15% still means at least 100k.
        # So the formula is max(100% * min(total, 100k), 15% * min(total, 1M), ...)
        # Then we subsample the vectors to 100 * num_partitions

        typical_doclen = CHUNK_LENGTH
        sample_num: int = int(16 * np.sqrt(typical_doclen * num_chunks))
        # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
        # Convert to number of passages required to sample
        sample_num = int(sample_num / num_chunks_per_pid)
        # Subsample the passages (Due to computational cost)
        sample_num = int(sample_num * KMEANS_SAMPLE_RATE)
        sample_num = min(1 + sample_num, num_passages)
        # The sample number should be bigger than disk chunk size
        sample_num = max(sample_num, self.sampled_collection.get_disk_chunk_size())
        assert sample_num >= self.sampled_collection.get_disk_chunk_size(), (
            f"num_passages = {sample_num} must be greater than or equal to "
            f"disk_chunk_size = {self.sampled_collection.get_disk_chunk_size()}"
        )

        logger.info(f"num_passages = {num_passages}, sample_num = {sample_num}")
        sampled_pids = random.sample(range(num_passages), sample_num)
        if self.verbose > 1:
            Run().print_main(
                f"# of sampled PIDs = {len(sampled_pids)} \t sampled_pids[:3] = {sampled_pids[:3]}"
            )

        return list(sampled_pids)

    def _sample_embeddings(self, sampled_pids: List[int]):
        # Create a sampled collection dataset
        sampled_collection = SampledCollection(
            dataset=self.collection._data,
            sample_pids=sampled_pids,
            rank=self.rank,
            nranks=self.nranks,
        )

        # Create dataloader with worker initialization
        collection_dataloader = DataLoader(
            sampled_collection,
            batch_size=self.encoder.batch_size,
            num_workers=self.encoder.num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_with_worker_tokenizers,
        )

        # Process batches
        all_embs = []
        all_doclens = []
        for batch in tqdm.tqdm(
            collection_dataloader,
            disable=self.rank > 0,
            total=len(collection_dataloader),
            desc="Encoding samples",
        ):
            # Get batch data
            token_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            doclens = batch["doc_lens"]
            # Encode
            local_sample_embs = self.encoder.encode_from_token_ids(
                token_ids, attention_mask
            )
            # Append
            all_embs.append(local_sample_embs)
            all_doclens.extend(doclens)

        local_sample_embs = torch.cat(all_embs, dim=0)

        if torch.cuda.is_available():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
                torch.distributed.all_reduce(self.num_sample_embs)
            else:
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
        else:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cpu()
                torch.distributed.all_reduce(self.num_sample_embs)
            else:
                self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cpu()

        logger.info(
            f"avg_doclen_est = {self.avg_doclen_est} \t len(local_sample) = {len(local_sample_embs):,}"
        )
        path = os.path.join(self.config.index_path_, f"sample.{self.rank}.pt")
        torch.save(
            local_sample_embs,
            path,
        )
        # Free GPU memory
        torch.cuda.empty_cache()

        return None

    def _load_plan(self):
        config = self.config
        self.plan_path = os.path.join(config.index_path_, "plan.json")
        assert os.path.exists(
            self.plan_path
        ), f"plan.json does not exist at {self.plan_path}"

        with open(self.plan_path, "r") as f:
            plan = ujson.load(f)

        return {
            "num_disk_chunks": plan["num_chunks"],
            "num_partitions": plan["num_partitions"],
            "num_embeddings_est": plan["num_embeddings_est"],
            "avg_doclen_est": plan["avg_doclen_est"],
        }

    def _save_plan(self):
        if self.rank < 1:
            config = self.config
            self.plan_path = os.path.join(config.index_path_, "plan.json")
            logger.info("#> Saving the indexing plan to", self.plan_path, "..")

            with open(self.plan_path, "w") as f:
                d = {"config": config.export()}
                d["num_chunks"] = self.num_disk_chunks
                d["num_partitions"] = self.num_partitions
                d["num_embeddings_est"] = self.num_embeddings_est
                d["avg_doclen_est"] = self.avg_doclen_est
                d["num_global_sample_pids"] = len(
                    self.sampled_collection.global_sample_pids
                )

                f.write(ujson.dumps(d, indent=4) + "\n")

    def train(self, shared_lists):
        if self.rank > 0:
            return

        sample, heldout = self._concatenate_and_split_sample()

        centroids = self._train_kmeans(sample, shared_lists)

        print_memory_stats(f"RANK:{self.rank}")
        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(
            centroids, heldout
        )

        if self.verbose > 1:
            print_message(f"avg_residual = {avg_residual}")

        # Compute and save codec into avg_residual.pt, buckets.pt and centroids.pt
        codec = ResidualCodec(
            config=self.config,
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )
        self.saver.save_codec(codec)

    def _concatenate_and_split_sample(self):
        print_memory_stats(f"***1*** \t RANK:{self.rank}")

        # TODO: Allocate a float16 array. Load the samples from disk, copy to array.
        sample = torch.empty(self.num_sample_embs, self.config.dim, dtype=torch.float32)

        offset = 0
        for r in range(self.nranks):
            sub_sample_path = os.path.join(self.config.index_path_, f"sample.{r}.pt")
            sub_sample = torch.load(sub_sample_path)
            os.remove(sub_sample_path)

            endpos = offset + sub_sample.size(0)
            sample[offset:endpos] = sub_sample
            offset = endpos

        assert endpos == sample.size(0), (endpos, sample.size())

        print_memory_stats(f"***2*** \t RANK:{self.rank}")

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample = sample[torch.randperm(sample.size(0))]

        print_memory_stats(f"***3*** \t RANK:{self.rank}")

        heldout_fraction = 0.05
        heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
        sample, sample_heldout = sample.split(
            [sample.size(0) - heldout_size, heldout_size], dim=0
        )

        print_memory_stats(f"***4*** \t RANK:{self.rank}")

        return sample, sample_heldout

    def _train_kmeans(self, sample, shared_lists):
        if self.use_gpu:
            torch.cuda.empty_cache()

        do_fork_for_faiss = False  # set to True to free faiss GPU-0 memory at the cost of one more copy of `sample`.

        args_ = [self.config.dim, self.num_partitions, self.config.kmeans_niters]

        if do_fork_for_faiss:
            # For this to work reliably, write the sample to disk. Pickle may not handle >4GB of data.
            # Delete the sample file after work is done.

            shared_lists[0][0] = sample
            return_value_queue = mp.Queue()

            args_ = args_ + [shared_lists, return_value_queue]
            proc = mp.Process(target=compute_faiss_kmeans, args=args_)

            proc.start()
            centroids = return_value_queue.get()
            proc.join()

        else:
            args_ = args_ + [[[sample]]]
            centroids = compute_faiss_kmeans(*args_)

        centroids = torch.nn.functional.normalize(centroids, dim=-1)
        if self.use_gpu:
            centroids = centroids.float()
        else:
            centroids = centroids.float()

        return centroids

    def _compute_avg_residual(self, centroids, heldout):
        compressor = ResidualCodec(
            config=self.config, centroids=centroids, avg_residual=None
        )

        heldout_reconstruct = compressor.compress_into_codes(
            heldout, out_device="cuda" if self.use_gpu else "cpu"
        )
        heldout_reconstruct = compressor.lookup_centroids(
            heldout_reconstruct, out_device="cuda" if self.use_gpu else "cpu"
        )
        if self.use_gpu:
            heldout_avg_residual = heldout.cuda() - heldout_reconstruct
        else:
            heldout_avg_residual = heldout - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        logger.info([round(x, 3) for x in avg_residual.squeeze().tolist()])

        num_options = 2**self.config.nbits
        quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (
            1 / num_options
        )
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[
            1:
        ], quantiles + (0.5 / num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        if self.verbose > 2:
            print_message(
                f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}"
            )
            print_message(
                f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}"
            )

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

        # EVENTAULLY: Compare the above with non-heldout sample. If too different, we can do better!
        # sample = sample[subsample_idxs]
        # sample_reconstruct = get_centroids_for(centroids, sample)
        # sample_avg_residual = (sample - sample_reconstruct).mean(dim=0)

    def index(self):
        """
        Encode embeddings for all passages in collection.
        Each embedding is converted to code (centroid id) and residual.
        Embeddings stored according to passage order in contiguous chunks of memory.

        Saved data files described below:
            {CHUNK#}.codes.pt:      centroid id for each embedding in chunk
            {CHUNK#}.residuals.pt:  16-bits residual for each embedding in chunk
            doclens.{CHUNK#}.pt:    number of embeddings within each passage in chunk
        """
        collate_fn = partial(
            collate_fn_with_worker_tokenizers,
            check_disk_chunk_id=True,
            tokenizers=[
                self.collection.src_tokenizer,
                self.collection.tgt_tokenizer,
            ],
        )

        # Create dataloader with worker initialization
        collection_dataloader = DataLoader(
            self.sampled_collection,
            batch_size=self.encoder.batch_size,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

        embs_buffer: List[torch.Tensor] = []
        doclens_buffer: List[int] = []
        disk_chunk_ids_buffer: Optional[int] = None
        offset: int | None = None
        seen_global_indices: int = -1
        with self.saver.thread():
            for batch in tqdm.tqdm(
                collection_dataloader,
                disable=self.rank > 0,
                total=len(collection_dataloader),
                desc="Encoding text chunks",
            ):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                doclens = batch["doc_lens"]
                disk_chunk_id = batch["disk_chunk_id"]
                global_indices = batch["global_indices"]
                # Check for debugging purposes
                assert seen_global_indices < min(global_indices), (
                    seen_global_indices,
                    min(global_indices),
                )
                # Update the seen global indices to the latest global index
                seen_global_indices = max(global_indices)
                # Path to the metadata file
                metadata_path = os.path.join(
                    self.config.index_path_, f"{disk_chunk_id}.metadata.json"
                )

                # Check if the data is already saved in the disk
                if self.config.resume and os.path.exists(metadata_path):
                    continue

                # Save the buffer if the disk_chunk_id has changed
                if (
                    disk_chunk_ids_buffer is not None
                    and disk_chunk_id != disk_chunk_ids_buffer
                ):
                    self.saver.save_chunk(
                        disk_chunk_ids_buffer,
                        offset,
                        torch.cat(embs_buffer, dim=0),
                        doclens_buffer,
                    )
                    embs_buffer = []
                    doclens_buffer = []
                    disk_chunk_ids_buffer = None
                    offset = None
                    # Free memory
                    gc.collect()
                    torch.cuda.empty_cache()

                # Conduct the encoding
                embs = self.encoder.encode_from_token_ids(input_ids, attention_mask)
                assert embs.dtype == torch.float32
                # Save to the buffer
                embs_buffer.append(embs)
                doclens_buffer.extend(doclens)
                disk_chunk_ids_buffer = disk_chunk_id
                # Set the offset if it's not set
                if offset is None:
                    offset = global_indices[0]

            # Save the remaining buffer
            if disk_chunk_ids_buffer is not None:
                self.saver.save_chunk(
                    disk_chunk_ids_buffer,
                    offset,
                    torch.cat(embs_buffer, dim=0),
                    doclens_buffer,
                )

    def finalize(self):
        """
        Aggregates and stores metadata for each chunk and the whole index
        Builds and saves inverse mapping from centroids to passage IDs

        Saved data files described below:
            {CHUNK#}.metadata.json: [ passage_offset, num_passages, num_embeddings, embedding_offset ]
            metadata.json: [ num_chunks, num_partitions, num_embeddings, avg_doclen ]
            inv.pid.pt: [ ivf, ivf_lengths ]
                ivf is an array of passage IDs for centroids 0, 1, ...
                ivf_length contains the number of passage IDs for each centroid
        """
        if self.rank > 0:
            return

        self._check_all_files_are_saved()
        self._collect_embedding_id_offset()

        self._build_ivf()
        self._update_metadata()

    def _check_all_files_are_saved(self):
        if self.verbose > 1:
            Run().print_main("#> Checking all files were saved...")
        success = True
        for chunk_idx in range(self.num_disk_chunks):
            if not self.saver.check_chunk_exists(chunk_idx):
                success = False
                Run().print_main(f"#> ERROR: Could not find chunk {chunk_idx}!")
                # TODO: Fail here?
        if success:
            if self.verbose > 1:
                Run().print_main("Found all files!")

    def _collect_embedding_id_offset(self):
        passage_offset = 0
        embedding_offset = 0

        self.embedding_offsets = []

        for chunk_idx in range(self.num_disk_chunks):
            metadata_path = os.path.join(
                self.config.index_path_, f"{chunk_idx}.metadata.json"
            )

            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)

                chunk_metadata["embedding_offset"] = embedding_offset
                self.embedding_offsets.append(embedding_offset)

                assert chunk_metadata["passage_offset"] == passage_offset, (
                    chunk_idx,
                    passage_offset,
                    chunk_metadata,
                )

                passage_offset += chunk_metadata["num_passages"]
                embedding_offset += chunk_metadata["num_embeddings"]

            with open(metadata_path, "w") as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + "\n")

        self.num_embeddings = embedding_offset
        assert len(self.embedding_offsets) == self.num_disk_chunks

    def _build_ivf(self):
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.

        if self.verbose > 1:
            Run().print_main("#> Building IVF...")

        codes = torch.zeros(
            self.num_embeddings,
        ).long()
        if self.verbose > 1:
            print_memory_stats(f"RANK:{self.rank}")

        if self.verbose > 1:
            Run().print_main("#> Loading codes...")

        for chunk_idx in tqdm.tqdm(range(self.num_disk_chunks)):
            offset = self.embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(
                self.config.index_path_, chunk_idx
            )

            codes[offset : offset + chunk_codes.size(0)] = chunk_codes

        assert offset + chunk_codes.size(0) == codes.size(0), (
            offset,
            chunk_codes.size(0),
            codes.size(),
        )
        if self.verbose > 1:
            Run().print_main(f"Sorting codes...")

            print_memory_stats(f"RANK:{self.rank}")

        codes = codes.sort()
        ivf, values = codes.indices, codes.values

        if self.verbose > 1:
            print_memory_stats(f"RANK:{self.rank}")

            Run().print_main(f"Getting unique codes...")

        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions

        if self.verbose > 1:
            print_memory_stats(f"RANK:{self.rank}")

        # Transforms centroid->embedding ivf to centroid->passage ivf
        _, _ = optimize_ivf(ivf, ivf_lengths, self.config.index_path_)

    def _update_metadata(self):
        config = self.config
        self.metadata_path = os.path.join(config.index_path_, "metadata.json")
        if self.verbose > 1:
            logger.info("#> Saving the indexing metadata to", self.metadata_path, "..")

        with open(self.metadata_path, "w") as f:
            d = {"config": config.export()}
            d["num_chunks"] = self.num_disk_chunks
            d["num_partitions"] = self.num_partitions
            d["num_embeddings"] = self.num_embeddings
            d["avg_doclen"] = self.num_embeddings / len(self.collection)

            f.write(ujson.dumps(d, indent=4) + "\n")


def compute_faiss_kmeans(
    dim, num_partitions, kmeans_niters, shared_lists, return_value_queue=None
):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    use_gpu = torch.cuda.is_available()
    Run().print_main("#> use_gpu =", use_gpu)
    kmeans = faiss.Kmeans(
        dim,
        num_partitions,
        niter=kmeans_niters,
        spherical=True,
        gpu=use_gpu,
        verbose=True,
        seed=123,
    )

    sample = shared_lists[0][0]
    sample = sample.float().numpy()

    kmeans.train(sample)

    centroids = torch.from_numpy(kmeans.centroids)

    print_memory_stats(f"RANK:0*")

    if return_value_queue is not None:
        return_value_queue.put(centroids)

    return centroids


"""
TODOs:

1. Consider saving/using heldout_avg_residual as a vector --- that is, using 128 averages!

2. Consider the operations with .cuda() tensors. Are all of them good for OOM?
"""
