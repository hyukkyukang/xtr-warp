# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import logging
import os
from functools import cached_property
from typing import *

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from configs import (
    CHUNK_LENGTH,
    INPUT_LENGTH,
    NUM_CHUNKS_PER_ITEM,
    SRC_TOKENIZER_NAME,
    TGT_TOKENIZER_NAME,
)
from warp.evaluation.loaders import load_collection
from warp.infra.run import Run
from warp.tokenizer import call_autotokenizer_with_hf_token

logger = logging.getLogger("Collection")


# Define worker initialization function
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return  # Running in the main process (no workers)

    # Store tokenizers inside the dataset instance (each worker gets its own copy)
    worker_info.dataset.src_tokenizer = call_autotokenizer_with_hf_token(
        SRC_TOKENIZER_NAME
    )
    worker_info.dataset.tgt_tokenizer = AutoTokenizer.from_pretrained(
        TGT_TOKENIZER_NAME
    )


# Define collate function that uses worker's tokenizers
def collate_fn_with_worker_tokenizers(
    batch,
    check_disk_chunk_id: bool = False,
    tokenizers: Tuple[AutoTokenizer, AutoTokenizer] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], int]:
    if check_disk_chunk_id:
        # Check that all items in the batch have the same disk chunk id
        disk_chunk_ids = [item["disk_chunk_id"] for item in batch]
        # TODO: Bug here...
        assert len(set(disk_chunk_ids)) == 1, f"disk_chunk_ids = {disk_chunk_ids}"
        disk_chunk_id = disk_chunk_ids[0]
    else:
        disk_chunk_id = None

    # Get worker info and tokenizers
    worker_info = torch.utils.data.get_worker_info()

    # Use that worker's tokenizers
    if tokenizers is None:
        src_tokenizer = worker_info.dataset.src_tokenizer
        tgt_tokenizer = worker_info.dataset.tgt_tokenizer
    else:
        src_tokenizer, tgt_tokenizer = tokenizers

    # Divide into chunks
    chunks = []
    for batch_item in [item["input_ids"] for item in batch]:
        for i in range(0, len(batch_item), CHUNK_LENGTH):
            chunks.append(batch_item[i : i + CHUNK_LENGTH])

    # Process with tokenizers
    texts = src_tokenizer.batch_decode(chunks, skip_special_tokens=True)
    tgt_outputs = tgt_tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,
    )

    tgt_texts = tgt_outputs["input_ids"]
    attention_mask = tgt_outputs["attention_mask"]
    doc_lens = torch.sum(attention_mask, dim=1).tolist()

    global_indices = [item["global_idx"] for item in batch]

    return {
        "global_indices": global_indices,
        "disk_chunk_id": disk_chunk_id,
        "input_ids": tgt_texts,
        "attention_mask": attention_mask,
        "doc_lens": doc_lens,
    }


class SampledCollection:
    def __init__(
        self,
        dataset: Dataset,
        sample_pids: range,
        rank: int,
        nranks: int,
        input_length: int = INPUT_LENGTH,
    ):
        self.dataset = dataset
        self.input_length = input_length  # Move constants to __init__
        self.rank = rank
        self.nranks = nranks
        # Calculate sample_pids for this rank once during init
        self._sampled_pids = self._make_total_num_items_divisible(sample_pids)

    @cached_property
    def global_sample_pids(self):
        """This is the list of pids that will be sampled from the dataset.
        This is the same for all ranks."""
        return list(self._sampled_pids)

    @cached_property
    def local_sample_pids(self):
        """This is the list of pids that will be sampled from the dataset.
        This is different for each rank."""
        start_idx = self.rank * self.num_item_per_rank
        end_idx = start_idx + self.num_item_per_rank
        return self.global_sample_pids[start_idx:end_idx]

    @cached_property
    def num_item_per_rank(self):
        assert self.get_disk_chunk_size() > self.nranks, (
            f"disk_chunk_size = {self.get_disk_chunk_size()} must be greater than "
            f"nranks = {self.nranks}"
        )
        assert self.get_disk_chunk_size() % self.nranks == 0, (
            f"disk_chunk_size = {self.get_disk_chunk_size()} must be divisible by "
            f"nranks = {self.nranks}"
        )
        assert len(self._sampled_pids) % self.get_disk_chunk_size() == 0, (
            f"len(self._sampled_pids) = {len(self._sampled_pids)} must be divisible by "
            f"get_disk_chunk_size() = {self.get_disk_chunk_size()}"
        )
        # Calculate the number of samples per rank
        num_sample_per_rank = len(self._sampled_pids) // self.nranks
        return num_sample_per_rank

    @property
    def total_num_disk_chunks(self):
        """Disk chunks describes the partition of the data saved in the disk."""
        assert len(self._sampled_pids) % self.get_disk_chunk_size() == 0, (
            f"len(self._sampled_pids) = {len(self._sampled_pids)} must be divisible by "
            f"get_disk_chunk_size() = {self.get_disk_chunk_size()}"
        )
        return len(self._sampled_pids) // self.get_disk_chunk_size()

    @property
    def total_num_chunks(self):
        """Chunks describes text chunk (passage in terms of MSMARCO) that is saved in the memory."""
        num_of_chunks_in_a_pid = self.input_length // CHUNK_LENGTH
        return len(self._sampled_pids) * num_of_chunks_in_a_pid

    def __iter__(self):
        for pid in self.local_sample_pids:
            input_ids = self.dataset[pid]["input_ids"]
            assert (
                len(input_ids) == self.input_length
            ), f"len(input_ids) = {len(input_ids)}"
            # Append global idx and disk chunk id to the dataset
            yield {
                "global_idx": self.get_global_idx(pid) * NUM_CHUNKS_PER_ITEM,
                "disk_chunk_id": self.get_disk_chunk_id(pid),
                "input_ids": input_ids,
            }

    def __getitem__(self, idx: int):
        input_ids = self.dataset[self.local_sample_pids[idx]]["input_ids"]
        assert len(input_ids) == self.input_length, f"len(input_ids) = {len(input_ids)}"
        return {
            "global_idx": self.get_global_idx(idx) * NUM_CHUNKS_PER_ITEM,
            "disk_chunk_id": self.get_disk_chunk_id(idx),
            "input_ids": input_ids,
        }

    def __len__(self) -> int:
        return len(self.local_sample_pids)

    def _make_total_num_items_divisible(self, sampled_pids):
        """Remove the last part of the data so that the total number of items is divisible by
        the multiple of nranks and disk_chunk_size.
        Meanwhile, disk_chunk_size must be divisible by batch_size_to_consider."""
        multiple_of_nranks_and_disk_chunk_size = (
            self.get_disk_chunk_size() * self.nranks
        )
        end_idx = (
            len(sampled_pids) // multiple_of_nranks_and_disk_chunk_size
        ) * multiple_of_nranks_and_disk_chunk_size
        assert (
            end_idx > 0
        ), f"end_idx = {end_idx} must be greater than 0. len(sampled_pids) = {len(sampled_pids)}. multiple_of_nranks_and_disk_chunk_size = {multiple_of_nranks_and_disk_chunk_size}"
        logger.info(
            f"Cutting the last {len(sampled_pids) - end_idx} items. Before: {len(sampled_pids)}, After: {end_idx}"
        )
        return sampled_pids[:end_idx]

    def get_global_idx(self, idx: int) -> int:
        """
        Get the global index of the item at index idx in the local sample pids.
        """
        return self.local_sample_pids[idx]

    def get_disk_chunk_id(self, idx: int) -> int:
        """
        Get the disk chunk id of the item at index idx in the local sample pids.
        """
        disk_chunk_id = self.get_global_idx(idx) // self.get_disk_chunk_size()
        return disk_chunk_id

    def is_valid_batch_size(self, batch_size: int) -> bool:
        """
        Check if the batch size is valid:
        Whether the disk chunk size is divisible by the batch size.
        This is for processing the items in batches.
        """
        return self.get_disk_chunk_size() % batch_size == 0

    def get_disk_chunk_size(self):
        num_of_chunks_in_a_pid = INPUT_LENGTH // CHUNK_LENGTH
        return 25_600 // num_of_chunks_in_a_pid
        # return min(
        #     25_000, 1 + len(self) // Run().nranks
        # )  # 25k is great, 10k allows things to reside on GPU??


class Collection:
    def __init__(self, path=None, data=None):
        self.path = path
        self._data = data  # Make this private
        self._length = None  # Cache for length
        self._iterator = None  # Cache for iterator
        # Initialize tokenizers but don't load data yet
        self.src_tokenizer = call_autotokenizer_with_hf_token("meta-llama/Llama-3.2-1B")
        self.tgt_tokenizer = AutoTokenizer.from_pretrained("google/xtr-base-en")
        self._ensure_data_loaded()

    def _ensure_data_loaded(self):
        """Lazy load data only when needed"""
        if self._data is None and self.path is not None:
            self._data = Dataset.load_from_disk(self.path)

    @property
    def is_dataset(self):
        return isinstance(self._data, Dataset)

    def __iter__(self):
        self._ensure_data_loaded()
        return self._data

    def __getitem__(self, item):
        self._ensure_data_loaded()  # Load data only when accessed
        return self._data[item]["input_ids"]

    def __len__(self):
        self._ensure_data_loaded()
        return len(self._data)

    def _load_file(self, path):
        return self._load_directory(path)

    def _load_directory(self, path):
        return Dataset.load_from_disk(path)

    def _load_tsv(self, path):
        return load_collection(path)

    def _load_jsonl(self, path):
        raise NotImplementedError()

    def provenance(self):
        return self.path

    def toDict(self):
        return {"provenance": self.provenance()}

    def save(self, new_path):
        assert new_path.endswith(".tsv"), "TODO: Support .json[l] too."
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, "w") as f:
            # TODO: expects content to always be a string here; no separate title!
            for pid, content in enumerate(self._data):
                content = f"{pid}\t{content}\n"
                f.write(content)

            return f.name

    def enumerate(self, rank):
        for chunk_id, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (chunk_id, offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_disk_chunk_size()
        data_length = len(self)

        total_ranks = Run().nranks
        number_of_chunks_per_rank = (data_length + total_ranks - 1) // total_ranks
        loop_start_offset = rank * number_of_chunks_per_rank
        chunk_id = 0
        for offset in range(loop_start_offset, data_length):
            loop_end_offset = min(
                loop_start_offset + number_of_chunks_per_rank, data_length
            )
            items = [self[idx] for idx in range(offset, loop_end_offset)]
            yield (chunk_id, offset, items)
            chunk_id += 1
        return

    def enumerate_indices(self, rank):
        for indices in self.enumerate_indices_batches(rank=rank):
            for idx in indices:
                yield idx

    def enumerate_indices_batches(self, rank):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = self.get_disk_chunk_size()
        data_length = len(self)

        total_ranks = Run().nranks
        number_of_data_per_rank = (data_length + total_ranks - 1) // total_ranks
        loop_start_offset = rank * number_of_data_per_rank
        loop_end_offset = min(loop_start_offset + number_of_data_per_rank, data_length)
        # Further divide into chunks of size chunksize
        for chunk_start in range(loop_start_offset, loop_end_offset, chunksize):
            chunk_end = min(chunk_start + chunksize, loop_end_offset)
            yield range(chunk_start, chunk_end)

    def get_disk_chunk_size(self):
        return min(
            25_000, 1 + len(self) // Run().nranks
        )  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.
