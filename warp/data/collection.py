# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
from typing import *

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from warp.evaluation.loaders import load_collection
from warp.infra.run import Run
from warp.tokenizer import call_autotokenizer_with_hf_token
from configs import SRC_TOKENIZER_NAME, TGT_TOKENIZER_NAME, CHUNK_LENGTH, INPUT_LENGTH


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
def collate_fn_with_worker_tokenizers(batch):
    # Get worker info and tokenizers
    worker_info = torch.utils.data.get_worker_info()

    # Use that worker's tokenizers
    src_tokenizer = worker_info.dataset.src_tokenizer
    tgt_tokenizer = worker_info.dataset.tgt_tokenizer

    # Process batch using appropriate tokenizers
    input_ids = batch  # Assuming batch is already a list of input_ids

    # Divide into chunks
    chunks = []
    for batch_item in input_ids:
        for i in range(0, len(batch_item), CHUNK_LENGTH):
            chunks.append(batch_item[i : i + CHUNK_LENGTH])

    # Process with tokenizers
    texts = src_tokenizer.batch_decode(chunks, skip_special_tokens=True)
    tgt_outputs = tgt_tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    tgt_texts = tgt_outputs["input_ids"]
    attention_mask = tgt_outputs["attention_mask"]
    doc_lens = torch.sum(attention_mask, dim=1).tolist()

    return tgt_texts, attention_mask, doc_lens


class SampledCollection:
    def __init__(
        self,
        dataset: Dataset,
        sample_pids: List[int],
        rank: int,
        nranks: int,
        input_length: int = INPUT_LENGTH,
    ):
        self.dataset = dataset
        self.input_length = input_length  # Move constants to __init__

        # Calculate sample_pids for this rank once during init
        num_item_per_rank = len(sample_pids) // nranks
        self.sample_pids = sample_pids[
            rank * num_item_per_rank : (rank + 1) * num_item_per_rank
        ]

    def __iter__(self):
        for pid in self.sample_pids:
            dataset = self.dataset[pid]["input_ids"]
            assert len(dataset) == self.input_length, f"len(dataset) = {len(dataset)}"
            yield dataset

    def __getitem__(self, item):
        dataset = self.dataset[self.sample_pids[item]]["input_ids"]
        assert len(dataset) == self.input_length, f"len(dataset) = {len(dataset)}"
        return dataset

    def __len__(self):
        return len(self.sample_pids)


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

        chunksize = chunksize or self.get_chunksize()
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

        chunksize = self.get_chunksize()
        data_length = len(self)

        total_ranks = Run().nranks
        number_of_data_per_rank = (data_length + total_ranks - 1) // total_ranks
        loop_start_offset = rank * number_of_data_per_rank
        loop_end_offset = min(loop_start_offset + number_of_data_per_rank, data_length)
        # Further divide into chunks of size chunksize
        for chunk_start in range(loop_start_offset, loop_end_offset, chunksize):
            chunk_end = min(chunk_start + chunksize, loop_end_offset)
            yield range(chunk_start, chunk_end)

    def get_chunksize(self):
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
