import logging
import os
from typing import List

import hkkang_utils.misc as misc_utils
from datasets import Dataset
from transformers import AutoTokenizer

from configs import (
    BATCH_SIZE,
    CHUNK_LENGTH,
    COLLECTION_PATH,
    DATASET_NAME,
    INDEX_ROOT,
    NBITS,
    NRANKS,
    NUM_CHUNKS_PER_ITEM,
    NUM_WORKERS,
    RESUME,
    SRC_TOKENIZER_NAME,
    K,
)
from custom import CustomWARPRunConfig
from warp.engine.searcher import WARPSearcher

logger = logging.getLogger("Indexing")


class CustomCollection:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


def main():
    os.environ["INDEX_ROOT"] = INDEX_ROOT

    # Define the collection (i.e., list of passages)
    collection = CustomCollection(name=DATASET_NAME, path=COLLECTION_PATH)
    config = CustomWARPRunConfig(
        nbits=NBITS,
        collection=collection,
        nranks=NRANKS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        resume=RESUME,
    )

    # Load dataset
    dataset = Dataset.load_from_disk(COLLECTION_PATH)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SRC_TOKENIZER_NAME)

    # Handle "user" queries using the searcher.
    def global_chunk_id_to_passage_id_and_local_chunk_id(global_chunk_id: int):
        passage_id = global_chunk_id // NUM_CHUNKS_PER_ITEM
        local_chunk_id = global_chunk_id % NUM_CHUNKS_PER_ITEM
        return passage_id, local_chunk_id

    def print_query(searcher: WARPSearcher, query: str):
        print(f"Query: {query}")
        passage_ids, _, scores = searcher.search(query, k=5)
        for rank, (pid, score) in enumerate(zip(passage_ids, scores)):
            passage_id, local_chunk_id = (
                global_chunk_id_to_passage_id_and_local_chunk_id(pid)
            )
            # Get the passage input ids
            input_ids: List[int] = dataset[passage_id]["input_ids"]
            # Get the chunk input ids
            chunk_input_ids: List[int] = input_ids[
                local_chunk_id * CHUNK_LENGTH : (local_chunk_id + 1) * CHUNK_LENGTH
            ]
            # Convert to tokens
            tokens: str = tokenizer.decode(chunk_input_ids)
            print(f"Rank {rank+1}: {score}")
            print(tokens)
            print("")

        print("====================")

    # Prepare for searching via the constructed index.
    searcher = WARPSearcher(config)
    print_query(searcher, "how do butterflies taste?")
    print_query(searcher, "quickest war in history?")

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
