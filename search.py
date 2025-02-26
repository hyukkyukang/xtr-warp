import logging
import os
from typing import Dict, List

import hkkang_utils.misc as misc_utils
import hkkang_utils.time as time_utils
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

    def show_results(passage_ids: List[int], scores: List[float]):
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

    def print_query(searcher: WARPSearcher, queries: List[str], batched: bool = True):
        print(f"Is batched: {batched}")
        print(f"Queries: {queries}")
        if batched:
            queries: Dict[str, str] = {str(i): query for i, query in enumerate(queries)}
            results = searcher.search_all(queries, k=5)
            with time_utils.Timer("Search").measure(print_measured_time=True):
                results = searcher.search_all(queries, k=5)
            for key, result in results.ranking.data.items():
                # Extract passage_ids and scores from the result
                passage_ids = [item[0] for item in result]
                scores = [item[2] for item in result]
                show_results(passage_ids, scores)
        else:
            for query in queries:
                passage_ids, _, scores = searcher.search(query, k=5)
                with time_utils.Timer("Search").measure(print_measured_time=True):
                    passage_ids, _, scores = searcher.search(query, k=5)
                show_results(passage_ids, scores)

        print("====================")

    # Prepare for searching via the constructed index.
    q1 = "how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste? how do butterflies taste?"
    q2 = "quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history? quickest war in history?"
    q3 = "When did Barack Obama become president? When did Barack Obama become president? When did Barack Obama become president? When did Barack Obama become president? When did Barack Obama become president? When did Barack Obama become president? When did Barack Obama become president? When did Barack Obama become president?"
    searcher = WARPSearcher(config)
    # Single query
    print_query(searcher, [q1], batched=False)
    print_query(searcher, [q2], batched=False)
    print_query(searcher, [q3], batched=False)
    # Batch queries
    print_query(searcher, [q1, q2, q3], batched=True)

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
