import logging
from typing import List, Tuple

import hkkang_utils.misc as misc_utils
from datasets import Dataset
from transformers import AutoTokenizer

from configs import (
    CHUNK_LENGTH,
    COLLECTION_PATH,
    HOST,
    PORT,
    SRC_TOKENIZER_NAME,
    NUM_CHUNKS_PER_ITEM,
)
from warp.custom_search_api import RemoteSearcher
from warp.custom_searcher import CustomSearcher
from warp.data.collection import SampledCollection

logger = logging.getLogger("Indexing")

dataset = None
tokenizer = None


class CustomCollection:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


def global_chunk_id_to_passage_id_and_local_chunk_id(
    global_chunk_id: int,
) -> Tuple[int, int]:
    passage_id = global_chunk_id // NUM_CHUNKS_PER_ITEM
    local_chunk_id = global_chunk_id % NUM_CHUNKS_PER_ITEM
    return passage_id, local_chunk_id


def convet_global_chunk_id_to_text(global_chunk_id: int) -> str:
    global dataset
    global tokenizer
    if dataset is None:
        dataset = Dataset.load_from_disk(COLLECTION_PATH)
        dataset = SampledCollection(
            dataset=dataset,
            sample_pids=range(len(dataset)),
            rank=0,
            nranks=1,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(SRC_TOKENIZER_NAME)
    # Convert the global text sequence idx to the text sequence idx
    passage_id, local_chunk_id = global_chunk_id_to_passage_id_and_local_chunk_id(
        global_chunk_id
    )
    input_ids = dataset[passage_id]["input_ids"]
    chunk_input_ids = input_ids[
        local_chunk_id * CHUNK_LENGTH : (local_chunk_id + 1) * CHUNK_LENGTH
    ]
    text = tokenizer.decode(chunk_input_ids, skip_special_tokens=True)
    return text


def test_retrieval_with_collection():
    # Load the collection
    # searcher = RemoteSearcher(server_url=f"http://{HOST}:{PORT}")
    searcher = CustomSearcher()
    dataset = Dataset.load_from_disk(COLLECTION_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SRC_TOKENIZER_NAME)
    TEST_NUM = 10
    for passage_idx in range(TEST_NUM):
        input_ids: List[int] = dataset[passage_idx]["input_ids"]
        retrieved_chunks: List[str] = []
        seen_text: List[str] = []
        for chunk_idx in range((len(input_ids) // CHUNK_LENGTH) - 1):
            current_chunk_input_ids = input_ids[
                chunk_idx * CHUNK_LENGTH : (chunk_idx + 1) * CHUNK_LENGTH
            ]
            next_chunk_input_ids = input_ids[
                (chunk_idx + 1) * CHUNK_LENGTH : (chunk_idx + 2) * CHUNK_LENGTH
            ]
            current_query = tokenizer.decode(
                current_chunk_input_ids, skip_special_tokens=True
            )
            next_query = tokenizer.decode(
                next_chunk_input_ids, skip_special_tokens=True
            )
            # Conduct retrieval with the current query
            result_ids: List[int] = searcher.search(
                current_query,
                k=3,
                return_as_text=False,
                passage_idx_to_ignore=passage_idx,
            )

            # Check passage_idx is not in the result
            for item in result_ids:
                pid, _ = global_chunk_id_to_passage_id_and_local_chunk_id(item)
                assert pid != passage_idx, f"passage_idx {passage_idx} is in the result"

            # Convert the results to text
            result_texts: List[str] = [
                convet_global_chunk_id_to_text(result) for result in result_ids
            ]
            # Add the current query to the seen_text
            seen_text.append(current_query)
            print(f"Current sequence: <{seen_text}>\n")
            print(f"Next sequence: <{next_query}>\n")
            print("=" * 100)
            # Push into retrieved_chunks as FIFO
            retrieved_chunks.append(result_texts)
            # Skip the first result because it's the current query
            for iidx, result in enumerate(retrieved_chunks[::-1]):
                print(f"Retrieved chunk {iidx}: <{result}>")
                print("/" * 100)
            input()
        print("-" * 100)
        print("-" * 100)
        print("-" * 100)
        print()

    return None


def test_single_query():
    # searcher = CustomSearcher()
    searcher = RemoteSearcher(server_url=f"http://{HOST}:{PORT}")
    results: List[str] = searcher.search(
        """Donald Trump""",
        k=5,
        return_as_text=True,
    )
    for idx, result in enumerate(results):
        print(f"{idx}: {result}")
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    test_retrieval_with_collection()
