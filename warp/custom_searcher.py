from typing import *

# import hkkang_utils.pattern as pattern_utils
from datasets import Dataset
from transformers import AutoTokenizer

from configs import (
    CHUNK_LENGTH,
    COLLECTION_PATH,
    DATASET_NAME,
    NBITS,
    NUM_CHUNKS_PER_ITEM,
    SRC_TOKENIZER_NAME,
)
from custom import CustomCollection, CustomWARPRunConfig
from warp.data.collection import SampledCollection
from warp.engine.searcher import WARPSearcher


# TODO: Make it a singleton class
class CustomSearcher:
    def __init__(
        self,
        nbits: int = NBITS,
        dataset_name: str = DATASET_NAME,
        collection_path: str = COLLECTION_PATH,
    ):
        self.nbits = nbits
        self.dataset_name = dataset_name
        self.collection_path = collection_path
        self.__initialize_searcher()

    @property
    def collection(self):
        if hasattr(self, "_dataset"):
            return self._dataset
        dataset = Dataset.load_from_disk(self.collection_path)
        self._dataset = SampledCollection(
            dataset=dataset,
            sample_pids=range(len(dataset)),
            rank=0,
            nranks=1,
        )
        return self._dataset

    @property
    def tokenizer(self):
        if hasattr(self, "_tokenizer"):
            return self._tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(SRC_TOKENIZER_NAME)
        return self._tokenizer

    def __initialize_searcher(self):
        warp_tmp_collection = CustomCollection(
            name=self.dataset_name, path=self.collection_path
        )
        config = CustomWARPRunConfig(
            nbits=self.nbits,
            collection=warp_tmp_collection,
        )
        self.searcher = WARPSearcher(config)

    def search(
        self, query: str, k: int = 10, return_as_text: bool = False
    ) -> Union[List[int], List[str]]:
        # Search with the query
        global_chunk_ids, _, scores = self.searcher.search(query, k)

        # If return_as_text is True, convert the global chunk ids to texts
        if return_as_text:
            texts: List[str] = []
            for global_chunk_id in global_chunk_ids:
                text = self.convert_global_chunk_id_to_text(global_chunk_id)
                # Add the text to the results
                texts.append(text)
            return texts
        return global_chunk_ids

    def search_batch(
        self, queries: List[str], k: int = 10, return_as_text: bool = False
    ) -> Union[List[List[int]], List[List[str]]]:
        # Search with the queries
        results = self.searcher.search_all(queries, k)

        # Get the global chunk ids
        global_chunk_ids_list: List[List[int]] = []
        for result in results.ranking.data.values():
            global_chunk_ids_list.append([item[0] for item in result])

        # If return_as_text is True, convert the global chunk ids to texts
        if return_as_text:
            texts_list: List[List[str]] = []
            for global_chunk_ids in global_chunk_ids_list:
                texts_list.append(
                    [
                        self.convert_global_chunk_id_to_text(global_chunk_id)
                        for global_chunk_id in global_chunk_ids
                    ]
                )
            return texts_list
        return global_chunk_ids_list

    def convert_global_chunk_id_to_text(self, global_chunk_id: int) -> str:
        """Find the text of the chunk with the given global chunk id."""
        # Convert global chunk id to passage id and local chunk id
        passage_id, local_chunk_id = (
            self._global_chunk_id_to_passage_id_and_local_chunk_id(global_chunk_id)
        )
        # Get the passage input ids
        input_ids: List[int] = self.collection[passage_id]["input_ids"]
        # Get the chunk input ids
        chunk_input_ids: List[int] = input_ids[
            local_chunk_id * CHUNK_LENGTH : (local_chunk_id + 1) * CHUNK_LENGTH
        ]
        # Convert the chunk input ids to text
        text = self.tokenizer.decode(chunk_input_ids, skip_special_tokens=True)
        return text

    def _global_chunk_id_to_passage_id_and_local_chunk_id(
        self, global_chunk_id: int
    ) -> Tuple[int, int]:
        passage_id = global_chunk_id // NUM_CHUNKS_PER_ITEM
        local_chunk_id = global_chunk_id % NUM_CHUNKS_PER_ITEM
        return passage_id, local_chunk_id
