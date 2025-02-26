import logging
import os

import hkkang_utils.misc as misc_utils

from configs import (
    BATCH_SIZE,
    COLLECTION_PATH,
    DATASET_NAME,
    INDEX_ROOT,
    NBITS,
    NRANKS,
    NUM_WORKERS,
    K,
    RESUME,
)
from custom import CustomWARPRunConfig, construct_index

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

    # Construct an index over the provided collection.
    construct_index(config)

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
