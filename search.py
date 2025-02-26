import logging
from typing import List

import hkkang_utils.misc as misc_utils

from warp.custom_searcher import CustomSearcher

logger = logging.getLogger("Indexing")


class CustomCollection:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


def main():
    searcher = CustomSearcher()
    results: List[str] = searcher.search(
        "When did Barack Obama become president?", k=5, return_as_text=True
    )
    print(results)
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
