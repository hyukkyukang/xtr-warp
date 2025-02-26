from flask import Flask, request, jsonify

from configs import COLLECTION_PATH, DATASET_NAME, NBITS
from custom import CustomCollection, CustomWARPRunConfig, print_query
from warp.engine.searcher import WARPSearcher


class WARPSearcherServer:
    def __init__(self, config: CustomWARPRunConfig):
        self.config = config
        self.__initialize_searcher()

    def __initialize_searcher(self):
        collection = CustomCollection(name=DATASET_NAME, path=COLLECTION_PATH)
        config = CustomWARPRunConfig(
            nbits=NBITS,
            collection=collection,
        )
        self.searcher = WARPSearcher(config)

    def search(self, query: str, k: int = 10):
        return self.searcher.search(query, k)


# Initialize Flask app
app = Flask(__name__)

# Create global searcher instance
searcher_config = CustomWARPRunConfig(
    nbits=NBITS,
    collection=CustomCollection(name=DATASET_NAME, path=COLLECTION_PATH),
)
searcher_server = WARPSearcherServer(searcher_config)


@app.route("/search", methods=["POST"])
def search_endpoint():
    """
    Search endpoint that accepts JSON with query and optional k parameter
    Example request:
    {
        "query": "how do butterflies taste?",
        "k": 10
    }
    """
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Missing query parameter"}), 400

        query = data["query"]
        k = data.get("k", 10)  # Default to 10 results if k not specified

        # Perform search
        passage_ids, ranks, scores = searcher_server.search(query, k=k)

        # Format results
        results = []
        for pid, rank, score in zip(passage_ids, ranks, scores):
            results.append(
                {"passage_id": int(pid), "rank": int(rank), "score": float(score)}
            )

        return jsonify({"query": query, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})


def main():
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
