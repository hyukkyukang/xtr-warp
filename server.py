from typing import *
from flask import Flask, jsonify, request

from warp.custom_searcher import CustomSearcher
from configs import HOST, PORT

# Initialize Flask app
app = Flask(__name__)

searcher_server = CustomSearcher()


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
        return_as_text = data.get("return_as_text", False)

        # Perform search
        global_chunk_ids: List[int] = searcher_server.search(
            query, k=k, return_as_text=return_as_text
        )

        # Format results
        return jsonify({"query": query, "results": global_chunk_ids})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})


def main():
    # Run the Flask app
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
