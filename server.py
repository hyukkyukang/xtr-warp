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


@app.route("/search_multiple", methods=["POST"])
def search_multiple_endpoint():
    """
    Search multiple endpoint that accepts JSON with queries and optional k parameter
    Processes queries one by one sequentially

    Example request:
    {
        "queries": ["how do butterflies taste?", "when did Obama become president?"],
        "k": 10,
        "return_as_text": false
    }
    """
    try:
        data = request.get_json()

        if not data or "queries" not in data:
            return jsonify({"error": "Missing queries parameter"}), 400

        queries = data["queries"]
        k = data.get("k", 10)  # Default to 10 results if k not specified
        return_as_text = data.get("return_as_text", False)

        # Perform search_multiple
        results = searcher_server.search_multiple(
            queries, k=k, return_as_text=return_as_text
        )

        # Format results
        return jsonify({"queries": queries, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search_batch", methods=["POST"])
def search_batch_endpoint():
    """
    Batch search endpoint that accepts JSON with queries and optional k parameter
    Processes all queries in a single batch operation

    Example request:
    {
        "queries": ["how do butterflies taste?", "when did Obama become president?"],
        "k": 10,
        "return_as_text": false
    }
    """
    try:
        data = request.get_json()

        if not data or "queries" not in data:
            return jsonify({"error": "Missing queries parameter"}), 400

        queries = data["queries"]
        k = data.get("k", 10)  # Default to 10 results if k not specified
        return_as_text = data.get("return_as_text", False)

        # Perform batch search
        results = searcher_server.search_batch(
            queries, k=k, return_as_text=return_as_text
        )

        # Format results
        return jsonify({"queries": queries, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/convert_to_text", methods=["POST"])
def convert_to_text_endpoint():
    """
    Endpoint to convert a global chunk ID to text

    Example request:
    {
        "global_chunk_id": 123
    }
    """
    try:
        data = request.get_json()

        if not data or "global_chunk_id" not in data:
            return jsonify({"error": "Missing global_chunk_id parameter"}), 400

        global_chunk_id = data["global_chunk_id"]

        # Convert to text
        text = searcher_server.convert_global_chunk_id_to_text(global_chunk_id)

        # Return the text
        return jsonify({"global_chunk_id": global_chunk_id, "text": text})

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
