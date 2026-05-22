import copy
import json


def save_graph(graph: dict, path: str) -> None:
    """Serialize the graph dict to a JSON file.
    Strips ``image_bytes`` from nodes to keep the file small."""
    out = copy.deepcopy(graph)
    for node in out.get("nodes", []):
        node.pop("image_bytes", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_graph(path: str) -> dict:
    """Load a graph JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)