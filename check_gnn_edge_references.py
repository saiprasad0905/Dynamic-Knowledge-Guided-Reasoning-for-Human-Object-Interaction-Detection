
import torch
from pathlib import Path

DATA_DIR = Path("training_hybrid_graph_dataset")

def main():
    total_graphs = 0
    broken_human_edges = 0
    broken_object_edges = 0
    broken_hoi_edges = 0

    for path in sorted(DATA_DIR.glob("training_graph_chunk_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)

        for graph in data["graphs"]:
            total_graphs += 1

            human_ids = set(graph["humans"].keys())
            object_ids = set(graph["objects"].keys())

            for edge in graph["human_edges"]:
                if edge["source"] not in human_ids or edge["target"] not in human_ids:
                    broken_human_edges += 1

            for edge in graph["object_edges"]:
                if edge["source"] not in object_ids or edge["target"] not in object_ids:
                    broken_object_edges += 1

            for edge in graph["hoi_edges"]:
                if (
                    edge["subject_id"] not in human_ids
                    or edge["object_id"] not in object_ids
                ):
                    broken_hoi_edges += 1

    print("Graphs checked:", total_graphs)
    print("Broken human edges:", broken_human_edges)
    print("Broken object edges:", broken_object_edges)
    print("Broken HOI edges:", broken_hoi_edges)

    if broken_human_edges or broken_object_edges or broken_hoi_edges:
        raise ValueError("Some graph edges reference missing nodes.")

    print("PASS: all graph references are valid.")

if __name__ == "__main__":
    main()
