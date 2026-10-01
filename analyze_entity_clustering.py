import glob
import torch

CHUNK_DIR = "structured_hoi"
THRESHOLDS = [0.01, 0.05, 0.10]
OBJECT_IOU_THRESHOLD = 0.90


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def cluster_objects(object_nodes):
    """
    Conservative complete-link clustering.

    A new object can join a cluster only if its box has
    IoU >= threshold with EVERY object already in that cluster.

    This prevents the chaining problem seen with naive IoU merging.
    """

    clusters = []

    for node in object_nodes:

        placed = False

        for cluster in clusters:

            same_category = all(
                member["object_category"]
                == node["object_category"]
                for member in cluster
            )

            if not same_category:
                continue

            compatible = all(
                compute_iou(
                    member["object_bbox"],
                    node["object_bbox"]
                ) >= OBJECT_IOU_THRESHOLD
                for member in cluster
            )

            if compatible:
                cluster.append(node)
                placed = True
                break

        if not placed:
            clusters.append([node])

    return clusters


def main():
    files = sorted(
        glob.glob(f"{CHUNK_DIR}/structured_chunk_*.pt")
    )

    totals = {
        threshold: {
            "images": 0,
            "human_nodes": 0,
            "raw_object_nodes": 0,
            "clustered_object_nodes": 0,
            "edges": 0,
            "shared_object_entities": 0,
        }
        for threshold in THRESHOLDS
    }

    for file_path in files:

        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False
        )

        for image in chunk["images"]:

            for threshold in THRESHOLDS:

                candidates = [
                    hoi
                    for hoi in image["hois"]
                    if hoi["score"] >= threshold
                ]

                # Keep the highest-scoring verb for each
                # subject-object query pair.
                unique_pairs = {}

                for hoi in candidates:

                    pair = (
                        hoi["subject_id"],
                        hoi["object_id"]
                    )

                    if (
                        pair not in unique_pairs
                        or hoi["score"]
                        > unique_pairs[pair]["score"]
                    ):
                        unique_pairs[pair] = hoi

                pairs = list(unique_pairs.values())

                # Unique human query entities.
                human_ids = {
                    hoi["subject_id"]
                    for hoi in pairs
                }

                # Unique object query entities.
                object_ids = {}

                for hoi in pairs:

                    object_id = hoi["object_id"]

                    if (
                        object_id not in object_ids
                        or hoi["score"] > object_ids[object_id]["score"]
                    ):
                        object_ids[object_id] = hoi

                object_nodes = list(object_ids.values())

                # Cluster highly overlapping same-category
                # object detections.
                clusters = cluster_objects(object_nodes)

                # Map every original object ID to its cluster ID.
                object_to_cluster = {}

                for cluster_id, cluster in enumerate(clusters):

                    for node in cluster:
                        object_to_cluster[
                            node["object_id"]
                        ] = cluster_id

                # Build final graph edges:
                # human -> clustered object.
                graph_edges = set()

                for hoi in pairs:

                    human_id = hoi["subject_id"]
                    object_id = hoi["object_id"]

                    cluster_id = object_to_cluster[object_id]

                    graph_edges.add(
                        (human_id, cluster_id)
                    )

                # Count object clusters connected to
                # more than one distinct human.
                cluster_humans = {}

                for human_id, cluster_id in graph_edges:

                    if cluster_id not in cluster_humans:
                        cluster_humans[cluster_id] = set()

                    cluster_humans[cluster_id].add(human_id)

                shared_object_entities = sum(
                    1
                    for humans in cluster_humans.values()
                    if len(humans) > 1
                )

                totals[threshold]["images"] += 1
                totals[threshold]["human_nodes"] += len(human_ids)
                totals[threshold]["raw_object_nodes"] += len(object_nodes)
                totals[threshold]["clustered_object_nodes"] += len(clusters)
                totals[threshold]["edges"] += len(graph_edges)
                totals[threshold][
                    "shared_object_entities"
                ] += shared_object_entities

    print("\nConservative entity clustering analysis")
    print("=" * 70)

    print(
        f"Object IoU clustering threshold: "
        f"{OBJECT_IOU_THRESHOLD}"
    )

    for threshold in THRESHOLDS:

        data = totals[threshold]

        print(f"\nScore threshold >= {threshold}")
        print("-" * 40)

        print("Images:", data["images"])
        print("Human nodes:", data["human_nodes"])
        print("Raw object nodes:", data["raw_object_nodes"])
        print(
            "Clustered object nodes:",
            data["clustered_object_nodes"]
        )
        print("Final graph edges:", data["edges"])
        print(
            "Shared-object entities:",
            data["shared_object_entities"]
        )


if __name__ == "__main__":
    main()