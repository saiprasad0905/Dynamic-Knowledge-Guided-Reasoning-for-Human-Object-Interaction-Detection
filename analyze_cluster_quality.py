import glob
from collections import Counter

import torch

CHUNK_DIR = "structured_hoi"

SCORE_THRESHOLD = 0.01
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
    clusters = []

    for node in object_nodes:

        placed = False

        for cluster in clusters:

            if not all(
                member["object_category"]
                == node["object_category"]
                for member in cluster
            ):
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

    cluster_sizes = Counter()

    largest_cluster = None
    largest_cluster_size = 0

    multi_node_clusters = 0
    total_clusters = 0

    images_with_large_cluster = 0

    for file_path in files:

        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False
        )

        for image in chunk["images"]:

            candidates = [
                hoi
                for hoi in image["hois"]
                if hoi["score"] >= SCORE_THRESHOLD
            ]

            unique_objects = {}

            for hoi in candidates:

                object_id = hoi["object_id"]

                if (
                    object_id not in unique_objects
                    or hoi["score"]
                    > unique_objects[object_id]["score"]
                ):
                    unique_objects[object_id] = hoi

            object_nodes = list(unique_objects.values())

            clusters = cluster_objects(object_nodes)

            image_has_large_cluster = False

            for cluster in clusters:

                size = len(cluster)

                cluster_sizes[size] += 1
                total_clusters += 1

                if size > 1:
                    multi_node_clusters += 1

                if size >= 3:
                    image_has_large_cluster = True

                if size > largest_cluster_size:

                    largest_cluster_size = size

                    largest_cluster = {
                        "size": size,
                        "category": cluster[0]["object_category"],
                        "nodes": [
                            {
                                "object_id": node["object_id"],
                                "score": node["score"],
                                "bbox": node["object_bbox"],
                            }
                            for node in cluster
                        ],
                    }

            if image_has_large_cluster:
                images_with_large_cluster += 1

    print("\nCluster quality analysis")
    print("=" * 70)

    print("Score threshold:", SCORE_THRESHOLD)
    print("Object IoU threshold:", OBJECT_IOU_THRESHOLD)

    print("\nCluster size distribution")
    print("-" * 40)

    for size in sorted(cluster_sizes):
        print(
            f"Size {size}: "
            f"{cluster_sizes[size]} clusters"
        )

    print("\nSummary")
    print("-" * 40)

    print("Total clusters:", total_clusters)
    print(
        "Clusters with >1 node:",
        multi_node_clusters
    )
    print(
        "Percentage multi-node:",
        round(
            100 * multi_node_clusters / total_clusters,
            2
        ),
        "%"
    )

    print(
        "Images with cluster size >=3:",
        images_with_large_cluster
    )

    print(
        "Largest cluster size:",
        largest_cluster_size
    )

    if largest_cluster is not None:

        print("\nLargest cluster example")
        print("-" * 40)



        print(
            "Object category:",
            largest_cluster["category"]
        )

        print(
            "Cluster size:",
            largest_cluster["size"]
        )

        for node in largest_cluster["nodes"]:
            print(
                "  object_id:",
                node["object_id"],
                "score:",
                round(node["score"], 6),
                "bbox:",
                node["bbox"]
            )


if __name__ == "__main__":
    main()