import math

import torch


STRUCTURED_FILE = (
    "structured_hoi/structured_chunk_0000.pt"
)

IMAGE_INDEX = 0

K_VALUES = [3, 5]


def box_center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def center_distance(box_a, box_b):
    center_a = box_center(box_a)
    center_b = box_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def inspect_knn(nodes, k):

    node_items = list(nodes.items())

    results = {}

    for node_id, node in node_items:

        distances = []

        for other_id, other in node_items:

            if node_id == other_id:
                continue

            distance = center_distance(
                node["bbox"],
                other["bbox"],
            )

            distances.append(
                (
                    distance,
                    other_id,
                    other["category"],
                )
            )

        distances.sort(
            key=lambda item: item[0]
        )

        results[node_id] = distances[:k]

    return results


def print_knn_results(
    title,
    nodes,
    knn_results,
    sample_count=5,
):

    print()
    print(title)
    print("-" * 70)

    for index, (node_id, neighbors) in enumerate(
        knn_results.items()
    ):

        node = nodes[node_id]

        print(
            f"Node {node_id} "
            f"(category={node['category']})"
        )

        for rank, (
            distance,
            neighbor_id,
            neighbor_category,
        ) in enumerate(
            neighbors,
            start=1,
        ):

            print(
                f"  {rank}. "
                f"Node {neighbor_id} "
                f"(category={neighbor_category}), "
                f"distance={distance:.2f}"
            )

        if index + 1 >= sample_count:
            break


def main():

    data = torch.load(
        STRUCTURED_FILE,
        map_location="cpu",
        weights_only=False,
    )

    image = data["images"][IMAGE_INDEX]

    humans = {}
    objects = {}

    for hoi in image["hois"]:

        subject_id = hoi["subject_id"]
        object_id = hoi["object_id"]

        if subject_id not in humans:
            humans[subject_id] = {
                "category": hoi["subject_category"],
                "bbox": hoi["subject_bbox"],
            }

        if object_id not in objects:
            objects[object_id] = {
                "category": hoi["object_category"],
                "bbox": hoi["object_bbox"],
            }

    print()
    print("=" * 70)
    print("KNN CONTEXTUAL EDGE INSPECTION")
    print("=" * 70)

    print(
        "Image ID:",
        image["image_id"]
    )

    print(
        "Human nodes:",
        len(humans)
    )

    print(
        "Object nodes:",
        len(objects)
    )

    for k in K_VALUES:

        human_knn = inspect_knn(
            humans,
            k,
        )

        object_knn = inspect_knn(
            objects,
            k,
        )

        print()
        print("=" * 70)
        print(f"K = {k}")
        print("=" * 70)

        print_knn_results(
            "Human-human nearest neighbors",
            humans,
            human_knn,
        )

        print_knn_results(
            "Object-object nearest neighbors",
            objects,
            object_knn,
        )


if __name__ == "__main__":
    main()
