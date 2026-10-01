import glob
from collections import Counter

import torch


STRUCTURED_DIR = "structured_hoi"


def main():
    files = sorted(
        glob.glob(
            f"{STRUCTURED_DIR}/structured_chunk_*.pt"
        )
    )

    total_images = 0
    total_human_nodes = 0
    total_object_nodes = 0
    total_edges = 0
    total_hoi_relations = 0

    image_human_counts = []
    image_object_counts = []
    image_edge_counts = []
    image_hoi_counts = []

    for file_path in files:

        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        for image in chunk["images"]:

            hois = image["hois"]

            humans = set()
            objects = set()
            edges = set()

            for hoi in hois:

                humans.add(hoi["subject_id"])
                objects.add(hoi["object_id"])

                edges.add(
                    (
                        hoi["subject_id"],
                        hoi["object_id"],
                    )
                )

            total_images += 1

            total_human_nodes += len(humans)
            total_object_nodes += len(objects)
            total_edges += len(edges)
            total_hoi_relations += len(hois)

            image_human_counts.append(len(humans))
            image_object_counts.append(len(objects))
            image_edge_counts.append(len(edges))
            image_hoi_counts.append(len(hois))

    print("\nStructured graph entity analysis")
    print("=" * 70)

    print("Images:", total_images)

    print("\nTotal entities")
    print("-" * 40)

    print("Human nodes:", total_human_nodes)
    print("Object nodes:", total_object_nodes)
    print("Unique human-object edges:", total_edges)
    print("HOI relations:", total_hoi_relations)

    print("\nAverage per image")
    print("-" * 40)

    print(
        "Human nodes:",
        round(
            total_human_nodes / total_images,
            3,
        ),
    )

    print(
        "Object nodes:",
        round(
            total_object_nodes / total_images,
            3,
        ),
    )

    print(
        "Unique H-O edges:",
        round(
            total_edges / total_images,
            3,
        ),
    )

    print(
        "HOI relations:",
        round(
            total_hoi_relations / total_images,
            3,
        ),
    )

    print("\nDistribution")
    print("-" * 40)

    print(
        "Human nodes:",
        Counter(image_human_counts),
    )

    print(
        "Object nodes:",
        Counter(image_object_counts),
    )

    print(
        "Unique H-O edges:",
        Counter(image_edge_counts),
    )

    print(
        "HOI relations:",
        Counter(image_hoi_counts),
    )


if __name__ == "__main__":
    main()