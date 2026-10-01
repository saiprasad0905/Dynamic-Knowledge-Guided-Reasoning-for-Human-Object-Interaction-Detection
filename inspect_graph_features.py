import torch
from pathlib import Path


STRUCTURED_FILE = Path(
    "structured_hoi/structured_chunk_0000.pt"
)


def main():
    data = torch.load(
        STRUCTURED_FILE,
        map_location="cpu",
        weights_only=False,
    )

    image = data["images"][0]
    hois = image["hois"]

    print("Graph feature inspection")
    print("=" * 70)

    print("\nImage")
    print("-" * 40)
    print("Image ID:", image["image_id"])
    print("Number of HOI records:", len(hois))

    humans = {}
    objects = {}
    pairs = {}

    for hoi in hois:
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

        pair = (subject_id, object_id)

        if pair not in pairs:
            pairs[pair] = []

        pairs[pair].append(hoi)

    print("\nHuman node example")
    print("-" * 40)

    human_id = sorted(humans.keys())[0]
    print("Node ID:", human_id)
    print("Category:", humans[human_id]["category"])
    print("Bounding box:", humans[human_id]["bbox"])

    print("\nObject node example")
    print("-" * 40)

    object_id = sorted(objects.keys())[0]
    print("Node ID:", object_id)
    print("Category:", objects[object_id]["category"])
    print("Bounding box:", objects[object_id]["bbox"])

    print("\nH-O edge example")
    print("-" * 40)

    pair = sorted(pairs.keys())[0]

    print("Subject ID:", pair[0])
    print("Object ID:", pair[1])

    for hoi in sorted(
        pairs[pair],
        key=lambda item: item["score"],
        reverse=True,
    ):
        print(
            f"verb_id={hoi['verb_id']}, "
            f"verb={hoi['verb']}, "
            f"score={hoi['score']:.6f}"
        )

    print("\nRaw H-O record fields")
    print("-" * 40)

    print(
        sorted(
            pairs[pair][0].keys()
        )
    )

    print("\nRaw H-O record")
    print("-" * 40)

    print(pairs[pair][0])

    print("\nStructured file top-level fields")
    print("-" * 40)

    print(sorted(image.keys()))


if __name__ == "__main__":
    main()