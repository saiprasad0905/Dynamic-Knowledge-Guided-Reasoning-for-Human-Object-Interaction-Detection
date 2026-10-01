import torch
from pathlib import Path


CHUNK_DIR = Path("structured_hoi")

THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
]


def analyze_image(image, threshold):
    selected = [
        hoi
        for hoi in image["hois"]
        if hoi["score"] >= threshold
    ]

    humans = set()
    objects = set()
    pairs = set()

    for hoi in selected:
        humans.add(hoi["subject_id"])
        objects.add(hoi["object_id"])
        pairs.add(
            (
                hoi["subject_id"],
                hoi["object_id"],
            )
        )

    return (
        len(selected),
        len(humans),
        len(objects),
        len(pairs),
    )


def main():
    chunk_files = sorted(
        CHUNK_DIR.glob("structured_chunk_*.pt")
    )

    results = {
        threshold: {
            "hois": [],
            "humans": [],
            "objects": [],
            "pairs": [],
        }
        for threshold in THRESHOLDS
    }

    total_images = 0

    print("Graph score threshold analysis")
    print("=" * 70)

    for chunk_file in chunk_files:
        print("Processing:", chunk_file.name)

        data = torch.load(
            chunk_file,
            map_location="cpu",
            weights_only=False,
        )

        for image in data["images"]:
            for threshold in THRESHOLDS:
                (
                    hois,
                    humans,
                    objects,
                    pairs,
                ) = analyze_image(
                    image,
                    threshold,
                )

                results[threshold]["hois"].append(hois)
                results[threshold]["humans"].append(humans)
                results[threshold]["objects"].append(objects)
                results[threshold]["pairs"].append(pairs)

            total_images += 1

    print("\nResults")
    print("-" * 70)
    print("Total images:", total_images)

    for threshold in THRESHOLDS:
        hois = results[threshold]["hois"]
        humans = results[threshold]["humans"]
        objects = results[threshold]["objects"]
        pairs = results[threshold]["pairs"]

        print(
            f"\nScore >= {threshold}"
        )

        print(
            f"  HOI records: "
            f"min={min(hois)}, "
            f"max={max(hois)}, "
            f"avg={sum(hois) / len(hois):.2f}"
        )

        print(
            f"  Human nodes: "
            f"min={min(humans)}, "
            f"max={max(humans)}, "
            f"avg={sum(humans) / len(humans):.2f}"
        )

        print(
            f"  Object nodes: "
            f"min={min(objects)}, "
            f"max={max(objects)}, "
            f"avg={sum(objects) / len(objects):.2f}"
        )

        print(
            f"  H-O pairs: "
            f"min={min(pairs)}, "
            f"max={max(pairs)}, "
            f"avg={sum(pairs) / len(pairs):.2f}"
        )


if __name__ == "__main__":
    main()