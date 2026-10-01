import torch
from pathlib import Path


CHUNK_DIR = Path("structured_hoi")
K_VALUES = [1, 3, 5, 10, 15, 20]


def unique_pairs(image):
    pairs = {}

    for hoi in image["hois"]:
        pair = (
            hoi["subject_id"],
            hoi["object_id"],
        )

        if pair not in pairs:
            pairs[pair] = hoi["score"]
        else:
            pairs[pair] = max(
                pairs[pair],
                hoi["score"],
            )

    return pairs


def main():
    chunk_files = sorted(
        CHUNK_DIR.glob("structured_chunk_*.pt")
    )

    total_images = 0

    pair_counts = []
    threshold_counts = {
        k: 0
        for k in K_VALUES
    }

    retained_candidates = {
        k: []
        for k in K_VALUES
    }

    print("Top-K H-O pair analysis")
    print("=" * 70)

    for chunk_file in chunk_files:
        print("Processing:", chunk_file.name)

        data = torch.load(
            chunk_file,
            map_location="cpu",
            weights_only=False,
        )

        for image in data["images"]:
            pairs = unique_pairs(image)

            # Sort pairs by their strongest HOI score.
            sorted_pairs = sorted(
                pairs.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            num_pairs = len(sorted_pairs)

            pair_counts.append(num_pairs)

            for k in K_VALUES:
                retained = min(
                    k,
                    num_pairs,
                )

                retained_candidates[k].append(
                    retained
                )

                if num_pairs >= k:
                    threshold_counts[k] += 1

            total_images += 1

    print("\nResults")
    print("-" * 70)
    print("Total images:", total_images)

    print(
        "\nUnique H-O pairs per image:"
    )

    print(
        f"  min={min(pair_counts)}, "
        f"max={max(pair_counts)}, "
        f"avg={sum(pair_counts) / len(pair_counts):.2f}"
    )

    for k in K_VALUES:
        retained = retained_candidates[k]

        images_with_k = threshold_counts[k]

        percentage = (
            100.0
            * images_with_k
            / total_images
        )

        avg_retained = (
            sum(retained)
            / len(retained)
        )

        print(
            f"\nTop-{k} pairs:"
        )

        print(
            f"  Images with >= {k} pairs: "
            f"{images_with_k} "
            f"({percentage:.2f}%)"
        )

        print(
            f"  Average retained pairs: "
            f"{avg_retained:.2f}"
        )


if __name__ == "__main__":
    main()