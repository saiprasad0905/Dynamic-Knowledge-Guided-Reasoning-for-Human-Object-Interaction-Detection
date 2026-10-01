import torch
from pathlib import Path


CHUNK_DIR = Path("structured_hoi")


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

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


def build_graph_counts(image):
    hois = image["hois"]

    humans = {}
    objects = {}
    pairs = set()

    for hoi in hois:
        subject_id = hoi["subject_id"]
        object_id = hoi["object_id"]

        humans[subject_id] = {
            "category": hoi["subject_category"],
            "bbox": hoi["subject_bbox"],
        }

        objects[object_id] = {
            "category": hoi["object_category"],
            "bbox": hoi["object_bbox"],
        }

        pairs.add((subject_id, object_id))

    object_edges = 0

    object_items = list(objects.items())

    for i in range(len(object_items)):
        _, object_a = object_items[i]

        for j in range(i + 1, len(object_items)):
            _, object_b = object_items[j]

            if object_a["category"] != object_b["category"]:
                continue

            object_edges += 1

    human_edges = 0

    human_items = list(humans.items())

    for i in range(len(human_items)):
        _, human_a = human_items[i]

        for j in range(i + 1, len(human_items)):
            _, human_b = human_items[j]

            human_edges += 1

    return (
        len(humans),
        len(objects),
        len(pairs),
        object_edges,
        human_edges,
    )


def main():
    chunk_files = sorted(
        CHUNK_DIR.glob("structured_chunk_*.pt")
    )

    print("Hybrid graph dataset analysis")
    print("=" * 70)
    print("Structured chunks:", len(chunk_files))

    total_images = 0

    human_counts = []
    object_counts = []
    ho_edge_counts = []
    oo_edge_counts = []
    hh_edge_counts = []

    for chunk_file in chunk_files:
        print("Processing:", chunk_file.name)

        data = torch.load(
            chunk_file,
            map_location="cpu",
            weights_only=False,
        )

        for image in data["images"]:
            (
                humans,
                objects,
                ho_edges,
                oo_edges,
                hh_edges,
            ) = build_graph_counts(image)

            human_counts.append(humans)
            object_counts.append(objects)
            ho_edge_counts.append(ho_edges)
            oo_edge_counts.append(oo_edges)
            hh_edge_counts.append(hh_edges)

            total_images += 1

    print("\nResults")
    print("-" * 70)

    print("Total images:", total_images)

    print(
        "\nHuman nodes per image:"
        f"\n  min  = {min(human_counts)}"
        f"\n  max  = {max(human_counts)}"
        f"\n  avg  = {sum(human_counts) / len(human_counts):.2f}"
    )

    print(
        "\nObject nodes per image:"
        f"\n  min  = {min(object_counts)}"
        f"\n  max  = {max(object_counts)}"
        f"\n  avg  = {sum(object_counts) / len(object_counts):.2f}"
    )

    print(
        "\nH-O edges per image:"
        f"\n  min  = {min(ho_edge_counts)}"
        f"\n  max  = {max(ho_edge_counts)}"
        f"\n  avg  = {sum(ho_edge_counts) / len(ho_edge_counts):.2f}"
    )

    print(
        "\nObject-object edges per image:"
        f"\n  min  = {min(oo_edge_counts)}"
        f"\n  max  = {max(oo_edge_counts)}"
        f"\n  avg  = {sum(oo_edge_counts) / len(oo_edge_counts):.2f}"
    )

    print(
        "\nHuman-human edges per image:"
        f"\n  min  = {min(hh_edge_counts)}"
        f"\n  max  = {max(hh_edge_counts)}"
        f"\n  avg  = {sum(hh_edge_counts) / len(hh_edge_counts):.2f}"
    )


if __name__ == "__main__":
    main()