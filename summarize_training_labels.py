import torch

path = "training_labeled_hoi/training_labeled_chunk_0000.pt"
data = torch.load(path, map_location="cpu", weights_only=False)

total_candidates = 0
positive_labels = 0
good_box_matches = 0
box_match_but_negative = 0

for image in data["images"]:
    for hoi in image["hois"]:
        total_candidates += 1

        is_positive = hoi["label"] == 1
        good_boxes = (
            hoi["subject_iou"] >= 0.5
            and hoi["object_iou"] >= 0.5
        )

        positive_labels += int(is_positive)
        good_box_matches += int(good_boxes)

        if good_boxes and not is_positive:
            box_match_but_negative += 1

print("Total candidates:", total_candidates)
print("Positive labels:", positive_labels)
print("Candidates with both box IoUs >= 0.5:", good_box_matches)
print("Good box matches labeled negative:", box_match_but_negative)