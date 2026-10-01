gt = [2, 71, 579, 261]
pred = [5.9685516357421875, 107.64239501953125, 588.1085205078125, 435.1251525878906]

x1 = max(gt[0], pred[0])
y1 = max(gt[1], pred[1])
x2 = min(gt[2], pred[2])
y2 = min(gt[3], pred[3])

intersection = max(0, x2 - x1) * max(0, y2 - y1)
gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
iou = intersection / (gt_area + pred_area - intersection)

print("Ground-truth knife box:", gt)
print("Predicted knife box    :", pred)
print("IoU                    :", round(iou, 4))
