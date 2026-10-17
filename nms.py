import numpy as np


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    x_intersection = np.maximum(0, np.minimum(x2, x4) - np.maximum(x1, x3))
    y_intersection = np.maximum(0, np.minimum(y2, y4) - np.maximum(y1, y3))
    intersection = x_intersection * y_intersection
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    denom = area1 + area2 - intersection
    return np.divide(intersection, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)

def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(-scores)
    assert isinstance(boxes, np.ndarray)
    sorted_boxes = boxes[sorted_indices]
    deleted_indices = set()
    for i, box in enumerate(sorted_boxes):
        if i in deleted_indices:
            continue
        ious = iou(box, sorted_boxes[i + 1:])
        selected_indices = set(np.nonzero(ious > iou_threshold)[0] + i + 1)
        deleted_indices = deleted_indices.union(selected_indices)
    return [sorted_indices[i] for i in range(len(boxes)) if i not in deleted_indices]

