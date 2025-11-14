#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
import os
import cv2  # 

def load_custom_predictions(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f) 
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{file_path}'. Error: {e}")
        raise
    return data


def load_custom_ground_truth(file_path):
    """
    Loads the custom ground truth JSON and structures it for evaluation.
    Expected GT format:
      {
        "path/or/name.jpg": {
            "Face-1": {"bbox":[x1,y1,x2,y2] or [x,y,w,h], ...},
            "Face-2": {"bbox":[...]}
        },
        ...
      }
    Returns: dict[str, list[[x1,y1,x2,y2], ...]]  (raw; later normalized)
    """
    ground_truths = defaultdict(list)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for filename, faces_dict in data.items():
            proc_filename = os.path.basename(filename)  # normalize to filename
            bboxes = []
            for _, face_data in faces_dict.items():
                if 'bbox' in face_data:
                    bboxes.append(face_data['bbox'])
            # [] if no faces -> valid negative image
            ground_truths[proc_filename] = bboxes
        return ground_truths
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{file_path}'")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{file_path}'. Error: {e}")
        raise

# ==================== Box utils ====================
# ... [_to_float4, is_valid_box, calculate_iou are unchanged] ...
def _to_float4(box):
    """
    Try to coerce a box-like object to [x1,y1,x2,y2] float list.
    Accepts:
      - list/tuple/np.array length>=4
      - dict with 'box' or 'bbox'
    Also converts [x,y,w,h] -> [x1,y1,x2,y2] if needed.
    Returns list [x1,y1,x2,y2] or None if invalid.
    """
    if box is None:
        return None

    # unwrap dict forms
    if isinstance(box, dict):
        if "box" in box and isinstance(box["box"], (list, tuple, np.ndarray)):
            box = box["box"]
        elif "bbox" in box and isinstance(box["bbox"], (list, tuple, np.ndarray)):
            box = box["bbox"]

    if not isinstance(box, (list, tuple, np.ndarray)) or len(box) < 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, box[:4])
    except Exception:
        return None

    # If looks like [x,y,w,h] (non-negative w/h and degenerate x2<=x1 or y2<=y1), convert
    if (x2 <= x1 or y2 <= y1) and (x2 >= 0.0 and y2 >= 0.0):
        x2 = x1 + x2
        y2 = y1 + y2

    # final sanity: ensure proper ordering
    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def is_valid_box(box):
    return _to_float4(box) is not None

def calculate_iou(boxA, boxB):
    """
    IoU for [x1,y1,x2,y2] boxes (robust to malformed input via _to_float4).
    """
    A = _to_float4(boxA)
    B = _to_float4(boxB)
    if A is None or B is None:
        return 0.0
    x1 = max(A[0], B[0]); y1 = max(A[1], B[1])
    x2 = min(A[2], B[2]); y2 = min(A[3], B[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    areaA = (A[2]-A[0]) * (A[3]-A[1])
    areaB = (B[2]-B[0]) * (B[3]-B[1])
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return inter / (union + 1e-12)

# ==================== FP/FN collection & visualization ====================

def collect_errors(predictions, ground_truths, confidence_threshold, iou_threshold):
    """
    Collects False Positives (FPs) and False Negatives (FNs).
    
    Returns: dict[image_id, dict{'fps': list[pred], 'fns': list[gt], 'gts': list[gt]}]
             'fps': list of FP predictions (dict with 'box' and 'score')
             'fns': list of FN ground truth boxes (list)
             'gts': list of ALL ground truth boxes for context
    """
    image_ids = set(predictions.keys()) | set(ground_truths.keys())
    problems = defaultdict(lambda: {'fps': [], 'fns': [], 'gts': []})
    invalid_pred_boxes = 0
    invalid_gt_boxes = 0
    final_problems= []

    for img_id in image_ids:
        # 1. Get and validate GTs
        raw_gts = ground_truths.get(img_id, [])
        gts = []
        for gb in raw_gts:
            n = _to_float4(gb)
            if n is None:
                invalid_gt_boxes += 1
            else:
                gts.append(n)
        problems[img_id]['gts'] = gts # Store all gts for context
        gt_matched = [False] * len(gts)

        # 2. Get and validate predictions
        preds_for_img = sorted(
            [p for p in predictions.get(img_id, []) if p.get("score", 0.0) >= confidence_threshold],
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        
        # 3. Match preds to GTs
        for pred in preds_for_img:
            pbox = _to_float4(pred.get("box"))
            if pbox is None:
                invalid_pred_boxes += 1
                continue

            best_iou = 0.0
            best_gt_idx = -1
            for j, gt_box in enumerate(gts):
                iou = calculate_iou(pbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_gt_idx != -1 and best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True  # This is a TP
                else:
                    # Matched a GT that was already matched by a higher-scoring pred
                   final_problems.append(img_id) # This is a duplicate (FP)
            else:
                # No match or match below threshold
                final_problems.append(img_id)
        
        # 4. Collect FNs (unmatched GTs)
        for i, gt_box in enumerate(gts):
            if not gt_matched[i]:
                final_problems.append(img_id)

    if invalid_pred_boxes or invalid_gt_boxes:
        print(f"[info] Skipped invalid boxes -> preds: {invalid_pred_boxes}, gts: {invalid_gt_boxes}")
    
    return final_problems



# ==================== CLI ====================

def main(args):
    predictions = load_custom_predictions(args.predictions_file)
    ground_truths = load_custom_ground_truth(args.gt_json)

    print(f"Loaded {len(predictions)} prediction image-keys and {len(ground_truths)} GT images.")

    problems = collect_errors(
        predictions=predictions,
        ground_truths=ground_truths,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
    )
    with open("hard_frames.json", "w") as f:
        json.dump(problems, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect & visualize false positives for object detection.")
    parser.add_argument("--predictions_file", type=str,
                        default="../data/cloud-face.json",
                        help="Path to the JSON prediction file.")
    parser.add_argument("--gt_json", type=str,
                        default="../data/FINAL_combined_output.json",
                        help="Path to the *custom* ground truth JSON file.")
    parser.add_argument("--conf", type=float, default=0.9,
                        help="Confidence threshold for counting/visualizing predictions.")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="IoU threshold for matching predictions to GT.")

    args = parser.parse_args()
    main(args)
