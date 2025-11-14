#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import cv2  # pip install opencv-python
import argparse
from tqdm import tqdm
import sys
import math
# ==================== PARSERS (Unchanged) ====================

def parse_box_line(line):
    """Parses a YOLO box line: [class] [x_c] [y_c] [w] [h]"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        # Returns [x_c, y_c, w, h]
        return list(map(float, parts[1:5]))
    except ValueError:
        return None

def parse_landmark_line(line):
    """
    Parses a YOLO-v8 pose-style landmark line.
    Format assumes: [class] [x y w h] [l1x l1y v1] ... [l5x l5y v5] (20 parts total)
    """
    parts = line.strip().split()
    
    if len(parts) < 20: 
        print(f"Warning: Skipping landmark line with {len(parts)} parts, expected 20.")
        return None
        
    try:
        # We will extract all 15 landmark parts (x, y, and visibility)
        # [l1x, l1y, v1, l2x, l2y, v2, ..., l5x, l5y, v5]
        landmark_data_with_vis = [
            parts[5],  # l1x
            parts[6],  # l1y
            parts[7],  # v1
            parts[8],  # l2x
            parts[9],  # l2y
            parts[10], # v2
            parts[11], # l3x (Nose)
            parts[12], # l3y (Nose)
            parts[13], # v3
            parts[14], # l4x
            parts[15], # l4y
            parts[16], # v4
            parts[17], # l5x
            parts[18], # l5y
            parts[19]  # v5
        ]
        
        # Convert all to float
        return list(map(float, landmark_data_with_vis))
    
    except (ValueError, IndexError) as e:
        print(f"Error parsing landmark line: {e} \nLine: {line}")
        return None

# ==================== HELPER FUNCTIONS (Updated) ====================

# ==================== HELPER FUNCTIONS (Updated) ====================

def is_landmark_inside_box(landmark_norm, box_norm):
    """
    Checks if a normalized (lx, ly) point is inside a
    normalized (xc, yc, w, h) box.
    """
    lx, ly = landmark_norm
    xc, yc, w, h = box_norm
    
    # Calculate box boundaries (normalized)
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    
    # Check if point is inside
    return (x1 <= lx <= x2) and (y1 <= ly <= y2)

def get_landmark_centroid(landmark_data_norm):
    """
    Calculates the centroid (mean x, y) of the 5 landmarks.
    Assumes landmark_data_norm is the 15-element list [l1x, l1y, v1, ...].
    """
    sum_x = 0.0
    sum_y = 0.0
    
    for i in range(0, 15, 3):
        sum_x += landmark_data_norm[i]
        sum_y += landmark_data_norm[i+1]
        
    mean_x = sum_x / 5.0
    mean_y = sum_y / 5.0
    
    return (mean_x, mean_y)

def calculate_distance(p1, p2):
    """
    Calculates Euclidean distance between two (x, y) points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ==================== CORE LOGIC (Updated) ====================

def convert_merged_yolo_to_json(image_dir, box_dir, landmark_dir, output_json):
    """
    Merges separate YOLO box and landmark .txt files into a single JSON
    by geometrically matching landmarks to boxes using the landmark centroid.
    Solves ambiguity by choosing the box with the closest center.
    """
    ground_truth_data = {}
    
    debug_output_dir = "debug_visualizations"
    os.makedirs(debug_output_dir, exist_ok=True)
    print(f"Saving ambiguity visualizations to: {os.path.abspath(debug_output_dir)}")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images. Matching boxes to landmarks...")

    for image_filename in tqdm(image_files, desc="Matching data"):
        image_filenamew = image_filename.replace(";", "_")
        
        if image_filename == ("dmd_gA_1_s2_gA_1_s2_2019-03-08T09;21;03+01;00_rgb_body_frame_6061.png"):
            print("som")
            
        image_path = os.path.join(image_dir, image_filename)
        base_filename = os.path.splitext(image_filenamew)[0]
        
        box_label_path = os.path.join(box_dir, base_filename + ".txt")
        landmark_label_path = os.path.join(landmark_dir, base_filename + ".txt")

        # 1. Get image dimensions
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"\nWarning: Could not read image {image_path}. Skipping.")
                continue
            img_height, img_width, _ = img.shape
        except Exception as e:
            print(f"\nError reading {image_path}: {e}. Skipping.")
            continue
            
        # 2. Check if label files exist
        if not os.path.exists(box_label_path) or not os.path.exists(landmark_label_path):
            if not os.path.exists(box_label_path) and not os.path.exists(landmark_label_path):
                ground_truth_data[image_filename] = {} 
            continue

        # 3. Read lines from both files
        try:
            with open(box_label_path, 'r') as f_box:
                box_lines = f_box.readlines()
            with open(landmark_label_path, 'r') as f_landmark:
                landmark_lines = f_landmark.readlines()
        except Exception as e:
            print(f"\nError reading label files for {image_filename}: {e}. Skipping.")
            continue
        
        # 4. Parse all lines (landmarks are now 15-element lists)
        parsed_boxes_norm = [parse_box_line(l) for l in box_lines if parse_box_line(l)]
        parsed_landmarks_norm = [parse_landmark_line(l) for l in landmark_lines if parse_landmark_line(l)]

        if not parsed_boxes_norm and not parsed_landmarks_norm:
            ground_truth_data[image_filename] = {}
            continue
        
        if not parsed_boxes_norm or not parsed_landmarks_norm:
            print(f"\nWarning: Mismatch for {base_filename}. Boxes: {len(parsed_boxes_norm)}, Landmarks: {len(parsed_landmarks_norm)}. Skipping.")
            continue

        # 5. NEW: Landmark-centric matching with Centroid and Closest-Center Ambiguity Solving
        
        faces_dict = {}
        face_counter = 1
        used_box_indices = set() # Keep track of boxes that are "claimed"

        # Iterate over each LANDMARK set
        for landmark_index, landmark_data_norm in enumerate(parsed_landmarks_norm):
            
            # Calculate the centroid for this landmark set
            centroid_norm = get_landmark_centroid(landmark_data_norm)
            
            # Stores (box_index, distance_to_center, box_data_norm)
            matching_boxes = [] 
            
            # Find all available boxes that this centroid fits inside
            for box_index, box_data_norm in enumerate(parsed_boxes_norm):
                # Skip if this box is already claimed by another landmark
                if box_index in used_box_indices:
                    continue
                
                if is_landmark_inside_box(centroid_norm, box_data_norm):
                    # --- MODIFICATION ---
                    # Calculate distance from landmark centroid to box center
                    box_center_norm = (box_data_norm[0], box_data_norm[1]) # (xc, yc)
                    distance = calculate_distance(centroid_norm, box_center_norm)
                    matching_boxes.append((box_index, distance, box_data_norm))
                    # --- END MODIFICATION ---

            # --- Now, decide what to do with the matches ---
            
            if len(matching_boxes) == 0:
                # No available box matches this landmark, so skip it
                continue
            
            # --- MODIFICATION ---
            # Find the best match (the one with the *smallest distance*)
            best_match = min(matching_boxes, key=lambda b: b[1]) # b[1] is the distance
            best_box_index, best_box_distance, best_box_data_norm = best_match
            # --- END MODIFICATION ---

            if len(matching_boxes) > 1:
                # --- AMBIGUITY DETECTED (Logic updated) ---
                print(f"\n[DEBUG AMBIGUITY] Image: {image_filename}")
                print(f"  > Landmark Set #{landmark_index} (Centroid: {centroid_norm})")
                print(f"  >... fit in MULTIPLE available boxes:")
                for (b_idx, b_dist, b_data) in matching_boxes:
                    print(f"    - Box #{b_idx} (Distance: {b_dist:.4f}) Data: {b_data}")
                print(f"  >... SOLVED: Chose Box #{best_box_index} (Closest Center).")
                # (You could add visualization logic here if needed)
            
            # --- We have a final match (either unique or solved) ---
            
            # Mark this box as used
            used_box_indices.add(best_box_index)
            
            # De-normalize Box
            xc_norm, yc_norm, w_norm, h_norm = best_box_data_norm
            w_abs = w_norm * img_width
            h_abs = h_norm * img_height
            x1 = (xc_norm * img_width) - (w_abs / 2)
            y1 = (yc_norm * img_height) - (h_abs / 2)
            x2 = x1 + w_abs
            y2 = y1 + h_abs
            bbox_abs = [x1, y1, x2, y2]

            # De-normalize Landmarks
            landmarks_abs = []
            for j in range(0, 15, 3): 
                lx_norm = landmark_data_norm[j]
                ly_norm = landmark_data_norm[j+1]
                v_norm  = landmark_data_norm[j+2]
                
                lx_abs = lx_norm * img_width
                ly_abs = ly_norm * img_height
                v_int = int(v_norm)
                
                landmarks_abs.append([lx_abs, ly_abs, v_int])
            
            # Add to dict
            faces_dict[f"Face-{face_counter}"] = {
                "bbox": bbox_abs,
                "landmarks": landmarks_abs
            }
            face_counter += 1
            
        ground_truth_data[image_filename] = faces_dict

    # 7. Save the final JSON
    try:
        with open(output_json, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)
        print(f"\nSuccessfully created merged ground truth file at {output_json}")
    except Exception as e:
        print(f"\nError writing final JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="Merge separate YOLO box and landmark .txt files to a single JSON.")
    parser.add_argument("--image-dir", type=str, default="/home/syntonym4090/idil/telus/cloud-backbone/merged_problematic_images",
                        help="Path to the directory containing your images (.jpg, .png).")
    parser.add_argument("--box-dir", type=str, default="/home/syntonym4090/Desktop/coone-posttune-box/labels/train",
                        help="Path to the directory containing YOLO-formatted box .txt files.")
    parser.add_argument("--landmark-dir", type=str, default="/home/syntonym4090/Desktop/cooneposttune-lms/labels/train",
                        help="Path to the directory containing YOLO-formatted landmark .txt files.")
    parser.add_argument("--output-json", type=str, default="ground_truth_with_landmarks.json",
                        help="Path to save the output merged JSON file.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found at {args.image_dir}")
        sys.exit(1)
    if not os.path.isdir(args.box_dir):
        print(f"Error: Box label directory not found at {args.box_dir}")
        sys.exit(1)
    if not os.path.isdir(args.landmark_dir):
        print(f"Error: Landmark label directory not found at {args.landmark_dir}")
        sys.exit(1)

    convert_merged_yolo_to_json(args.image_dir, args.box_dir, args.landmark_dir, args.output_json)

if __name__ == "__main__":
    main()