import os
import argparse
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import sys
import json

def get_video_id(image_filename: str) -> str:
    """
    Extracts a unique video ID from an image filename.
    Assumes a 'filename_number.extension' format (e.g., 'video_001.png').
    """
    
    # Get the filename without the .png, .jpg, etc.
    stem = Path(image_filename).stem

    
    # Try splitting the stem by the last underscore
    # e.g., "video_001" -> ['video', '001']
    parts = stem.rsplit('_', 1)

    if parts[1] == "sim" or parts[1] == "brightdif":
        parts = parts[0].rsplit('_', 1)
        if len(parts) > 1 and parts[1].isdigit():
            # If yes, it matches 'filename_number'. Return 'filename'.
            return parts[0] + "_" + parts[1]
    # Check if the part after the underscore is purely a number
    if len(parts) > 1 and parts[1].isdigit():
        # If yes, it matches 'filename_number'. Return 'filename'.
        return parts[0]

    
    # If no pattern matches (e.g., 'single_image.png') OR
    # the part after the underscore is not a number (e.g., 'video_clip_a.png'),
    # then treat the entire stem as the unique ID.
    return stem

def write_split_file(output_path, video_ids, video_data_map):
    """Writes all data for the given video IDs to the output file."""
    # Shuffle the IDs before writing to mix them up within the file
    random.shuffle(video_ids) 
    with open(output_path, 'w') as f:
        for video_id in video_ids:
            lines_to_write = video_data_map.get(video_id)
            if lines_to_write:
                f.writelines(lines_to_write)


def random_split(array, ratio=0.5):
    # Shuffle a copy to avoid modifying the original
    shuffled = array.copy()
    random.shuffle(shuffled)
    
    # Determine split index
    split_index = int(len(shuffled) * ratio)
    
    # Split into two parts
    part1 = shuffled[:split_index]
    part2 = shuffled[split_index:]
    
    return part1, part2


def main(args):
    print(f"Reading full dataset from: {args.retinaface_txt}")
    
    # --- 1. Parse and Group by Video ID ---
    # video_data_map will store: {'video_id_1': ['# img1_path\n', 'bbox_line1\n']}
    video_data_map = defaultdict(list)
    current_video_id = None
    
    with open(args.retinaface_txt, 'r') as f:
        for line in tqdm(f, desc="Parsing full dataset"):
            if line.startswith('#'):
                # This is an image path line
                image_filename = Path(line.strip()[2:]).name
                current_video_id = get_video_id(image_filename)
                video_data_map[current_video_id].append(line)
            elif current_video_id:
                # This is an annotation line
                video_data_map[current_video_id].append(line)

    print(f"Found {len(video_data_map)} unique videos.")

    # --- 2. Identify Hard/Easy Video Groups ---
    print(f"Loading hard case list from: {args.hard_list_json}")
    try:
        with open(args.hard_list_json, 'r') as f:
            # --- MODIFIED ---
            # Load the JSON file, which is a list of filenames
            hard_image_filenames_list = json.load(f)
            hard_image_filenames = set(hard_image_filenames_list)
            # --- END MODIFIED ---
            
        print(f"Loaded {len(hard_image_filenames)} unique hard image names.")
    except FileNotFoundError:
        print(f"ERROR: Hard list JSON file not found at {args.hard_list_json}.")
        print("This file is required to create the validation set. Please create it and try again.")
        sys.exit(1) # Stop execution
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode {args.hard_list_json}. Is it a valid JSON list?")
        sys.exit(1)

    hard_video_ids = set()
    all_video_ids = set(video_data_map.keys())

    # Find all videos that contain at least one hard frame
    for video_id, lines in video_data_map.items():
        is_hard_video = False
        for line in lines:
            if line.startswith('#'):
                image_filename = Path(line.strip()[2:]).name
                if image_filename in hard_image_filenames:
                    is_hard_video = True
                    break
        
        if is_hard_video:
            hard_video_ids.add(video_id)

    # --- 3. Create the deterministic split ---
    
    # Validation set = all videos containing at least one hard frame
    hard_video_ides = list(hard_video_ids)
    
    # Training set = all other videos
    easy_video_ids = list(all_video_ids - hard_video_ids) 

    print(f"Identified {len(hard_video_ides)} 'Hard' videos for VALIDATION.")
    print(f"Identified {len(easy_video_ids)} 'Easy' videos for TRAINING.")

    hard1, hard2 = random_split(hard_video_ides, ratio=0.5)

    train_video_ids = easy_video_ids + hard1
    val_video_ids = hard2

    # --- 4. Write Output Files ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, 'train.txt')
    val_path = os.path.join(args.output_dir, 'val.txt')

    print(f"\nWriting {len(train_video_ids)} videos to {train_path}...")
    write_split_file(train_path, train_video_ids, video_data_map)
    
    print(f"Writing {len(val_video_ids)} videos to {val_path}...")
    write_split_file(val_path, val_video_ids, video_data_map)

    print("\n--- Split Complete! ---")
    print(f"Train set: {len(train_video_ids)} videos")
    print(f"Val set:   {len(val_video_ids)} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split RetinaFace dataset into train (easy) and val (hard) by video.")
    
    parser.add_argument("--retinaface-txt", type=str, default="../data/retinaface_full.txt",
                        help="Path to the complete 'retinaface_train.txt' file.")
    
    # --- MODIFIED ARGUMENT ---
    parser.add_argument("--hard-list-json", type=str, default="hard_frames.json",
                        help="Path to a .json file listing your 'hard' (FN/FP) image filenames.")
    
    parser.add_argument("--output-dir", type=str, default="splits",
                        help="Directory to save the new train.txt and val.txt files.")
    
    args = parser.parse_args()
    main(args)