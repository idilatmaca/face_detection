import json
import os
import argparse
from tqdm import tqdm
import sys

def convert_json_to_retinaface(json_path, output_txt_path, image_prefix):
    """
    Converts the final combined JSON into the RetinaFace .txt training format.
    """
    
    # 1. Load the ground truth JSON
    try:
        with open(json_path, 'r') as f:
            ground_truths = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'.")
        sys.exit(1)

    if not ground_truths:
        print("Error: The JSON file is empty.")
        return

    image_count = 0
    face_count = 0

    # 2. Open the output .txt file
    with open(output_txt_path, 'w') as f_out:
        print(f"Starting conversion... Writing to {output_txt_path}")
        
        # Loop through each image in the JSON
        for image_filename, data in tqdm(ground_truths.items(), desc="Processing images"):
            
            # Get all face keys (e.g., "Face-1", "Face-2")
            face_keys = [k for k in data.keys() if k.startswith('Face-')]
            
            # If there are no faces, skip this image (RetinaFace can't train on negatives)
            if not face_keys:
                continue
            
            # --- Write the image path header ---
            # We must handle files with semicolons that we fixed earlier
            image_filename_fixed =image_filename # image_filename.replace(";", "_")
            
            # Add the image prefix to create the relative path
            full_image_path = os.path.join(image_prefix, image_filename_fixed)
            # Use forward slashes, as is standard in most data files
            f_out.write(f"# {full_image_path.replace(os.sep, '/')}\n")

            # Sort keys to ensure "Face-1" comes before "Face-10"
            face_keys.sort(key=lambda x: int(x.split('-')[-1]))

            # Loop through each face in the image
            for face_key in face_keys:
                face_data = data[face_key]
                bbox = face_data.get('bbox')
                landmarks = face_data.get('landmarks')

                # Ensure we have valid data
                if not bbox or not landmarks or len(landmarks) < 5:
                    print(f"\nWarning: Skipping {face_key} in {image_filename} (missing data).")
                    continue
                
                # --- Format the Bounding Box ---
                # Input is [x1, y1, x2, y2]
                # Format as "x1 y1 x2 y2" with 2 decimal places
                bbox_str = " ".join([f"{coord:.2f}" for coord in bbox])
                
                # --- Format the Landmarks ---
                # Input is [[lx1, ly1, v1], [lx2, ly2, v2], ...]
                # Output needs to be "lx1 ly1 lx2 ly2 ..."
                landmarks_flat = []
                for lm_point in landmarks:
                    landmarks_flat.append(lm_point[0]) # lx
                    landmarks_flat.append(lm_point[1]) # ly
                
                # Format as "lx1 ly1 ... lx5 ly5" with 2 decimal places
                landmarks_str = " ".join([f"{coord:.2f}" for coord in landmarks_flat])

                # --- Write the final line ---
                f_out.write(f"{bbox_str} {landmarks_str}\n")
                face_count += 1
            
            image_count += 1

    print("\n--- Conversion Complete ---")
    print(f"Successfully processed {image_count} images.")
    print(f"Wrote {face_count} total face annotations.")
    print(f"Output file saved to: {output_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert combined JSON to RetinaFace .txt format.")
    
    parser.add_argument("--json-path", type=str, 
                        default="../data/FINAL_combined_output.json",
                        help="Path to the final combined JSON file.")
    
    parser.add_argument("--output-txt", type=str, 
                        default="../data/retinaface_full.txt",
                        help="Path to save the output 'train.txt' file.")
    
    parser.add_argument("--image-prefix", type=str, 
                        default="/media/syn3090/16c65c3c-0381-482a-a5df-5f99340fac70/deneme/light_resnet/All/",
                        help="The relative path prefix to add to each image filename. "
                             "Adjust this to match your training folder structure.")
    
    args = parser.parse_args()

    # Add a trailing slash to prefix if it's not empty and doesn't have one
    if args.image_prefix and not args.image_prefix.endswith(('/', '\\')):
        args.image_prefix += os.sep

    convert_json_to_retinaface(args.json_path, args.output_txt, args.image_prefix)

if __name__ == "__main__":
    main()