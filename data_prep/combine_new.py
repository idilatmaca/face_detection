import json
import os

# --- 1. Define Paths ---

# This is the new JSON with your corrected landmark data
short_json_path = "/home/syntonym4090/idil/telus/cloud-backbone/ground_truth_with_landmarks.json"

# This is your main JSON file that you want to update
main_json_path = "/home/syntonym4090/idil/telus/cloud-backbone/combined_output.json"

# This will be the new, merged file
output_json_path = "/home/syntonym4090/idil/telus/cloud-backbone/FINAL_combined_output.json"

# --- 2. Load Both JSON Files ---

print(f"Loading main JSON from: {main_json_path}")
try:
    with open(main_json_path, "r") as f:
        main_data = json.load(f)
    print(f"Loaded {len(main_data)} total entries from main JSON.")
except FileNotFoundError:
    print(f"Error: Main JSON not found at {main_json_path}")
    main_data = {} # Start with an empty dict if it doesn't exist
except json.JSONDecodeError:
    print(f"Error: Main JSON at {main_json_path} is corrupted. Stopping.")
    exit()


print(f"Loading corrected JSON from: {short_json_path}")
try:
    with open(short_json_path, "r") as f:
        short_data = json.load(f)
    print(f"Loaded {len(short_data)} corrected entries to apply.")
except FileNotFoundError:
    print(f"Error: Corrected JSON not found at {short_json_path}. Stopping.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Corrected JSON at {short_json_path} is corrupted. Stopping.")
    exit()

# --- 3. Merge the Data ---

# We will loop through the corrected data and update the main data.
# This will overwrite any existing keys and add any new ones.

print("\nStarting merge...")
updated_count = 0
added_count = 0

# 'short_data' is a dict: {"filename1": {...}, "filename2": {...}}
for filename, corrected_value in short_data.items():
    if filename in main_data:
        updated_count += 1
    else:
        added_count += 1
    
    # This is the merge: it overwrites the old value or adds a new one.
    main_data[filename] = corrected_value

print("Merge complete.")
print(f"  > {updated_count} entries were UPDATED.")
print(f"  > {added_count} entries were ADDED.")
print(f"  > Total entries in final data: {len(main_data)}")

# --- 4. Save the Final Merged File ---

print(f"\nSaving final merged data to: {output_json_path}")
try:
    with open(output_json_path, "w") as f:
        json.dump(main_data, f, indent=2)
    print("Successfully saved!")
except Exception as e:
    print(f"Error saving final file: {e}")