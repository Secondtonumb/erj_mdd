#!/usr/bin/env python3
"""
Script to create a new JSON file:
train_unlabeled_erj_spk_close_train_filtered.json - train_unlabeled_erj_spk_open_train.json
"""

import json
import os

def load_json_file(filepath):
    """Load a JSON file and return the data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def save_json_file(data, filepath):
    """Save data to a JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {filepath}")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")

def main():
    # File paths
    filtered_close_file = "data/train_unlabeled_erj_spk_close_train_filtered.json"
    open_train_file = "data/train_unlabeled_erj_spk_open_train.json"
    
    # Load data
    print("Loading filtered close data...")
    filtered_close_data = load_json_file(filtered_close_file)
    
    print("Loading open train data...")
    open_train_data = load_json_file(open_train_file)
    
    if not filtered_close_data or not open_train_data:
        print("Error: Could not load one or more required files")
        return
    
    print(f"Filtered close data: {len(filtered_close_data)} files")
    print(f"Open train data: {len(open_train_data)} files")
    
    # Get file IDs from open train data
    open_train_file_ids = set(open_train_data.keys())
    print(f"Open train file IDs: {len(open_train_file_ids)}")
    
    # Create final unlabeled data (filtered close - open train)
    print("\nCreating final unlabeled data...")
    final_unlabeled_data = {}
    removed_count = 0
    
    for file_id, item in filtered_close_data.items():
        if file_id not in open_train_file_ids:
            final_unlabeled_data[file_id] = item
        else:
            removed_count += 1
    
    print(f"Removed {removed_count} files that are in open train data")
    print(f"Kept {len(final_unlabeled_data)} files for final unlabeled data")
    
    # Save final unlabeled data
    output_file = "data/train_unlabeled_erj_spk_close_train_final.json"
    save_json_file(final_unlabeled_data, output_file)
    
    # Print summary
    print("\nSummary:")
    print(f"Final unlabeled data: {len(final_unlabeled_data)} files")
    print(f"Output file: {output_file}")
    
    # Check speaker distribution
    print("\nSpeaker distribution in final unlabeled data:")
    speaker_counts = {}
    for item in final_unlabeled_data.values():
        spk_id = item.get('spk_id', 'unknown')
        speaker_counts[spk_id] = speaker_counts.get(spk_id, 0) + 1
    
    for spk_id in sorted(speaker_counts.keys()):
        print(f"  {spk_id}: {speaker_counts[spk_id]} files")

if __name__ == "__main__":
    main() 