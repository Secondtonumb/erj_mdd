#!/usr/bin/env python3
"""
Script to check what speakers are available in different train files
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

def get_speaker_ids(data):
    """Extract unique speaker IDs from data"""
    speaker_ids = set()
    for item in data.values():
        if 'spk_id' in item:
            speaker_ids.add(item['spk_id'])
    return speaker_ids

def main():
    # Check different train files
    train_files = [
        "data/train_erj_spk_open_train.json",
        "data/train_erj.json", 
        "data/train_erj-train.json",
        "data/train_erj_new.json",
        "data/train_erj-train_new.json"
    ]
    
    # Load test data to get target speakers
    test_data = load_json_file("data/test_erj_spk_open_test.json")
    test_speaker_ids = get_speaker_ids(test_data)
    print(f"Target speakers (from test): {sorted(test_speaker_ids)}")
    print()
    
    for train_file in train_files:
        if os.path.exists(train_file):
            print(f"Checking {train_file}...")
            train_data = load_json_file(train_file)
            if train_data:
                train_speaker_ids = get_speaker_ids(train_data)
                print(f"  Total speakers: {len(train_speaker_ids)}")
                print(f"  Speakers: {sorted(train_speaker_ids)}")
                
                # Check overlap with test speakers
                overlap = train_speaker_ids.intersection(test_speaker_ids)
                print(f"  Overlap with test speakers: {sorted(overlap)}")
                print(f"  Number of overlapping speakers: {len(overlap)}")
                
                # Count files for overlapping speakers
                overlap_count = 0
                for item in train_data.values():
                    if 'spk_id' in item and item['spk_id'] in test_speaker_ids:
                        overlap_count += 1
                print(f"  Files for overlapping speakers: {overlap_count}")
                print()
            else:
                print(f"  Could not load file")
                print()
        else:
            print(f"File not found: {train_file}")
            print()

if __name__ == "__main__":
    main() 