#!/usr/bin/env python3
"""
Script to check all JSON files in data directory for test speakers
"""

import json
import os
import glob

def load_json_file(filepath):
    """Load a JSON file and return the data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

def get_speaker_ids(data):
    """Extract unique speaker IDs from data"""
    speaker_ids = set()
    for item in data.values():
        if 'spk_id' in item:
            speaker_ids.add(item['spk_id'])
    return speaker_ids

def main():
    # Get all JSON files in data directory
    json_files = glob.glob("data/*.json")
    
    # Load test data to get target speakers
    test_data = load_json_file("data/test_erj_spk_open_test.json")
    test_speaker_ids = get_speaker_ids(test_data)
    print(f"Target speakers (from test): {sorted(test_speaker_ids)}")
    print()
    
    files_with_test_speakers = []
    
    for json_file in sorted(json_files):
        print(f"Checking {json_file}...")
        data = load_json_file(json_file)
        if data and isinstance(data, dict):
            speaker_ids = get_speaker_ids(data)
            if speaker_ids:
                # Check overlap with test speakers
                overlap = speaker_ids.intersection(test_speaker_ids)
                if overlap:
                    files_with_test_speakers.append(json_file)
                    print(f"  ✓ Found test speakers: {sorted(overlap)}")
                    
                    # Count files for overlapping speakers
                    overlap_count = 0
                    for item in data.values():
                        if 'spk_id' in item and item['spk_id'] in test_speaker_ids:
                            overlap_count += 1
                    print(f"  Files for test speakers: {overlap_count}")
                else:
                    print(f"  ✗ No test speakers found")
            else:
                print(f"  ✗ No speaker IDs found")
        else:
            print(f"  ✗ Could not load or invalid format")
        print()
    
    print("=" * 50)
    print("Files containing test speakers:")
    for file in files_with_test_speakers:
        print(f"  - {file}")

if __name__ == "__main__":
    main() 