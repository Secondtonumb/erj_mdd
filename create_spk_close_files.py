#!/usr/bin/env python3
"""
Script to create 2 JSON files for speaker close training:
1. Get all data from unlabeled closed and remove those whose IDs are in test_erj_spk_open_test.json
2. Get the same speakers from test_erj_spk_open_test.json from train files, but remove duplicated ones
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

def get_speaker_ids_from_test(test_data):
    """Extract unique speaker IDs from test data"""
    speaker_ids = set()
    for item in test_data.values():
        if 'spk_id' in item:
            speaker_ids.add(item['spk_id'])
    return speaker_ids

def get_file_ids_from_test(test_data):
    """Extract file IDs (keys) from test data"""
    return set(test_data.keys())

def main():
    # File paths
    test_file = "data/test_erj_spk_open_test.json"
    unlabeled_close_file = "data/train_unlabeled_erj_spk_close_train.json"
    train_file = "data/train_erj_spk_open_train.json"
    
    # Load data
    print("Loading test data...")
    test_data = load_json_file(test_file)
    
    print("Loading unlabeled close data...")
    unlabeled_close_data = load_json_file(unlabeled_close_file)
    
    print("Loading train data...")
    train_data = load_json_file(train_file)
    
    if not test_data or not unlabeled_close_data or not train_data:
        print("Error: Could not load one or more required files")
        return
    
    # Get speaker IDs from test data
    test_speaker_ids = get_speaker_ids_from_test(test_data)
    print(f"Speaker IDs in test data: {sorted(test_speaker_ids)}")
    
    # Get file IDs from test data
    test_file_ids = get_file_ids_from_test(test_data)
    print(f"Number of files in test data: {len(test_file_ids)}")
    
    # Task 1: Create unlabeled close training data (remove test files)
    print("\nCreating unlabeled close training data...")
    unlabeled_close_train = {}
    removed_count = 0
    
    for file_id, item in unlabeled_close_data.items():
        if file_id not in test_file_ids:
            unlabeled_close_train[file_id] = item
        else:
            removed_count += 1
    
    print(f"Removed {removed_count} files that are in test data")
    print(f"Kept {len(unlabeled_close_train)} files for unlabeled close training")
    
    # Save unlabeled close training data
    save_json_file(unlabeled_close_train, "data/train_unlabeled_erj_spk_close_train_filtered.json")
    
    # Task 2: Create labeled close training data (same speakers as test, remove duplicates)
    print("\nCreating labeled close training data...")
    labeled_close_train = {}
    duplicate_count = 0
    
    # First, collect all files from train data that have the same speakers as test
    for file_id, item in train_data.items():
        if 'spk_id' in item and item['spk_id'] in test_speaker_ids:
            # Check if this file is not in test data and not already added
            if file_id not in test_file_ids and file_id not in labeled_close_train:
                labeled_close_train[file_id] = item
            else:
                duplicate_count += 1
    
    print(f"Found {len(labeled_close_train)} unique files with same speakers as test")
    print(f"Skipped {duplicate_count} duplicate files")
    
    # Save labeled close training data
    save_json_file(labeled_close_train, "data/train_erj_spk_close_train.json")
    
    # Print summary
    print("\nSummary:")
    print(f"1. Unlabeled close training: {len(unlabeled_close_train)} files")
    print(f"2. Labeled close training: {len(labeled_close_train)} files")
    print(f"   - Speakers: {sorted(test_speaker_ids)}")

if __name__ == "__main__":
    main() 