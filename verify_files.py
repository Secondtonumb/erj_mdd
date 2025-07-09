#!/usr/bin/env python3
"""
Script to verify the created files meet the requirements
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

def get_file_ids(data):
    """Extract file IDs (keys) from data"""
    return set(data.keys())

def main():
    # Load all relevant files
    print("Loading files for verification...")
    
    test_data = load_json_file("data/test_erj_spk_open_test.json")
    unlabeled_filtered = load_json_file("data/train_unlabeled_erj_spk_close_train_filtered.json")
    labeled_close = load_json_file("data/train_erj_spk_close_train.json")
    
    if not test_data or not unlabeled_filtered or not labeled_close:
        print("Error: Could not load one or more files")
        return
    
    # Get test information
    test_speaker_ids = get_speaker_ids(test_data)
    test_file_ids = get_file_ids(test_data)
    
    print(f"Test file: {len(test_data)} files, speakers: {sorted(test_speaker_ids)}")
    print(f"Unlabeled filtered: {len(unlabeled_filtered)} files")
    print(f"Labeled close: {len(labeled_close)} files")
    print()
    
    # Verification 1: Check that unlabeled filtered doesn't contain test files
    print("Verification 1: Unlabeled filtered should not contain test files")
    overlap = get_file_ids(unlabeled_filtered).intersection(test_file_ids)
    if len(overlap) == 0:
        print("✓ PASS: No test files found in unlabeled filtered data")
    else:
        print(f"✗ FAIL: Found {len(overlap)} test files in unlabeled filtered data")
    print()
    
    # Verification 2: Check that labeled close contains only test speakers
    print("Verification 2: Labeled close should contain only test speakers")
    labeled_speaker_ids = get_speaker_ids(labeled_close)
    if labeled_speaker_ids.issubset(test_speaker_ids):
        print("✓ PASS: Labeled close contains only test speakers")
        print(f"  Speakers in labeled close: {sorted(labeled_speaker_ids)}")
    else:
        print("✗ FAIL: Labeled close contains non-test speakers")
        print(f"  Speakers in labeled close: {sorted(labeled_speaker_ids)}")
        print(f"  Non-test speakers: {sorted(labeled_speaker_ids - test_speaker_ids)}")
    print()
    
    # Verification 3: Check that labeled close doesn't contain test files
    print("Verification 3: Labeled close should not contain test files")
    labeled_file_ids = get_file_ids(labeled_close)
    overlap = labeled_file_ids.intersection(test_file_ids)
    if len(overlap) == 0:
        print("✓ PASS: No test files found in labeled close data")
    else:
        print(f"✗ FAIL: Found {len(overlap)} test files in labeled close data")
    print()
    
    # Verification 4: Check speaker distribution in labeled close
    print("Verification 4: Speaker distribution in labeled close")
    speaker_counts = {}
    for item in labeled_close.values():
        spk_id = item.get('spk_id', 'unknown')
        speaker_counts[spk_id] = speaker_counts.get(spk_id, 0) + 1
    
    for spk_id in sorted(speaker_counts.keys()):
        print(f"  {spk_id}: {speaker_counts[spk_id]} files")
    
    # Verification 5: Check that all files have required fields
    print("\nVerification 5: Checking required fields")
    
    # Check unlabeled filtered
    unlabeled_missing_fields = 0
    for file_id, item in unlabeled_filtered.items():
        if 'wav' not in item or 'spk_id' not in item:
            unlabeled_missing_fields += 1
    
    if unlabeled_missing_fields == 0:
        print("✓ PASS: All unlabeled filtered files have required fields")
    else:
        print(f"✗ FAIL: {unlabeled_missing_fields} unlabeled filtered files missing required fields")
    
    # Check labeled close
    labeled_missing_fields = 0
    for file_id, item in labeled_close.items():
        if 'wav' not in item or 'spk_id' not in item:
            labeled_missing_fields += 1
    
    if labeled_missing_fields == 0:
        print("✓ PASS: All labeled close files have required fields")
    else:
        print(f"✗ FAIL: {labeled_missing_fields} labeled close files missing required fields")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main() 