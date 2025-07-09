#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
import pandas as pd

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def detailed_speaker_analysis(data, filename):
    """Perform detailed speaker distribution analysis."""
    print(f"\n{'='*60}")
    print(f"DETAILED SPEAKER ANALYSIS: {filename}")
    print(f"{'='*60}")
    
    # Count speakers
    speaker_counts = Counter()
    speaker_details = defaultdict(list)
    total_utterances = 0
    
    for key, value in data.items():
        if isinstance(value, dict) and 'spk_id' in value:
            speaker = value['spk_id']
            speaker_counts[speaker] += 1
            speaker_details[speaker].append({
                'file': key,
                'duration': value.get('duration', 0),
                'sentence': value.get('wrd', '').strip()
            })
            total_utterances += 1
    
    print(f"Total utterances: {total_utterances}")
    print(f"Number of unique speakers: {len(speaker_counts)}")
    
    # Create detailed table
    print(f"\n{'Speaker':<8} {'Count':<8} {'Percentage':<12} {'Min Dur':<8} {'Max Dur':<8} {'Avg Dur':<8}")
    print("-" * 60)
    
    speakers_sorted = sorted(speaker_counts.items(), key=lambda x: x[0])
    for speaker, count in speakers_sorted:
        percentage = (count / total_utterances) * 100
        durations = [item['duration'] for item in speaker_details[speaker]]
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        
        print(f"{speaker:<8} {count:<8} {percentage:<11.1f}% {min_dur:<8.2f} {max_dur:<8.2f} {avg_dur:<8.2f}")
    
    # Check for balance
    print(f"\nBALANCE ANALYSIS:")
    counts = list(speaker_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    mean_count = sum(counts) / len(counts)
    std_dev = (sum((x - mean_count) ** 2 for x in counts) / len(counts)) ** 0.5
    
    print(f"  Min utterances per speaker: {min_count}")
    print(f"  Max utterances per speaker: {max_count}")
    print(f"  Mean utterances per speaker: {mean_count:.1f}")
    print(f"  Standard deviation: {std_dev:.1f}")
    print(f"  Coefficient of variation: {(std_dev/mean_count)*100:.1f}%")
    
    if max_count - min_count <= 1:
        print(f"  ✓ Speakers are well balanced (max difference ≤ 1)")
    elif max_count - min_count <= 3:
        print(f"  ⚠ Speakers are moderately balanced (max difference ≤ 3)")
    else:
        print(f"  ✗ Speakers are NOT balanced (max difference > 3)")
    
    return speaker_counts, speaker_details

def compare_speaker_distributions(train_data, dev_data):
    """Compare speaker distributions between datasets."""
    print(f"\n{'='*60}")
    print(f"SPEAKER DISTRIBUTION COMPARISON")
    print(f"{'='*60}")
    
    train_speakers = set()
    dev_speakers = set()
    
    for key, value in train_data.items():
        if isinstance(value, dict) and 'spk_id' in value:
            train_speakers.add(value['spk_id'])
    
    for key, value in dev_data.items():
        if isinstance(value, dict) and 'spk_id' in value:
            dev_speakers.add(value['spk_id'])
    
    print(f"Train dataset speakers: {len(train_speakers)}")
    print(f"Dev dataset speakers: {len(dev_speakers)}")
    
    train_only = train_speakers - dev_speakers
    dev_only = dev_speakers - train_speakers
    common = train_speakers & dev_speakers
    
    print(f"\nSpeakers in train only: {sorted(train_only)}")
    print(f"Speakers in dev only: {sorted(dev_only)}")
    print(f"Common speakers: {len(common)}")
    
    # Detailed comparison for common speakers
    if common:
        print(f"\nDetailed comparison for common speakers:")
        print(f"{'Speaker':<8} {'Train':<8} {'Dev':<8} {'Ratio':<8}")
        print("-" * 35)
        
        for speaker in sorted(common):
            train_count = sum(1 for k, v in train_data.items() 
                            if isinstance(v, dict) and v.get('spk_id') == speaker)
            dev_count = sum(1 for k, v in dev_data.items() 
                          if isinstance(v, dict) and v.get('spk_id') == speaker)
            ratio = train_count / dev_count if dev_count > 0 else float('inf')
            print(f"{speaker:<8} {train_count:<8} {dev_count:<8} {ratio:<8.1f}")

def main():
    # File paths
    train_file = "data/train_erj_spk_open_train-train.json"
    dev_file = "data/train_erj_spk_open_train-dev.json"
    
    # Load data
    print("Loading JSON files...")
    train_data = load_json_file(train_file)
    dev_data = load_json_file(dev_file)
    
    if train_data is None or dev_data is None:
        print("Failed to load one or both files. Exiting.")
        return
    
    # Detailed analysis for each dataset
    train_speakers, train_details = detailed_speaker_analysis(train_data, "TRAIN DATASET")
    dev_speakers, dev_details = detailed_speaker_analysis(dev_data, "DEV DATASET")
    
    # Compare distributions
    compare_speaker_distributions(train_data, dev_data)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Train dataset: {len(train_data)} entries, {len(train_speakers)} speakers")
    print(f"Dev dataset: {len(dev_data)} entries, {len(dev_speakers)} speakers")
    
    train_balance = "BALANCED" if max(train_speakers.values()) - min(train_speakers.values()) <= 1 else "NOT BALANCED"
    dev_balance = "BALANCED" if max(dev_speakers.values()) - min(dev_speakers.values()) <= 1 else "NOT BALANCED"
    
    print(f"Train speaker balance: {train_balance}")
    print(f"Dev speaker balance: {dev_balance}")

if __name__ == "__main__":
    main() 