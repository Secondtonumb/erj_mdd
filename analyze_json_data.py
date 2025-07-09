#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
import os

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_speaker_distribution(data, filename):
    """Analyze speaker distribution in the dataset."""
    print(f"\n=== Speaker Distribution Analysis for {filename} ===")
    
    # Count speakers
    speaker_counts = Counter()
    total_utterances = 0
    
    for key, value in data.items():
        if isinstance(value, dict) and 'spk_id' in value:
            speaker_counts[value['spk_id']] += 1
            total_utterances += 1
    
    print(f"Total utterances: {total_utterances}")
    print(f"Number of unique speakers: {len(speaker_counts)}")
    print("\nSpeaker distribution:")
    for speaker, count in sorted(speaker_counts.items()):
        percentage = (count / total_utterances) * 100
        print(f"  {speaker}: {count} utterances ({percentage:.1f}%)")
    
    return speaker_counts

def find_duplicate_sentences(data, filename):
    """Find duplicate sentences in the dataset."""
    print(f"\n=== Duplicate Sentence Analysis for {filename} ===")
    
    sentence_counts = Counter()
    sentence_to_files = defaultdict(list)
    
    for key, value in data.items():
        if isinstance(value, dict) and 'wrd' in value:
            sentence = value['wrd'].strip()
            sentence_counts[sentence] += 1
            sentence_to_files[sentence].append(key)
    
    # Find duplicates
    duplicates = {sentence: count for sentence, count in sentence_counts.items() if count > 1}
    
    if duplicates:
        print(f"Found {len(duplicates)} unique sentences that appear multiple times:")
        for sentence, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"\n  Sentence: '{sentence.strip()}'")
            print(f"  Count: {count}")
            print(f"  Files:")
            for file_path in sentence_to_files[sentence]:
                print(f"    - {file_path}")
    else:
        print("No duplicate sentences found.")
    
    return duplicates, sentence_to_files

def compare_datasets(train_data, dev_data):
    """Compare the two datasets for overlapping sentences."""
    print(f"\n=== Cross-Dataset Comparison ===")
    
    # Extract sentences from both datasets
    train_sentences = set()
    dev_sentences = set()
    
    for key, value in train_data.items():
        if isinstance(value, dict) and 'wrd' in value:
            train_sentences.add(value['wrd'].strip())
    
    for key, value in dev_data.items():
        if isinstance(value, dict) and 'wrd' in value:
            dev_sentences.add(value['wrd'].strip())
    
    # Find overlapping sentences
    overlapping = train_sentences.intersection(dev_sentences)
    
    print(f"Train dataset unique sentences: {len(train_sentences)}")
    print(f"Dev dataset unique sentences: {len(dev_sentences)}")
    print(f"Overlapping sentences between datasets: {len(overlapping)}")
    
    if overlapping:
        print("\nOverlapping sentences:")
        for sentence in sorted(overlapping):
            print(f"  - '{sentence.strip()}'")
    
    return overlapping

def analyze_duration_distribution(data, filename):
    """Analyze duration distribution of utterances."""
    print(f"\n=== Duration Analysis for {filename} ===")
    
    durations = []
    for key, value in data.items():
        if isinstance(value, dict) and 'duration' in value:
            durations.append(value['duration'])
    
    if durations:
        durations.sort()
        print(f"Total utterances with duration: {len(durations)}")
        print(f"Min duration: {min(durations):.3f}s")
        print(f"Max duration: {max(durations):.3f}s")
        print(f"Mean duration: {sum(durations)/len(durations):.3f}s")
        print(f"Median duration: {durations[len(durations)//2]:.3f}s")
        
        # Duration ranges
        ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, float('inf'))]
        print("\nDuration distribution:")
        for start, end in ranges:
            count = sum(1 for d in durations if start <= d < end)
            percentage = (count / len(durations)) * 100
            if end == float('inf'):
                print(f"  {start}+s: {count} utterances ({percentage:.1f}%)")
            else:
                print(f"  {start}-{end}s: {count} utterances ({percentage:.1f}%)")

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
    
    # Analyze each dataset
    train_speakers = analyze_speaker_distribution(train_data, "Train Dataset")
    dev_speakers = analyze_speaker_distribution(dev_data, "Dev Dataset")
    
    # Find duplicates within each dataset
    train_duplicates, train_sentence_map = find_duplicate_sentences(train_data, "Train Dataset")
    dev_duplicates, dev_sentence_map = find_duplicate_sentences(dev_data, "Dev Dataset")
    
    # Compare datasets
    overlapping = compare_datasets(train_data, dev_data)
    
    # Analyze duration distribution
    analyze_duration_distribution(train_data, "Train Dataset")
    analyze_duration_distribution(dev_data, "Dev Dataset")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Train dataset: {len(train_data)} entries")
    print(f"Dev dataset: {len(dev_data)} entries")
    print(f"Train speakers: {list(train_speakers.keys())}")
    print(f"Dev speakers: {list(dev_speakers.keys())}")
    print(f"Train duplicates: {len(train_duplicates)}")
    print(f"Dev duplicates: {len(dev_duplicates)}")
    print(f"Cross-dataset overlaps: {len(overlapping)}")

if __name__ == "__main__":
    main() 