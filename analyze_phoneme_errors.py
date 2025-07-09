#!/usr/bin/env python3
import re
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_mpd_file(file_path):
    """Parse the MPD results file to extract phoneme error patterns."""
    
    # Data structures to store error patterns
    canonical_perceived_errors = Counter()  # Canonical vs Perceived errors
    canonical_hypothesis_errors = Counter()  # Canonical vs Hypothesis errors
    human_annotation_errors = []  # Store full human annotations
    model_prediction_errors = []  # Store full model predictions
    
    current_file = None
    current_section = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and headers
        if not line or line.startswith('MPD results') or line.startswith('Overall MPD'):
            continue
            
        # Check if this is a file path
        if line.startswith('/common/db/'):
            current_file = line
            current_section = None
            continue
            
        # Check for section headers
        if 'Human annotation: Canonical vs Perceived:' in line:
            current_section = 'human_canonical_perceived'
            continue
        elif 'Model Prediction: Canonical vs Hypothesis:' in line:
            current_section = 'model_canonical_hypothesis'
            continue
        elif line.startswith('True Accept:') or line.startswith('='):
            current_section = None
            continue
            
        # Parse phoneme alignments
        if current_section and ';' in line:
            parts = [p.strip() for p in line.split(';')]
            
            if current_section == 'human_canonical_perceived':
                if len(parts) >= 3:
                    canonical = parts[0]
                    operation = parts[1]
                    perceived = parts[2]
                    
                    if operation in ['S', 'D', 'I']:  # Substitution, Deletion, Insertion
                        canonical_perceived_errors[(canonical, perceived, operation)] += 1
                        
                    human_annotation_errors.append({
                        'file': current_file,
                        'canonical': canonical,
                        'operation': operation,
                        'perceived': perceived
                    })
                    
            elif current_section == 'model_canonical_hypothesis':
                if len(parts) >= 3:
                    canonical = parts[0]
                    operation = parts[1]
                    hypothesis = parts[2]
                    
                    if operation in ['S', 'D', 'I']:
                        canonical_hypothesis_errors[(canonical, hypothesis, operation)] += 1
                        
                    model_prediction_errors.append({
                        'file': current_file,
                        'canonical': canonical,
                        'operation': operation,
                        'hypothesis': hypothesis
                    })
    
    return canonical_perceived_errors, canonical_hypothesis_errors, human_annotation_errors, model_prediction_errors

def analyze_error_patterns(canonical_perceived_errors, canonical_hypothesis_errors):
    """Analyze the most frequent error patterns."""
    
    print("=" * 80)
    print("PHONEME ERROR ANALYSIS IN JAPANESE ENGLISH SPEECH")
    print("=" * 80)
    
    # 1. Most frequent Canonical vs Perceived errors (Human annotation)
    print("\n1. MOST FREQUENT CANONICAL vs PERCEIVED ERRORS (Human Annotation)")
    print("-" * 60)
    
    # Group by operation type
    substitutions = [(k, v) for k, v in canonical_perceived_errors.items() if k[2] == 'S']
    deletions = [(k, v) for k, v in canonical_perceived_errors.items() if k[2] == 'D']
    insertions = [(k, v) for k, v in canonical_perceived_errors.items() if k[2] == 'I']
    
    print(f"Total Substitutions: {len(substitutions)}")
    print(f"Total Deletions: {len(deletions)}")
    print(f"Total Insertions: {len(insertions)}")
    
    print("\nTop 20 Substitutions (Canonical → Perceived):")
    for (canonical, perceived, op), count in sorted(substitutions, key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {canonical:>3} → {perceived:<3} : {count:>3} times")
    
    print("\nTop 10 Deletions (Canonical → <eps>):")
    for (canonical, perceived, op), count in sorted(deletions, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {canonical:>3} → <eps> : {count:>3} times")
    
    print("\nTop 10 Insertions (<eps> → Perceived):")
    for (canonical, perceived, op), count in sorted(insertions, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  <eps> → {perceived:<3} : {count:>3} times")
    
    # 2. Most frequent Canonical vs Hypothesis errors (Model prediction)
    print("\n\n2. MOST FREQUENT CANONICAL vs HYPOTHESIS ERRORS (Model Prediction)")
    print("-" * 60)
    
    # Group by operation type
    model_substitutions = [(k, v) for k, v in canonical_hypothesis_errors.items() if k[2] == 'S']
    model_deletions = [(k, v) for k, v in canonical_hypothesis_errors.items() if k[2] == 'D']
    model_insertions = [(k, v) for k, v in canonical_hypothesis_errors.items() if k[2] == 'I']
    
    print(f"Total Substitutions: {len(model_substitutions)}")
    print(f"Total Deletions: {len(model_deletions)}")
    print(f"Total Insertions: {len(model_insertions)}")
    
    print("\nTop 20 Substitutions (Canonical → Hypothesis):")
    for (canonical, hypothesis, op), count in sorted(model_substitutions, key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {canonical:>3} → {hypothesis:<3} : {count:>3} times")
    
    print("\nTop 10 Deletions (Canonical → <eps>):")
    for (canonical, hypothesis, op), count in sorted(model_deletions, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {canonical:>3} → <eps> : {count:>3} times")
    
    print("\nTop 10 Insertions (<eps> → Hypothesis):")
    for (canonical, hypothesis, op), count in sorted(model_insertions, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  <eps> → {hypothesis:<3} : {count:>3} times")
    
    return substitutions, deletions, insertions, model_substitutions, model_deletions, model_insertions

def analyze_phoneme_clusters():
    """Analyze phoneme clusters and their error patterns."""
    
    # Define phoneme clusters
    vowel_clusters = {
        'front_vowels': ['iy', 'ih', 'ey', 'eh', 'ae'],
        'back_vowels': ['uw', 'uh', 'ow', 'ao', 'aa'],
        'central_vowels': ['ah', 'er', 'ax', 'ay', 'aw', 'oy']
    }
    
    consonant_clusters = {
        'stops': ['p', 'b', 't', 'd', 'k', 'g'],
        'fricatives': ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh'],
        'affricates': ['ch', 'jh'],
        'nasals': ['m', 'n', 'ng'],
        'liquids': ['l', 'r'],
        'glides': ['w', 'y']
    }
    
    print("\n\n3. PHONEME CLUSTER ANALYSIS")
    print("-" * 60)
    
    return vowel_clusters, consonant_clusters

def compare_human_model_errors(human_errors, model_errors):
    """Compare human annotation errors vs model prediction errors."""
    
    print("\n\n4. COMPARISON: HUMAN vs MODEL ERROR DETECTION")
    print("-" * 60)
    
    # Create sets of error patterns
    human_error_patterns = set(human_errors.keys())
    model_error_patterns = set(model_errors.keys())
    
    # Find common errors
    common_errors = human_error_patterns & model_error_patterns
    human_only = human_error_patterns - model_error_patterns
    model_only = model_error_patterns - human_error_patterns
    
    print(f"Total unique human error patterns: {len(human_error_patterns)}")
    print(f"Total unique model error patterns: {len(model_error_patterns)}")
    print(f"Common error patterns: {len(common_errors)}")
    print(f"Human-only error patterns: {len(human_only)}")
    print(f"Model-only error patterns: {len(model_only)}")
    
    print("\nTop 10 Common Error Patterns (Human vs Model):")
    common_errors_list = [(k, human_errors[k], model_errors[k]) for k in common_errors]
    common_errors_list.sort(key=lambda x: x[1] + x[2], reverse=True)
    
    for (canonical, target, op), human_count, model_count in common_errors_list[:10]:
        print(f"  {canonical:>3} → {target:<3} ({op}): Human={human_count:>2}, Model={model_count:>2}")
    
    print("\nTop 10 Human-Only Error Patterns (Model missed):")
    human_only_list = [(k, human_errors[k]) for k in human_only]
    human_only_list.sort(key=lambda x: x[1], reverse=True)
    
    for (canonical, target, op), count in human_only_list[:10]:
        print(f"  {canonical:>3} → {target:<3} ({op}): {count:>2} times (Model missed)")
    
    print("\nTop 10 Model-Only Error Patterns (False positives):")
    model_only_list = [(k, model_errors[k]) for k in model_only]
    model_only_list.sort(key=lambda x: x[1], reverse=True)
    
    for (canonical, target, op), count in model_only_list[:10]:
        print(f"  {canonical:>3} → {target:<3} ({op}): {count:>2} times (False positive)")
    
    return common_errors, human_only, model_only

def analyze_difficult_errors(human_errors, model_errors):
    """Analyze which types of errors are most difficult to detect."""
    
    print("\n\n5. DIFFICULT ERROR ANALYSIS")
    print("-" * 60)
    
    # Find errors that are frequent in human annotation but rare in model prediction
    difficult_errors = []
    
    for (canonical, target, op), human_count in human_errors.items():
        model_count = model_errors.get((canonical, target, op), 0)
        if human_count >= 3 and model_count < human_count * 0.5:  # Model detects less than 50%
            difficult_errors.append((canonical, target, op, human_count, model_count))
    
    difficult_errors.sort(key=lambda x: x[3], reverse=True)  # Sort by human frequency
    
    print("Errors that are difficult for the model to detect (Human freq ≥ 3, Model < 50% of Human):")
    for canonical, target, op, human_count, model_count in difficult_errors[:20]:
        detection_rate = (model_count / human_count) * 100 if human_count > 0 else 0
        print(f"  {canonical:>3} → {target:<3} ({op}): Human={human_count:>2}, Model={model_count:>2} ({detection_rate:>5.1f}%)")
    
    return difficult_errors

def generate_summary_report(canonical_perceived_errors, canonical_hypothesis_errors):
    """Generate a comprehensive summary report."""
    
    print("\n\n" + "=" * 80)
    print("SUMMARY REPORT: JAPANESE ENGLISH PHONEME ERROR PATTERNS")
    print("=" * 80)
    
    # Overall statistics
    total_human_errors = sum(canonical_perceived_errors.values())
    total_model_errors = sum(canonical_hypothesis_errors.values())
    
    print(f"\nOverall Statistics:")
    print(f"  Total human-annotated errors: {total_human_errors}")
    print(f"  Total model-predicted errors: {total_model_errors}")
    
    # Most problematic phonemes
    human_canonical_freq = Counter()
    for (canonical, _, _), count in canonical_perceived_errors.items():
        human_canonical_freq[canonical] += count
    
    print(f"\nMost problematic phonemes (by human annotation):")
    for phoneme, count in human_canonical_freq.most_common(10):
        print(f"  {phoneme}: {count} errors")
    
    # Error type distribution
    human_subs = sum(1 for (_, _, op) in canonical_perceived_errors.keys() if op == 'S')
    human_dels = sum(1 for (_, _, op) in canonical_perceived_errors.keys() if op == 'D')
    human_ins = sum(1 for (_, _, op) in canonical_perceived_errors.keys() if op == 'I')
    
    print(f"\nHuman error type distribution:")
    print(f"  Substitutions: {human_subs}")
    print(f"  Deletions: {human_dels}")
    print(f"  Insertions: {human_ins}")
    
    # Key findings
    print(f"\nKey Findings:")
    print(f"  1. Most frequent substitution patterns in Japanese English")
    print(f"  2. Phonemes that are commonly deleted/inserted")
    print(f"  3. Model's ability to detect different error types")
    print(f"  4. Systematic pronunciation patterns in Japanese English")

def main():
    # Parse the MPD file
    mpd_file = "results/erj_spk_open/mpd.txt"
    
    print("Parsing MPD results file...")
    canonical_perceived_errors, canonical_hypothesis_errors, human_annotations, model_predictions = parse_mpd_file(mpd_file)
    
    # Analyze error patterns
    human_subs, human_dels, human_ins, model_subs, model_dels, model_ins = analyze_error_patterns(
        canonical_perceived_errors, canonical_hypothesis_errors
    )
    
    # Analyze phoneme clusters
    vowel_clusters, consonant_clusters = analyze_phoneme_clusters()
    
    # Compare human vs model errors
    common_errors, human_only, model_only = compare_human_model_errors(
        canonical_perceived_errors, canonical_hypothesis_errors
    )
    
    # Analyze difficult errors
    difficult_errors = analyze_difficult_errors(canonical_perceived_errors, canonical_hypothesis_errors)
    
    # Generate summary report
    generate_summary_report(canonical_perceived_errors, canonical_hypothesis_errors)
    
    # Save detailed results to files
    print(f"\nSaving detailed analysis to files...")
    
    # Save human error patterns
    with open("human_error_patterns.txt", "w") as f:
        f.write("Human Annotation Error Patterns\n")
        f.write("=" * 40 + "\n\n")
        for (canonical, perceived, op), count in canonical_perceived_errors.most_common():
            f.write(f"{canonical} → {perceived} ({op}): {count}\n")
    
    # Save model error patterns
    with open("model_error_patterns.txt", "w") as f:
        f.write("Model Prediction Error Patterns\n")
        f.write("=" * 40 + "\n\n")
        for (canonical, hypothesis, op), count in canonical_hypothesis_errors.most_common():
            f.write(f"{canonical} → {hypothesis} ({op}): {count}\n")
    
    # Save difficult errors
    with open("difficult_errors.txt", "w") as f:
        f.write("Difficult Errors for Model Detection\n")
        f.write("=" * 40 + "\n\n")
        for canonical, target, op, human_count, model_count in difficult_errors:
            detection_rate = (model_count / human_count) * 100 if human_count > 0 else 0
            f.write(f"{canonical} → {target} ({op}): Human={human_count}, Model={model_count} ({detection_rate:.1f}%)\n")
    
    print(f"Analysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 