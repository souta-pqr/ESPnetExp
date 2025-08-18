#!/usr/bin/env python3
"""
Comprehensive disfluency detection evaluation script.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""

import sys
import argparse
from collections import defaultdict

def load_labels(file_path):
    """Load disfluency labels from file."""
    utts = {}
    total_tokens = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: Invalid line {line_num} in {file_path}: {line}", file=sys.stderr)
                continue
                
            utt_id = parts[0]
            try:
                labels = [int(x) for x in parts[1:]]
                utts[utt_id] = labels
                total_tokens += len(labels)
            except ValueError as e:
                print(f"Warning: Invalid labels in line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"Loaded {len(utts)} utterances with {total_tokens} tokens from {file_path}")
    return utts

def compute_metrics(ref_labels, hyp_labels, num_classes=4):
    """Compute detailed classification metrics."""
    # Convert to common length
    min_len = min(len(ref_labels), len(hyp_labels))
    if min_len == 0:
        return {}
    
    ref_labels = ref_labels[:min_len]
    hyp_labels = hyp_labels[:min_len]
    
    # Overall accuracy
    correct = sum(1 for r, h in zip(ref_labels, hyp_labels) if r == h)
    accuracy = correct / len(ref_labels)
    
    # Per-class metrics
    class_metrics = {}
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    
    # Build confusion matrix
    for r, h in zip(ref_labels, hyp_labels):
        if 0 <= r < num_classes and 0 <= h < num_classes:
            confusion_matrix[r][h] += 1
    
    # Compute per-class precision, recall, F1
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[i][cls] for i in range(num_classes)) - tp
        fn = sum(confusion_matrix[cls][i] for i in range(num_classes)) - tp
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix,
        'total_samples': len(ref_labels)
    }

def evaluate_disfluency(ref_file, hyp_file, output_file):
    """Main evaluation function."""
    print(f"Evaluating disfluency detection:")
    print(f"  Reference: {ref_file}")
    print(f"  Hypothesis: {hyp_file}")
    print(f"  Output: {output_file}")
    
    # Load data
    ref_data = load_labels(ref_file)
    hyp_data = load_labels(hyp_file)
    
    if len(ref_data) == 0:
        print("Error: No reference data found!")
        return
    
    if len(hyp_data) == 0:
        print("Error: No hypothesis data found!")
        return
    
    # Collect labels for evaluation
    all_ref_labels = []
    all_hyp_labels = []
    utt_metrics = {}
    
    matched_utts = 0
    for utt_id in ref_data:
        if utt_id in hyp_data:
            ref_seq = ref_data[utt_id]
            hyp_seq = hyp_data[utt_id]
            
            # Compute per-utterance metrics
            utt_metrics[utt_id] = compute_metrics(ref_seq, hyp_seq)
            
            # Add to global lists
            min_len = min(len(ref_seq), len(hyp_seq))
            all_ref_labels.extend(ref_seq[:min_len])
            all_hyp_labels.extend(hyp_seq[:min_len])
            matched_utts += 1
    
    if len(all_ref_labels) == 0:
        print("Error: No matching utterances found!")
        return
    
    print(f"Matched {matched_utts}/{len(ref_data)} utterances")
    
    # Compute overall metrics
    overall_metrics = compute_metrics(all_ref_labels, all_hyp_labels)
    
    # Class names
    class_names = ['Fluent', 'Filler', 'Repetition', 'Interjection']
    
    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Disfluency Detection Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Total utterances (ref): {len(ref_data)}\n")
        f.write(f"  Total utterances (hyp): {len(hyp_data)}\n")
        f.write(f"  Matched utterances: {matched_utts}\n")
        f.write(f"  Total tokens evaluated: {overall_metrics['total_samples']}\n\n")
        
        # Overall accuracy
        f.write(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}\n\n")
        
        # Per-class detailed results
        f.write("Per-Class Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        
        macro_f1 = 0
        weighted_f1 = 0
        total_support = 0
        
        for cls in range(4):
            if cls in overall_metrics['class_metrics']:
                metrics = overall_metrics['class_metrics'][cls]
                class_name = class_names[cls] if cls < len(class_names) else f"Class{cls}"
                
                f.write(f"{class_name:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1']:<10.4f} {metrics['support']:<10}\n")
                
                macro_f1 += metrics['f1']
                weighted_f1 += metrics['f1'] * metrics['support']
                total_support += metrics['support']
        
        macro_f1 /= 4
        weighted_f1 /= total_support if total_support > 0 else 1
        
        f.write("-" * 70 + "\n")
        f.write(f"{'Macro avg':<12} {'':<10} {'':<10} {macro_f1:<10.4f} {total_support:<10}\n")
        f.write(f"{'Weighted avg':<12} {'':<10} {'':<10} {weighted_f1:<10.4f} {total_support:<10}\n")
        f.write("\n")
        
        # Confusion Matrix
        f.write("Confusion Matrix:\n")
        f.write("-" * 50 + "\n")
        f.write("Predicted ->   ")
        for i, name in enumerate(class_names):
            f.write(f"{i}({name[:3]})  ")
        f.write("\n")
        f.write("Actual ↓\n")
        
        cm = overall_metrics['confusion_matrix']
        for i, name in enumerate(class_names):
            f.write(f"{i}({name[:8]:<8})  ")
            for j in range(4):
                f.write(f"{cm[i][j]:<8} ")
            f.write("\n")
        f.write("\n")
        
        # Label distribution
        f.write("Label Distribution:\n")
        f.write("-" * 30 + "\n")
        ref_counts = [0] * 4
        hyp_counts = [0] * 4
        
        for label in all_ref_labels:
            if 0 <= label < 4:
                ref_counts[label] += 1
        
        for label in all_hyp_labels:
            if 0 <= label < 4:
                hyp_counts[label] += 1
        
        for i, name in enumerate(class_names):
            f.write(f"{name}: Ref={ref_counts[i]} ({ref_counts[i]/len(all_ref_labels)*100:.1f}%), "
                   f"Hyp={hyp_counts[i]} ({hyp_counts[i]/len(all_hyp_labels)*100:.1f}%)\n")
    
    # Print summary to console
    print(f"\nEvaluation completed successfully!")
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate disfluency detection performance")
    parser.add_argument("--ref", required=True, help="Reference disfluency file")
    parser.add_argument("--hyp", required=True, help="Hypothesis disfluency file")
    parser.add_argument("--output", required=True, help="Output evaluation file")
    
    args = parser.parse_args()
    evaluate_disfluency(args.ref, args.hyp, args.output)
