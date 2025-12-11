"""
Simple script to inspect labels from tgbn-genre dataset
Run this in Colab to understand the label structure
"""
import numpy as np
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

# Load the dataset
print("Loading tgbn-genre dataset...")
dataset = PyGNodePropPredDataset(name="tgbn-genre", root="datasets")

# Get the label dictionary
label_dict = dataset.dataset.label_dict
num_classes = dataset.num_classes

print(f"\n=== Dataset Info ===")
print(f"Number of classes (genres): {num_classes}")
print(f"Number of timestamps with labels: {len(label_dict)}")

# Count total labeled nodes
total_labels = sum(len(label_dict[ts]) for ts in label_dict.keys())
print(f"Total labeled nodes (across all timestamps): {total_labels}")

# Examine first few labels
print(f"\n=== Examining First 10 Labels ===")
count = 0
for timestamp in sorted(label_dict.keys())[:5]:  # Look at first 5 timestamps
    for node_id, label in label_dict[timestamp].items():
        if count >= 10:
            break

        # Convert to numpy array
        label_array = np.array(label)

        # Find which genres are active (non-zero)
        active_genres = np.where(label_array > 0)[0]

        print(f"\nLabel #{count + 1}:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Node ID: {node_id}")
        print(f"  Label shape: {label_array.shape}")
        print(f"  Number of active genres: {len(active_genres)}")
        print(f"  Active genre indices: {active_genres[:20]}...")  # Show first 20
        print(f"  Label values at active positions: {label_array[active_genres][:20]}")
        print(f"  Sum of label: {label_array.sum():.2f}")

        count += 1

    if count >= 10:
        break

# Statistics across all labels
print(f"\n=== Label Statistics Across Dataset ===")
num_active_per_label = []
label_sums = []

for timestamp in label_dict.keys():
    for node_id, label in label_dict[timestamp].items():
        label_array = np.array(label)
        num_active = (label_array > 0).sum()
        num_active_per_label.append(num_active)
        label_sums.append(label_array.sum())

print(f"Average number of active genres per label: {np.mean(num_active_per_label):.2f}")
print(f"Min number of active genres: {np.min(num_active_per_label)}")
print(f"Max number of active genres: {np.max(num_active_per_label)}")
print(f"Median number of active genres: {np.median(num_active_per_label):.0f}")
print(f"\nAverage sum of label values: {np.mean(label_sums):.2f}")
print(f"Label sum range: [{np.min(label_sums):.2f}, {np.max(label_sums):.2f}]")

# Distribution of number of active genres
print(f"\n=== Distribution of Active Genres ===")
unique, counts = np.unique(num_active_per_label, return_counts=True)
for num_genres, count in zip(unique[:10], counts[:10]):  # Show first 10
    print(f"  {num_genres} active genres: {count} labels ({count/len(num_active_per_label)*100:.1f}%)")

print("\n=== How NDCG@10 Works ===")
print("NDCG@10 means:")
print("  - Model predicts probabilities for all 513 genres")
print("  - We take the TOP 10 predicted genres (highest probabilities)")
print("  - We compare these top 10 to the TRUE labels")
print("  - NDCG measures how well the ranking matches (higher score = better)")
print("\nExample:")
print("  True labels: Genres [5, 12, 89] are active")
print("  Model predicts top 10: [5, 89, 12, 100, 3, ...]")
print("  NDCG@10 would be high because all 3 true genres are in top 10")
print("  and they're ranked highly (positions 1, 2, 3)")
