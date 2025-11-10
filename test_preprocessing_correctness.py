import numpy as np
from collections import defaultdict

# Test data
src_node_ids = np.array([5, 5, 5, 7, 7, 5])
node_interact_times = np.array([100, 200, 300, 150, 250, 400])
edge_ids = np.array([0, 1, 2, 3, 4, 5])

# Labels
labels = {
    (250, 5): "ClassA",  # User 5 at t=250 -> should map to edge 1 (t=200)
    (400, 5): "ClassB",  # User 5 at t=400 -> should map to edge 5 (t=400)
    (200, 7): "ClassC",  # User 7 at t=200 -> should map to edge 3 (t=150)
}

print("=== OLD METHOD ===")
labeled_node_interaction_indices_old = {}
for (node_label_time, src_node_id) in labels.keys():
    # OLD: Scan all edges
    nodes_historical_interactions_mask = (src_node_ids == src_node_id) & (node_interact_times <= node_label_time)
    if len(edge_ids[nodes_historical_interactions_mask]) > 0:
        nodes_most_recent_interaction_idx = edge_ids[nodes_historical_interactions_mask][-1]
    else:
        nodes_most_recent_interaction_idx = 0
    labeled_node_interaction_indices_old[(node_label_time, src_node_id)] = nodes_most_recent_interaction_idx
    print(f"Label at ({src_node_id}, t={node_label_time}) -> Edge {nodes_most_recent_interaction_idx}")

print("\n=== NEW METHOD (OPTIMIZED) ===")
labeled_node_interaction_indices_new = {}

# Group labels by node
labels_by_node = defaultdict(list)
for (node_label_time, src_node_id) in labels.keys():
    labels_by_node[src_node_id].append(node_label_time)

for src_node_id in labels_by_node.keys():
    # NEW: Get edges for THIS node only
    node_mask = src_node_ids == src_node_id
    node_interactions_indices = np.where(node_mask)[0]
    node_interactions_times = node_interact_times[node_mask]
    node_edge_ids = edge_ids[node_mask]

    print(f"\nNode {src_node_id} edges: {node_edge_ids} at times {node_interactions_times}")

    for node_label_time in labels_by_node[src_node_id]:
        # NEW: Binary search
        valid_idx = np.searchsorted(node_interactions_times, node_label_time, side='right') - 1

        if valid_idx >= 0 and valid_idx < len(node_interactions_indices):
            nodes_most_recent_interaction_idx = node_edge_ids[valid_idx]
        else:
            nodes_most_recent_interaction_idx = 0

        labeled_node_interaction_indices_new[(node_label_time, src_node_id)] = nodes_most_recent_interaction_idx
        print(f"  Label at t={node_label_time} -> searchsorted index {valid_idx} -> Edge {nodes_most_recent_interaction_idx}")

print("\n=== VERIFICATION ===")
print(f"Old method: {labeled_node_interaction_indices_old}")
print(f"New method: {labeled_node_interaction_indices_new}")
print(f"Results match: {labeled_node_interaction_indices_old == labeled_node_interaction_indices_new}")
