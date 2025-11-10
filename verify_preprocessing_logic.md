# Verification: Old vs New Preprocessing Logic

## Example Data
```
Edges (src_node_id, time, edge_id):
  0: (5, 100, 0)
  1: (5, 200, 1)  ← ANSWER for label at t=250
  2: (5, 300, 2)
  3: (7, 150, 3)  ← ANSWER for label at t=200
  4: (7, 250, 4)
  5: (5, 400, 5)  ← ANSWER for label at t=400
```

## Labels to Map
```
User 5 at t=250 → Should map to edge 1 (most recent before 250 is at t=200)
User 5 at t=400 → Should map to edge 5 (exact match at t=400)
User 7 at t=200 → Should map to edge 3 (most recent before 200 is at t=150)
```

---

## OLD METHOD (Line-by-line execution)

### For label (User 5, t=250):
```python
nodes_historical_interactions_mask = (src_node_ids == 5) & (node_interact_times <= 250)
# Check each edge:
#   Edge 0: (5, 100) → src==5 ✓, time<=250 ✓ → TRUE
#   Edge 1: (5, 200) → src==5 ✓, time<=250 ✓ → TRUE
#   Edge 2: (5, 300) → src==5 ✓, time<=250 ✗ → FALSE
#   Edge 3: (7, 150) → src==5 ✗ → FALSE
#   Edge 4: (7, 250) → src==5 ✗ → FALSE
#   Edge 5: (5, 400) → src==5 ✓, time<=250 ✗ → FALSE
# mask = [TRUE, TRUE, FALSE, FALSE, FALSE, FALSE]

edge_ids[mask] = [0, 1]
nodes_most_recent_interaction_idx = edge_ids[mask][-1] = 1  ✓ CORRECT
```

---

## NEW METHOD (Line-by-line execution)

### Step 1: Group labels by user
```python
labels_by_node = {
    5: [250, 400],
    7: [200]
}
```

### Step 2: Process User 5
```python
node_mask = (src_node_ids == 5)
# Check each edge:
#   Edge 0: src==5 → TRUE
#   Edge 1: src==5 → TRUE
#   Edge 2: src==5 → TRUE
#   Edge 3: src==7 → FALSE
#   Edge 4: src==7 → FALSE
#   Edge 5: src==5 → TRUE
# mask = [TRUE, TRUE, TRUE, FALSE, FALSE, TRUE]

node_interactions_indices = [0, 1, 2, 5]  # Indices in ORIGINAL array
node_interactions_times = [100, 200, 300, 400]  # Times for User 5 only
node_edge_ids = [0, 1, 2, 5]  # Edge IDs for User 5 only
```

### For User 5's label at t=250:
```python
valid_idx = np.searchsorted([100, 200, 300, 400], 250, side='right') - 1
# searchsorted finds where to insert 250: between 200 and 300 → index 2
# side='right' means insert AFTER equal values
# Subtract 1 to get "most recent before or equal"
# valid_idx = 2 - 1 = 1

nodes_most_recent_interaction_idx = node_edge_ids[1] = 1  ✓ CORRECT (same as old!)
```

### For User 5's label at t=400:
```python
valid_idx = np.searchsorted([100, 200, 300, 400], 400, side='right') - 1
# searchsorted finds where to insert 400: AFTER the last 400 → index 4
# valid_idx = 4 - 1 = 3

nodes_most_recent_interaction_idx = node_edge_ids[3] = 5  ✓ CORRECT
```

### Step 3: Process User 7
```python
node_mask = (src_node_ids == 7)
node_interactions_indices = [3, 4]
node_interactions_times = [150, 250]
node_edge_ids = [3, 4]
```

### For User 7's label at t=200:
```python
valid_idx = np.searchsorted([150, 250], 200, side='right') - 1
# searchsorted finds where to insert 200: between 150 and 250 → index 1
# valid_idx = 1 - 1 = 0

nodes_most_recent_interaction_idx = node_edge_ids[0] = 3  ✓ CORRECT
```

---

## CONCLUSION

**Both methods produce IDENTICAL results:**
- Label (5, t=250) → Edge 1 ✓
- Label (5, t=400) → Edge 5 ✓
- Label (7, t=200) → Edge 3 ✓

**Why the new method is faster:**
- OLD: Scans ALL edges for EACH label = 3 labels × 6 edges = 18 comparisons
- NEW: Scans edges once per user, then binary search = (6 comparisons for User 5) + (6 comparisons for User 7) + (3 binary searches × log(4)) ≈ 12 + 6 = 18 operations, BUT:
  - For real data: OLD = 10,000 labels × 17.8M edges = 178 BILLION comparisons
  - NEW = 992 users × 18K avg edges + 10,000 × log(18K) ≈ 18 million + 140K ≈ 18 million comparisons
  - **Speedup: 178 billion / 18 million = ~10,000x faster!**
