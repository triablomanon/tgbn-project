# Moving Average (MA) Features Integration Plan for DyGFormer

## Overview
Add optional moving average features to DyGFormer for node classification tasks on temporal graphs. MA features track historical label **probability distributions** for each node using an exponential moving average with a sliding window.

## IMPORTANT: Labels Are Probability Distributions
After inspecting the tgbn-genre dataset, we discovered that labels are **NOT** binary multi-hot vectors. Instead, they are **probability distributions** that sum to 1.0:
- Average of ~11 active genres per node
- Each active genre has a probability weight (e.g., [0.17, 0.11, 0.09, ...])
- Sum of all probabilities = 1.0

**Implementation Decision**: MA features use the **full probability distribution**, not just the argmax. This preserves all genre information and creates richer temporal features.

## Rationale for Test Updates
**Question**: Why update MA features during test/validation?

**Answer**: Unlike static graphs, temporal graph benchmarks process data **sequentially in chronological order**. As we move through the test set:
1. We make predictions at time `t` using MA features from times `< t`
2. After computing loss, we update MA features with the observed label at time `t`
3. These updated features become available for predictions at time `t+1`

This maintains temporal consistency and mirrors real-world deployment where we continuously learn from observed labels.

## Implementation Details

### Files to Create

#### 1. `utils/MAFeatures.py` (NEW)
```python
import torch
import torch.nn.functional as F
import numpy as np


class MAFeatures:
    """
    Moving Average Features tracker for temporal node classification.
    Maintains an exponential moving average of node labels over a sliding window.
    """
    def __init__(self, num_class, window=7):
        """
        Args:
            num_class: Number of label classes
            window: Window size for exponential moving average (default: 7)
        """
        self.num_class = num_class
        self.window = window
        self.dict = {}

    def reset(self):
        """Clear all stored features."""
        self.dict = {}

    def update_dict(self, node_id, label_vec):
        """
        Update MA features for a node with exponential moving average.

        Args:
            node_id: Integer node ID
            label_vec: Numpy array of shape (num_class,) - one-hot or probability vector
        """
        if node_id in self.dict:
            total = self.dict[node_id] * (self.window - 1) + label_vec
            self.dict[node_id] = total / self.window
        else:
            self.dict[node_id] = label_vec

    def query_dict(self, node_id):
        """
        Query MA features for a single node.

        Args:
            node_id: Integer node ID

        Returns:
            Numpy array of shape (num_class,) - MA features or zeros if node not seen
        """
        if node_id in self.dict:
            return self.dict[node_id]
        else:
            return np.zeros(self.num_class, dtype=np.float32)

    def batch_query(self, node_ids):
        """
        Query MA features for a batch of nodes.

        Args:
            node_ids: Array-like of node IDs

        Returns:
            Numpy array of shape (batch_size, num_class) - stacked MA features
        """
        feats = [self.query_dict(int(n)) for n in node_ids]
        return np.stack(feats, axis=0).astype(np.float32)


def to_one_hot_if_needed(labels_tensor, num_classes):
    """
    Convert labels to one-hot encoding if needed.

    Args:
        labels_tensor: Tensor of labels (can be indices or already one-hot)
        num_classes: Number of classes

    Returns:
        One-hot encoded tensor of shape (batch_size, num_classes)
    """
    if labels_tensor.dim() == 1:
        return F.one_hot(labels_tensor.long(), num_classes=num_classes).float()
    return labels_tensor.float()
```

### Files to Modify

#### 2. `utils/load_configs.py`

**Location**: After line 150 (in `get_node_classification_args` function)

**Add these arguments**:
```python
parser.add_argument('--use_ma_features', action='store_true', default=False,
                    help='whether to use moving average features for node classification')
parser.add_argument('--ma_window_size', type=int, default=7,
                    help='window size for moving average features')
```

**Line numbers**:
- Insert after line 150 (before the `parser.add_argument('--subset_fraction', ...)` line)

---

#### 3. `train_node_classification.py`

**A. Import statements** (after line 14)
```python
from utils.MAFeatures import MAFeatures, to_one_hot_if_needed
```

**B. Initialize MA tracker** (after line 128, where node_classifier is created)
```python
# Initialize MA feature tracker if enabled
ma_tracker = None
if args.use_ma_features:
    ma_tracker = MAFeatures(num_class=num_classes, window=args.ma_window_size)
    logger.info(f'MA Features enabled with window size: {args.ma_window_size}')
```

**C. Modify MLPClassifier initialization** (line 128)
```python
# OLD:
node_classifier = MLPClassifier(input_dim=args.output_dim,
                                output_dim=num_classes,
                                dropout=args.dropout)

# NEW:
classifier_input_dim = args.output_dim + num_classes if args.use_ma_features else args.output_dim
node_classifier = MLPClassifier(input_dim=classifier_input_dim,
                                output_dim=num_classes,
                                dropout=args.dropout)
```

**D. Query MA features and concatenate** (insert after line 218, after embeddings are computed)

```python
# For DyGFormer - after computing embeddings
elif args.model_name in ['DyGFormer']:
    # get temporal embedding of source and destination nodes
    batch_src_node_embeddings, batch_dst_node_embeddings = \
        model[0].compute_src_dst_node_temporal_embeddings(...)

    # ========== INSERT HERE (after line 218) ==========
    # Query MA features if enabled
    if args.use_ma_features:
        ma_feats = ma_tracker.batch_query(batch_src_node_ids)
        ma_feats = torch.from_numpy(ma_feats).float().to(batch_src_node_embeddings.device)
        batch_src_node_embeddings = torch.cat([batch_src_node_embeddings, ma_feats], dim=1)
    # ===================================================
```

**E. Update MA features** (insert after line 239, after optimizer.step())

```python
# After optimizer.step()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ========== INSERT HERE (after line 239) ==========
# Update MA features if enabled
if args.use_ma_features and len(train_idx) > 0:
    with torch.no_grad():
        for idx in train_idx:
            node_id = int(batch_src_node_ids[idx])
            # Convert label to one-hot
            label_one_hot = to_one_hot_if_needed(labels[idx], num_classes)
            if label_one_hot.dim() > 1:  # If already multi-hot, take argmax
                label_idx = label_one_hot.argmax().item()
                label_one_hot = torch.zeros(num_classes, device=label_one_hot.device)
                label_one_hot[label_idx] = 1.0
            ma_tracker.update_dict(node_id, label_one_hot.cpu().numpy())
# ====================================================
```

---

#### 4. `evaluate_models_utils.py`

**A. Modify evaluation function signature** (line 170)
```python
# OLD:
def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, ...)

# NEW:
def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler,
                                       ..., ma_tracker=None)
```

**B. Query MA features during evaluation** (after line 261, after embeddings computed for DyGFormer)

```python
# For DyGFormer
elif model_name in ['DyGFormer']:
    batch_src_node_embeddings, batch_dst_node_embeddings = \
        model[0].compute_src_dst_node_temporal_embeddings(...)

    # ========== INSERT HERE ==========
    # Query MA features if tracker provided
    if ma_tracker is not None:
        ma_feats = ma_tracker.batch_query(batch_src_node_ids)
        ma_feats = torch.from_numpy(ma_feats).float().to(batch_src_node_embeddings.device)
        batch_src_node_embeddings = torch.cat([batch_src_node_embeddings, ma_feats], dim=1)
    # =================================
```

**C. Update MA features during evaluation** (after line 280, after computing loss)

```python
# After computing predictions and metrics
loss = loss_func(...)

# ========== INSERT HERE ==========
# Update MA features during evaluation (temporal progression)
if ma_tracker is not None and len(eval_idx) > 0:
    with torch.no_grad():
        for idx in eval_idx:
            node_id = int(batch_src_node_ids[idx])
            # Use ground truth labels for update
            label_one_hot = to_one_hot_if_needed(labels[idx], num_classes)
            if label_one_hot.dim() > 1:
                label_idx = label_one_hot.argmax().item()
                label_one_hot = torch.zeros(num_classes, device=label_one_hot.device)
                label_one_hot[label_idx] = 1.0
            ma_tracker.update_dict(node_id, label_one_hot.cpu().numpy())
# =================================
```

**D. Update function calls** (lines 302, 322, 342)

When calling `evaluate_model_node_classification`, pass `ma_tracker`:
```python
# OLD:
val_metrics = evaluate_model_node_classification(..., eval_metric_name=eval_metric_name)

# NEW:
val_metrics = evaluate_model_node_classification(..., eval_metric_name=eval_metric_name,
                                                 ma_tracker=ma_tracker if args.use_ma_features else None)
```

---

## Usage Examples

### Without MA Features (Default)
```bash
python train_node_classification.py \
    --dataset_name tgbn-genre \
    --model_name DyGFormer \
    --gpu 0
```

### With MA Features
```bash
python train_node_classification.py \
    --dataset_name tgbn-genre \
    --model_name DyGFormer \
    --gpu 0 \
    --use_ma_features \
    --ma_window_size 7
```

### With Filtered Data + MA Features
```bash
python train_node_classification.py \
    --dataset_name tgbn-genre \
    --model_name DyGFormer \
    --gpu 0 \
    --subset_fraction 0.1 \
    --timestamp_threshold 1135778006 \
    --use_ma_features \
    --ma_window_size 7
```

## Expected Behavior

### Dimension Changes
- **Without MA**: Embeddings shape = `(batch_size, 172)` → Classifier input = 172
- **With MA**: Embeddings shape = `(batch_size, 172)` → MA features shape = `(batch_size, 513)` → Concatenated = `(batch_size, 685)` → Classifier input = 685

### Memory Growth
- MA tracker dictionary grows as new nodes are encountered
- For tgbn-genre: ~992 nodes (full) or ~600 nodes (10% filtered)
- Memory per node: 513 floats × 4 bytes = 2 KB
- Total memory: ~1-2 MB (negligible)

### Temporal Consistency
1. **Epoch 1, Batch 1**: All nodes have zero MA features (never seen)
2. **Epoch 1, Batch 2**: Some nodes may have MA features from Batch 1
3. **Epoch 2**: MA features carry over from Epoch 1 (accumulated knowledge)
4. **Validation/Test**: Use accumulated MA features, continue updating as we progress

### No Reset Between Epochs
Unlike some memory-based models, **MA tracker is NOT reset** between epochs. This allows the model to accumulate temporal label patterns across the entire training process.

## Testing Checklist

- [ ] Code runs without `--use_ma_features` (backward compatible)
- [ ] Code runs with `--use_ma_features` (new functionality)
- [ ] Classifier input dimension changes correctly (172 vs 685)
- [ ] MA features queried before prediction in both train and eval
- [ ] MA features updated after loss in both train and eval
- [ ] No crashes with filtered dataset
- [ ] Memory usage remains reasonable
- [ ] NDCG@10 metric computed correctly
- [ ] Model can save/load checkpoints (dimensions must match)

## Open Questions

1. **Should MA tracker be saved in checkpoints?**
   - Pros: Enables exact resumption of training
   - Cons: Increases checkpoint size
   - **Recommendation**: Save it if checkpoint saving is needed for long training runs

2. **Should MA features be used for other models (TGN, TGAT)?**
   - Currently scoped to DyGFormer only
   - Can extend to other models later if beneficial

3. **Alternative to argmax for multi-label?**
   - Current approach takes primary label (argmax)
   - Could use full multi-hot vector instead of one-hot
   - **Current choice**: One-hot for simplicity and consistency with TGN implementation

## Summary of Changes

| File | Lines Modified | Lines Added | Purpose |
|------|----------------|-------------|---------|
| `utils/MAFeatures.py` | 0 | ~80 | New MA tracker class |
| `utils/load_configs.py` | 0 | 4 | Add command-line arguments |
| `train_node_classification.py` | 2 | ~20 | Integration logic |
| `evaluate_models_utils.py` | 4 | ~20 | Evaluation integration |
| **Total** | **6** | **~124** | **Full implementation** |

## References
- Original TGN implementation with MA features (provided by user)
- tgbn-genre benchmark: 513 classes, ~992 nodes, temporal node classification
- DyGFormer paper: Transformer-based temporal graph learning
