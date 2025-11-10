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
