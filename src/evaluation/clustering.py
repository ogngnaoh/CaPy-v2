"""MOA clustering evaluation (FR-8.3)."""

from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_moa_clustering(
    embeddings: torch.Tensor,
    moa_labels: list[str | None],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute MOA clustering metrics (FR-8.3).

    Filters out null MOA labels, then computes:
    - k-NN MOA accuracy for each k in k_values (majority vote)
    - AMI (Adjusted Mutual Information) between k-means clusters and true MOA
    - ARI (Adjusted Rand Index) between k-means clusters and true MOA

    Args:
        embeddings: Tensor of shape [N, D].
        moa_labels: List of MOA labels (length N). None/null entries are filtered out.
        k_values: List of k values for k-NN accuracy. Defaults to [5, 10, 20].

    Returns:
        Dict with keys: "AMI", "ARI", "kNN_{k}_acc" for each k.
        Empty dict if ≤1 unique MOA after filtering.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # Filter out null MOA labels
    valid_indices = [
        i for i, label in enumerate(moa_labels) if label is not None
    ]
    if not valid_indices:
        logger.warning("No valid MOA labels found. Skipping MOA clustering.")
        return {}

    filtered_embeddings = embeddings[valid_indices]
    filtered_labels = [moa_labels[i] for i in valid_indices]

    # Check for ≤1 unique MOA
    unique_moas = set(filtered_labels)
    if len(unique_moas) <= 1:
        logger.warning(
            "Only %d unique MOA label(s) found. Skipping MOA clustering.",
            len(unique_moas),
        )
        return {}

    n_samples = len(filtered_labels)
    emb_np = filtered_embeddings.detach().cpu().numpy().astype(np.float64)

    # Encode string labels to integers for sklearn
    sorted_moas = sorted(m for m in unique_moas if m is not None)
    label_to_int = {label: i for i, label in enumerate(sorted_moas)}
    true_labels = np.array(
        [label_to_int[lbl] for lbl in filtered_labels if lbl is not None]
    )

    metrics: dict[str, float] = {}

    # k-NN MOA accuracy (majority vote)
    for k in k_values:
        k_eff = min(k, n_samples - 1)
        if k_eff < 1:
            metrics[f"kNN_{k}_acc"] = 0.0
            continue

        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
        nn.fit(emb_np)
        # +1 because the query itself is included in results
        _, indices = nn.kneighbors(emb_np)
        # Exclude self (first neighbor)
        neighbor_indices = indices[:, 1:]

        correct = 0
        for i in range(n_samples):
            neighbor_labels = true_labels[neighbor_indices[i]]
            majority_label = Counter(neighbor_labels).most_common(1)[0][0]
            if majority_label == true_labels[i]:
                correct += 1
        metrics[f"kNN_{k}_acc"] = correct / n_samples

    # k-means clustering
    n_clusters = len(unique_moas)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(emb_np)

    # AMI and ARI
    metrics["AMI"] = float(
        adjusted_mutual_info_score(true_labels, kmeans_labels)
    )
    metrics["ARI"] = float(adjusted_rand_score(true_labels, kmeans_labels))

    logger.info(
        "MOA clustering: %d compounds with MOA labels. "
        "AMI=%.4f, ARI=%.4f, kNN-5 acc=%.4f.",
        n_samples,
        metrics["AMI"],
        metrics["ARI"],
        metrics.get("kNN_5_acc", 0.0),
    )

    return metrics
