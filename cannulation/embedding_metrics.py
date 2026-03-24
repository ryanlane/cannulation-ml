import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict


def compute_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute cluster quality and intrinsic dimensionality metrics on raw
    (pre-t-SNE) embeddings. These measure how well the model has learned
    to organise its internal representation space.

    All metrics operate on the full-dimensional embeddings (e.g. 128-dim
    fc1 output), not on t-SNE projections, so they reflect the actual
    learned geometry rather than a 2D/3D approximation.
    """
    X = StandardScaler().fit_transform(embeddings)
    classes = np.unique(labels)

    # ── Cluster quality ──────────────────────────────────────────────
    silhouette   = float(silhouette_score(X, labels, sample_size=min(500, len(X)), random_state=42))
    davies_bouldin = float(davies_bouldin_score(X, labels))
    calinski     = float(calinski_harabasz_score(X, labels))

    # ── Per-class geometry ───────────────────────────────────────────
    centroids = np.array([X[labels == c].mean(axis=0) for c in classes])

    intra_dists = {
        int(c): float(np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1)))
        for i, c in enumerate(classes)
    }
    mean_intra = float(np.mean(list(intra_dists.values())))

    n = len(classes)
    inter_dists = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    mean_inter = float(np.mean(inter_dists))
    separation_ratio = mean_inter / (mean_intra + 1e-8)

    # Worst-separated class pair
    worst_pair = min(
        ((classes[i], classes[j], np.linalg.norm(centroids[i] - centroids[j]))
         for i in range(n) for j in range(i + 1, n)),
        key=lambda t: t[2],
    )

    # ── Intrinsic dimensionality via PCA ─────────────────────────────
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dims_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    dims_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    var_top10 = float(cumvar[min(9, len(cumvar) - 1)])

    return {
        "silhouette":            silhouette,
        "davies_bouldin":        davies_bouldin,
        "calinski_harabasz":     calinski,
        "mean_intra_class_dist": mean_intra,
        "mean_inter_class_dist": mean_inter,
        "separation_ratio":      separation_ratio,
        "intra_class_by_class":  intra_dists,
        "worst_separated_pair":  [int(worst_pair[0]), int(worst_pair[1]), float(worst_pair[2])],
        "dims_for_90pct_var":    dims_90,
        "dims_for_99pct_var":    dims_99,
        "var_explained_top10":   var_top10,
        "embedding_dim":         int(embeddings.shape[1]),
    }
