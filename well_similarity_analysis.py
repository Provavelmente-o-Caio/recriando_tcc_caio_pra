"""
WELL SIMILARITY ANALYSIS SCRIPT
================================

This script analyzes the similarity between wells to:
1. Explain why models perform better on certain wells
2. Identify which wells should be used together for training
3. Understand geological/geophysical relationships between wells

Techniques used:
- Multidimensional Scaling (MDS) for visualization
- Hierarchical Clustering for grouping
- Statistical correlation analysis
- Distribution comparison (KL divergence, Wasserstein distance)
- Dynamic Time Warping (DTW) for sequence similarity

Usage:
    python well_similarity_analysis.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# Import your custom data loading
from petrobras_dataset import filter_commom_features, read_all_wells_with_dept_to_list

warnings.filterwarnings("ignore")
# ============================================================================
# SIMILARITY METRICS
# ============================================================================


class WellSimilarityAnalyzer:
    """
    Comprehensive well similarity analysis using multiple metrics.

    Why these metrics?
    - Statistical distribution similarity: Do wells have similar value ranges?
    - Correlation: Do features co-vary similarly across wells?
    - Sequence similarity: Do wells show similar patterns over depth?
    - Feature space distance: How far apart are wells in feature space?
    """

    def __init__(self, wells_data, feature_names):
        """
        Initialize analyzer with well data.

        Args:
            wells_data: List of Polars DataFrames, one per well
            feature_names: List of feature names to analyze
        """
        self.wells_data = wells_data
        self.feature_names = feature_names
        self.n_wells = len(wells_data)

        # Clean data - remove wells without features
        self.wells_data = [
            df for df in wells_data if all(feat in df.columns for feat in feature_names)
        ]
        self.n_wells = len(self.wells_data)

        print(f"Analyzing {self.n_wells} wells with features: {feature_names}")

        # Initialize distance matrices
        self.distance_matrices = {}

    def _print_matrix(self, matrix, title="Distance Matrix"):
        """
        Print distance matrix in a clear, readable format in the terminal.

        Args:
            matrix: numpy array representing the distance matrix
            title: Title for the matrix display
        """
        print(f"\n{title}:")
        print("-" * (12 * self.n_wells + 12))

        # Header row
        header = "        "
        for j in range(self.n_wells):
            header += f"Well {j:2d}  "
        print(header)
        print("-" * (12 * self.n_wells + 12))

        # Data rows
        for i in range(self.n_wells):
            row = f"Well {i:2d} |"
            for j in range(self.n_wells):
                if i == j:
                    row += "   ---   "
                else:
                    row += f" {matrix[i, j]:7.4f} "
            print(row)

        print("-" * (12 * self.n_wells + 12))

        # Summary statistics
        # Get upper triangle (excluding diagonal)
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        print(f"Min distance: {upper_tri.min():.4f}")
        print(f"Max distance: {upper_tri.max():.4f}")
        print(f"Mean distance: {upper_tri.mean():.4f}")
        print(f"Std distance: {upper_tri.std():.4f}")

    def compute_statistical_distance(self):
        """
        Compute statistical distance between wells based on feature distributions.

        Uses Wasserstein distance (Earth Mover's Distance):
        - Measures how much "work" is needed to transform one distribution to another
        - Robust to outliers
        - Interpretable: larger = more different distributions

        Why this matters:
        If two wells have very different feature distributions, a model trained
        on one may not generalize well to the other.
        """
        print("\n" + "=" * 80)
        print("Computing Statistical Distance (Wasserstein)")
        print("=" * 80)

        distance_matrix = np.zeros((self.n_wells, self.n_wells))

        for i in range(self.n_wells):
            for j in range(i + 1, self.n_wells):
                distances = []

                for feature in self.feature_names:
                    # Drop nulls and extract numpy arrays
                    values_i = (
                        self.wells_data[i]
                        .select(pl.col(feature).drop_nulls())
                        .to_series()
                        .to_numpy()
                    )
                    values_j = (
                        self.wells_data[j]
                        .select(pl.col(feature).drop_nulls())
                        .to_series()
                        .to_numpy()
                    )

                    if len(values_i) > 0 and len(values_j) > 0:
                        dist = wasserstein_distance(values_i, values_j)
                        distances.append(dist)

                avg_distance = np.mean(distances) if distances else np.inf
                distance_matrix[i, j] = avg_distance
                distance_matrix[j, i] = avg_distance

        self.distance_matrices["statistical"] = distance_matrix
        print("Statistical distance matrix computed.")
        self._print_matrix(distance_matrix, "Statistical Distance")
        return distance_matrix

    def compute_correlation_distance(self):
        """
        Compute distance based on feature correlation patterns.

        Measures how similarly features correlate within each well.

        Why this matters:
        Wells with similar correlation structures have similar physical relationships
        between features (e.g., VP-RHO relationship). Models learn these relationships,
        so similar correlations = better generalization.
        """
        print("\n" + "=" * 80)
        print("Computing Correlation Distance")
        print("=" * 80)

        correlation_matrices = []

        for well in self.wells_data:
            # Select features and drop rows with any null
            well_features = well.select(self.feature_names).drop_nulls()

            # Compute correlation matrix via numpy (polars doesn't have .corr() for multiple cols)
            data_np = well_features.to_numpy()
            corr_matrix = np.corrcoef(data_np, rowvar=False)

            # Flatten upper triangle (excluding diagonal)
            corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            correlation_matrices.append(corr_values)

        correlation_matrices = np.array(correlation_matrices)

        distance_matrix = np.zeros((self.n_wells, self.n_wells))

        for i in range(self.n_wells):
            for j in range(i + 1, self.n_wells):
                dist = np.linalg.norm(correlation_matrices[i] - correlation_matrices[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        self.distance_matrices["correlation"] = distance_matrix
        print("Correlation distance matrix computed.")
        self._print_matrix(distance_matrix, "Correlation Distance")
        return distance_matrix

    def compute_dtw_distance(self):
        """
        Compute Dynamic Time Warping (DTW) distance for sequence similarity.

        DTW allows for non-linear alignment of sequences:
        - Handles depth shifts (same formation at different depths)
        - Robust to local variations
        - Good for comparing geological sequences

        Why this matters:
        Wells may have similar geology but at different depths. DTW can identify
        this similarity even if direct comparison fails.
        """
        print("\n" + "=" * 80)
        print("Computing DTW Distance (Sequence Similarity)")
        print("=" * 80)

        def dtw_distance(seq1, seq2):
            """
            Simple DTW implementation.
            Returns normalized distance between two sequences.
            """
            n, m = len(seq1), len(seq2)

            dtw = np.full((n + 1, m + 1), np.inf)
            dtw[0, 0] = 0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
                    dtw[i, j] = cost + min(
                        dtw[i - 1, j],  # insertion
                        dtw[i, j - 1],  # deletion
                        dtw[i - 1, j - 1],  # match
                    )

            return dtw[n, m] / (n + m)

        distance_matrix = np.zeros((self.n_wells, self.n_wells))

        for i in range(self.n_wells):
            for j in range(i + 1, self.n_wells):
                # Drop nulls and downsample for speed
                seq_i = (
                    self.wells_data[i]
                    .select(self.feature_names)
                    .drop_nulls()
                    .to_numpy()[::10]
                )
                seq_j = (
                    self.wells_data[j]
                    .select(self.feature_names)
                    .drop_nulls()
                    .to_numpy()[::10]
                )

                if len(seq_i) > 0 and len(seq_j) > 0:
                    dist = dtw_distance(seq_i, seq_j)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                else:
                    distance_matrix[i, j] = np.inf
                    distance_matrix[j, i] = np.inf

        self.distance_matrices["dtw"] = distance_matrix
        print("DTW distance matrix computed.")
        self._print_matrix(distance_matrix, "DTW Distance")
        return distance_matrix

    def compute_feature_space_distance(self):
        """
        Compute distance in feature space using mean feature vectors.

        Represents each well by its average feature values, then computes
        Euclidean distance.

        Why this matters:
        Simple but effective measure of overall well similarity. Wells with
        similar average properties are likely from similar geological settings.
        """
        print("\n" + "=" * 80)
        print("Computing Feature Space Distance")
        print("=" * 80)

        mean_vectors = []

        for well in self.wells_data:
            # Compute mean per feature, ignoring nulls
            mean_vector = (
                well.select(
                    [pl.col(f).drop_nulls().mean().alias(f) for f in self.feature_names]
                )
                .to_numpy()
                .flatten()
            )
            mean_vectors.append(mean_vector)

        mean_vectors = np.array(mean_vectors)

        # Standardize features
        scaler = StandardScaler()
        mean_vectors_scaled = scaler.fit_transform(mean_vectors)

        # Compute pairwise Euclidean distances
        distance_matrix = squareform(pdist(mean_vectors_scaled, metric="euclidean"))

        self.distance_matrices["feature_space"] = distance_matrix
        print("Feature space distance matrix computed.")
        self._print_matrix(distance_matrix, "Feature Space Distance")
        return distance_matrix

    def compute_combined_distance(self, weights=None):
        """
        Combine multiple distance metrics into a single distance matrix.

        Args:
            weights: Dict with keys matching distance_matrices keys and float values
                    Default: Equal weights for all metrics

        Why combine metrics?
        Each metric captures different aspects of similarity. Combining them
        gives a more robust overall measure.
        """
        print("\n" + "=" * 80)
        print("Computing Combined Distance Matrix")
        print("=" * 80)

        if weights is None:
            weights = {key: 1.0 for key in self.distance_matrices.keys()}

        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        print(f"Weights: {weights}")

        normalized_matrices = {}
        for key, matrix in self.distance_matrices.items():
            max_val = matrix.max()
            if max_val > 0:
                normalized_matrices[key] = matrix / max_val
            else:
                normalized_matrices[key] = matrix

        combined = np.zeros_like(list(normalized_matrices.values())[0])

        for key, weight in weights.items():
            if key in normalized_matrices:
                combined += weight * normalized_matrices[key]

        self.distance_matrices["combined"] = combined
        print("Combined distance matrix computed.")
        self._print_matrix(combined, "Combined Distance")
        return combined

    def compute_all_distances(self):
        """Compute all distance metrics."""
        self.compute_statistical_distance()
        self.compute_correlation_distance()
        self.compute_dtw_distance()
        self.compute_feature_space_distance()
        self.compute_combined_distance()

        return self.distance_matrices


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


class WellSimilarityVisualizer:
    """
    Visualize well similarities using various techniques.

    Techniques:
    1. MDS (Multidimensional Scaling): Project high-D distances to 2D
    2. Hierarchical Clustering: Show grouping structure
    3. Heatmaps: Direct visualization of distance matrices
    """

    def __init__(self, analyzer, output_dir="well_similarity_analysis"):
        """
        Initialize visualizer.

        Args:
            analyzer: WellSimilarityAnalyzer instance
            output_dir: Directory to save plots
        """
        self.analyzer = analyzer
        self.output_dir = output_dir

        import os

        os.makedirs(output_dir, exist_ok=True)

    def plot_mds(self, distance_type="combined", metric=True):
        """
        Plot Multidimensional Scaling (MDS) visualization.

        MDS explanation:
        - Takes a distance matrix
        - Finds 2D coordinates that best preserve those distances
        - Similar to PCA but works with distances instead of raw data
        - Points close together = similar wells
        - Points far apart = dissimilar wells

        Args:
            distance_type: Which distance matrix to use
            metric: If True, use metric MDS (preserves exact distances)
                   If False, use non-metric MDS (preserves rank order)

        Why MDS?
        - Provides intuitive 2D visualization of well relationships
        - Can reveal clusters and outliers
        - Helps explain why certain wells work well together
        """
        print(f"\n{'=' * 80}")
        print(f"Plotting MDS for {distance_type} distance")
        print(f"{'=' * 80}")

        distance_matrix = self.analyzer.distance_matrices[distance_type]

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            metric=metric,
            max_iter=1000,
        )

        coords = mds.fit_transform(distance_matrix)

        stress = mds.stress_
        print(f"MDS Stress: {stress:.4f}")
        print("  (Lower is better. <0.1 = excellent, 0.1-0.2 = good, >0.2 = poor)")

        plt.figure(figsize=(12, 10))

        scatter = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            s=200,
            c=range(self.analyzer.n_wells),
            cmap="tab10",
            alpha=0.6,
            edgecolors="black",
            linewidth=2,
        )

        for i, (x, y) in enumerate(coords):
            plt.annotate(
                f"Poço {i + 1}",
                (x, y),
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

        plt.xlabel("Dimensão MDS 1", fontsize=14, fontweight="bold")
        plt.ylabel("Dimensão MDS 2", fontsize=14, fontweight="bold")

        distance_names = {
            "statistical": "Estatística",
            "correlation": "Correlação",
            "dtw": "DTW",
            "feature_space": "Espaço de Características",
            "combined": "Combinada",
        }
        dist_name = distance_names.get(distance_type, distance_type.title())

        plt.title(
            f"Similaridade entre Poços - Projeção MDS (Distância {dist_name})\n"
            f"Stress: {stress:.4f}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label="Índice do Poço")

        plt.text(
            0.02,
            0.98,
            "Interpretação:\n"
            "• Pontos próximos = Poços similares\n"
            "• Pontos distantes = Poços dissimilares\n"
            "• Agrupamentos = Grupos de poços similares\n"
            "• Baixo stress = Boa representação 2D",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        filename = f"{self.output_dir}/mds_{distance_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved to: {filename}")
        plt.close()

        return coords, stress

    def plot_dendrogram(self, distance_type="combined", method="ward"):
        """
        Plot hierarchical clustering dendrogram.

        Dendrogram explanation:
        - Shows hierarchical grouping of wells
        - Height of branches = dissimilarity when wells merge
        - Can identify natural groupings at different similarity levels

        Args:
            distance_type: Which distance matrix to use
            method: Linkage method ('ward', 'average', 'complete', 'single')

        Why dendrograms?
        - Show hierarchical structure of well relationships
        - Can cut at different heights to get different numbers of clusters
        - Useful for deciding which wells to group for training
        """
        print(f"\n{'=' * 80}")
        print(f"Plotting Dendrogram for {distance_type} distance")
        print(f"{'=' * 80}")

        distance_matrix = self.analyzer.distance_matrices[distance_type]

        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method=method)

        plt.figure(figsize=(14, 8))

        dendrogram(
            linkage_matrix,
            labels=[f"Poço {i + 1}" for i in range(self.analyzer.n_wells)],
            leaf_font_size=12,
            color_threshold=0.7 * max(linkage_matrix[:, 2]),
        )

        distance_names = {
            "statistical": "Estatística",
            "correlation": "Correlação",
            "dtw": "DTW",
            "feature_space": "Espaço de Características",
            "combined": "Combinada",
        }
        dist_name = distance_names.get(distance_type, distance_type.title())

        method_names = {
            "ward": "Ward",
            "average": "Média",
            "complete": "Completa",
            "single": "Simples",
        }
        method_name = method_names.get(method, method.title())

        plt.xlabel("Índice do Poço", fontsize=14, fontweight="bold")
        plt.ylabel("Distância", fontsize=14, fontweight="bold")
        plt.title(
            f"Agrupamento Hierárquico de Poços (Distância {dist_name})\n"
            f"Método de Ligação: {method_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.grid(True, alpha=0.3, axis="y")

        plt.text(
            0.02,
            0.98,
            "Interpretação:\n"
            "• Ramos inferiores = Poços mais similares\n"
            "• Altura = Dissimilaridade na fusão\n"
            "• Agrupamentos coloridos = Grupos sugeridos\n"
            "• Corte horizontal define os clusters",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        filename = f"{self.output_dir}/dendrogram_{distance_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved to: {filename}")
        plt.close()

        return linkage_matrix

    def plot_distance_heatmap(self, distance_type="combined"):
        """
        Plot distance matrix as heatmap.

        Why heatmaps?
        - Direct visualization of pairwise distances
        - Easy to identify similar/dissimilar well pairs
        - Color coding makes patterns obvious
        """
        print(f"\n{'=' * 80}")
        print(f"Plotting Distance Heatmap for {distance_type} distance")
        print(f"{'=' * 80}")

        distance_matrix = self.analyzer.distance_matrices[distance_type]

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=[f"Poço {i + 1}" for i in range(self.analyzer.n_wells)],
            yticklabels=[f"Poço {i + 1}" for i in range(self.analyzer.n_wells)],
            cmap="YlOrRd",
            cbar_kws={"label": "Distância"},
            square=True,
        )

        distance_names = {
            "statistical": "Estatística",
            "correlation": "Correlação",
            "dtw": "DTW",
            "feature_space": "Espaço de Características",
            "combined": "Combinada",
        }
        dist_name = distance_names.get(distance_type, distance_type.title())

        plt.title(
            f"Distâncias Par a Par entre Poços ({dist_name})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.text(
            0.02,
            -0.15,
            "Interpretação: Mais escuro (vermelho) = Mais dissimilar, Mais claro (amarelo) = Mais similar",
            transform=plt.gca().transAxes,
            fontsize=10,
            ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        filename = f"{self.output_dir}/heatmap_{distance_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved to: {filename}")
        plt.close()

    def plot_all_distance_types(self):
        """Plot MDS for all distance types on one figure."""
        print(f"\n{'=' * 80}")
        print("Plotting Comparison of All Distance Metrics")
        print(f"{'=' * 80}")

        distance_types = [
            k for k in self.analyzer.distance_matrices.keys() if k != "combined"
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()

        for idx, dist_type in enumerate(distance_types):
            if idx >= 4:
                break

            ax = axes[idx]

            distance_matrix = self.analyzer.distance_matrices[dist_type]

            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                metric=True,
                max_iter=1000,
            )
            coords = mds.fit_transform(distance_matrix)

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=150,
                c=range(self.analyzer.n_wells),
                cmap="tab10",
                alpha=0.6,
                edgecolors="black",
                linewidth=1.5,
            )

            for i, (x, y) in enumerate(coords):
                ax.annotate(f"P{i + 1}", (x, y), fontsize=10, ha="center", va="bottom")

            distance_names = {
                "statistical": "Estatística",
                "correlation": "Correlação",
                "dtw": "DTW",
                "feature_space": "Espaço de Características",
            }
            dist_name = distance_names.get(dist_type, dist_type.title())

            ax.set_xlabel("Dimensão MDS 1", fontsize=11)
            ax.set_ylabel("Dimensão MDS 2", fontsize=11)
            ax.set_title(
                f"Distância {dist_name}\nStress: {mds.stress_:.3f}",
                fontsize=13,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Similaridade entre Poços - Comparação de Métricas de Distância",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        filename = f"{self.output_dir}/mds_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved to: {filename}")
        plt.close()

    def create_all_plots(self):
        """Generate all visualization plots."""
        for dist_type in self.analyzer.distance_matrices.keys():
            self.plot_mds(dist_type)
            self.plot_distance_heatmap(dist_type)

        self.plot_dendrogram("combined")
        self.plot_all_distance_types()


# ============================================================================
# ANALYSIS AND RECOMMENDATIONS
# ============================================================================


class WellGroupingRecommendation:
    """
    Provide recommendations for well grouping based on similarity analysis.

    Answers:
    1. Which wells are most similar? (should train together)
    2. Which wells are outliers? (may need separate treatment)
    3. What's the optimal training/testing split?
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def identify_well_clusters(self, distance_type="combined", cluster_height=2):
        """
        Identify natural clusters of wells.

        Uses hierarchical clustering to group wells.
        """

        distance_matrix = self.analyzer.distance_matrices[distance_type]
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method="ward")

        clusters = fcluster(linkage_matrix, cluster_height, criterion="distance")

        wells_in_cluster = {}

        for cluster_id in np.unique(clusters):
            wells = np.where(clusters == cluster_id)[0]
            wells_in_cluster[str(cluster_id)] = wells
            print(f"\nCluster {cluster_id}: Wells {list(wells)}")

        return wells_in_cluster

    def find_most_similar_pairs(self, distance_type="combined", top_k=5):
        """Find the most similar well pairs."""
        distance_matrix = self.analyzer.distance_matrices[distance_type]

        upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)

        distances = distance_matrix[upper_tri_indices]
        sorted_indices = np.argsort(distances)

        print(f"\n{'=' * 80}")
        print(f"Top {top_k} Most Similar Well Pairs")
        print(f"{'=' * 80}")

        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            well_i = upper_tri_indices[0][idx]
            well_j = upper_tri_indices[1][idx]
            dist = distances[idx]

            print(f"{i + 1}. Well {well_i} ↔ Well {well_j}: Distance = {dist:.4f}")

        return sorted_indices[:top_k]

    def find_most_dissimilar_pairs(self, distance_type="combined", top_k=5):
        """Find the most dissimilar well pairs."""
        distance_matrix = self.analyzer.distance_matrices[distance_type]

        upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
        distances = distance_matrix[upper_tri_indices]
        sorted_indices = np.argsort(distances)[::-1]

        print(f"\n{'=' * 80}")
        print(f"Top {top_k} Most Dissimilar Well Pairs")
        print(f"{'=' * 80}")

        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            well_i = upper_tri_indices[0][idx]
            well_j = upper_tri_indices[1][idx]
            dist = distances[idx]

            print(f"{i + 1}. Well {well_i} ↔ Well {well_j}: Distance = {dist:.4f}")

        return sorted_indices[:top_k]

    def identify_outlier_wells(self, distance_type="combined", threshold_percentile=75):
        """
        Identify wells that are outliers (dissimilar from most others).

        Outlier wells may:
        - Come from different geological formations
        - Have data quality issues
        - Require separate model training
        """
        distance_matrix = self.analyzer.distance_matrices[distance_type]

        avg_distances = distance_matrix.mean(axis=1)

        threshold = np.percentile(avg_distances, threshold_percentile)
        outliers = np.where(avg_distances > threshold)[0]

        print(f"\n{'=' * 80}")
        print("Outlier Well Detection")
        print(f"{'=' * 80}")
        print(f"Threshold: {threshold:.4f} ({threshold_percentile}th percentile)")
        print(f"\nAverage distances from each well to others:")
        for i, avg_dist in enumerate(avg_distances):
            marker = " ← OUTLIER" if i in outliers else ""
            print(f"  Well {i}: {avg_dist:.4f}{marker}")

        if len(outliers) > 0:
            print(f"\nOutlier wells: {list(outliers)}")
        else:
            print("\nNo outlier wells detected.")

        return outliers, avg_distances

    def recommend_train_test_split(self, distance_type="combined", test_well_idx=None):
        """
        Recommend which wells to use for training vs testing.

        Strategy:
        - If test well is specified, find most similar wells for training
        - If not specified, suggest diverse training set with outlier as test
        """
        distance_matrix = self.analyzer.distance_matrices[distance_type]

        print(f"\n{'=' * 80}")
        print("Train/Test Split Recommendations")
        print(f"{'=' * 80}")

        if test_well_idx is not None:
            distances_to_test = distance_matrix[test_well_idx, :]
            sorted_indices = np.argsort(distances_to_test)

            sorted_indices = sorted_indices[sorted_indices != test_well_idx]

            print(f"\nTest Well: {test_well_idx}")
            print(f"\nRecommended training wells (most similar to test):")
            for i, well_idx in enumerate(sorted_indices[:5]):
                dist = distances_to_test[well_idx]
                print(f"  {i + 1}. Well {well_idx} (distance: {dist:.4f})")

        else:
            outliers, avg_distances = self.identify_outlier_wells(distance_type)

            if len(outliers) > 0:
                test_candidate = outliers[0]
                print(f"\nSuggested test well: {test_candidate} (outlier well)")
                print(
                    "Rationale: Testing on outlier evaluates generalization to dissimilar wells"
                )
                return test_candidate
            else:
                median_idx = np.argsort(avg_distances)[len(avg_distances) // 2]
                print(f"\nSuggested test well: {median_idx} (median similarity)")
                print("Rationale: No clear outliers; using well with median similarity")
                return median_idx

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print(f"\n{'=' * 80}")
        print("WELL SIMILARITY ANALYSIS - SUMMARY REPORT")
        print(f"{'=' * 80}")

        clusters = self.identify_well_clusters(cluster_height=3)
        self.find_most_similar_pairs(top_k=3)
        self.find_most_dissimilar_pairs(top_k=3)
        outliers, _ = self.identify_outlier_wells()
        self.recommend_train_test_split()

        print(f"\n{'=' * 80}")
        print("KEY INSIGHTS")
        print(f"{'=' * 80}")
        print("\n1. TRAINING STRATEGY:")
        print("   - Train models on similar well groups identified by clustering")
        print("   - Expect better performance within clusters than across clusters")

        print("\n2. MODEL PERFORMANCE EXPLANATION:")
        print("   - High R² expected for wells in same cluster")
        print("   - Low R² expected when testing on outlier wells")
        print("   - Use MDS plot to explain performance patterns")

        print("\n3. CROSS-VALIDATION STRATEGY:")
        print("   - For robust CV: Ensure test well covers different clusters")
        print("   - For optimistic CV: Test on similar wells (same cluster)")
        print("   - For pessimistic CV: Test on outlier wells")

        print(f"\n{'=' * 80}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """
    Main execution function.

    Workflow:
    1. Load well data
    2. Compute all similarity metrics
    3. Visualize similarities
    4. Generate recommendations
    """

    print("=" * 80)
    print("WELL SIMILARITY ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\nSTEP 1: Loading well data...")

    wells = read_all_wells_with_dept_to_list(features="all")
    well_dfs = filter_commom_features(wells, ignore=["VS"])

    # Filter wells with VS (for analysis)
    wells_with_vs = [df for df in well_dfs if "VS" in df.columns]

    print(f"Loaded {len(wells_with_vs)} wells with VS data")

    # Define features to analyze (use what you have)
    features_to_analyze = ["VP", "RHO", "POROSIDADE", "SATURACAO"]

    # =========================================================================
    # STEP 2: Compute Similarities
    # =========================================================================
    print("\nSTEP 2: Computing well similarities...")

    analyzer = WellSimilarityAnalyzer(wells_with_vs, features_to_analyze)
    distance_matrices = analyzer.compute_all_distances()

    # =========================================================================
    # STEP 3: Visualize
    # =========================================================================
    print("\nSTEP 3: Creating visualizations...")

    visualizer = WellSimilarityVisualizer(analyzer)
    visualizer.create_all_plots()

    # =========================================================================
    # STEP 4: Recommendations
    # =========================================================================
    print("\nSTEP 4: Generating recommendations...")

    recommender = WellGroupingRecommendation(analyzer)
    recommender.generate_summary_report()

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll visualizations saved to: well_similarity_analysis/")
    print("\nKey outputs:")
    print("  - MDS plots: Show 2D well relationships")
    print("  - Dendrograms: Show hierarchical grouping")
    print("  - Heatmaps: Show pairwise distances")
    print("  - Console output: Recommendations and insights")
    print("\n" + "=" * 80)


def full_analysis():
    """
    Main execution function.

    Workflow:
    1. Load well data
    2. Compute all similarity metrics
    3. Visualize similarities
    4. Generate recommendations
    """

    print("=" * 80)
    print("WELL SIMILARITY ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\nSTEP 1: Loading well data...")

    wells = read_all_wells_with_dept_to_list(features="all")
    well_dfs = filter_commom_features(wells, ignore=["VS"])

    # Filter wells with VS (for analysis)
    wells_with_vs = [df for df in well_dfs if "VS" in df.columns]

    print(f"Loaded {len(wells_with_vs)} wells with VS data")

    # Define features to analyze (use what you have)
    features_to_analyze = ["VP", "RHO", "POROSIDADE", "SATURACAO"]

    # =========================================================================
    # STEP 2: Compute Similarities
    # =========================================================================
    print("\nSTEP 2: Computing well similarities...")

    analyzer = WellSimilarityAnalyzer(wells_with_vs, features_to_analyze)
    distance_matrices = analyzer.compute_all_distances()

    # =========================================================================
    # STEP 3: Visualize
    # =========================================================================
    print("\nSTEP 3: Creating visualizations...")

    visualizer = WellSimilarityVisualizer(analyzer)
    visualizer.create_all_plots()

    # =========================================================================
    # STEP 4: Recommendations
    # =========================================================================
    print("\nSTEP 4: Generating recommendations...")

    recommender = WellGroupingRecommendation(analyzer)
    recommender.generate_summary_report()

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll visualizations saved to: well_similarity_analysis/")
    print("\nKey outputs:")
    print("  - MDS plots: Show 2D well relationships")
    print("  - Dendrograms: Show hierarchical grouping")
    print("  - Heatmaps: Show pairwise distances")
    print("  - Console output: Recommendations and insights")
    print("\n" + "=" * 80)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print(recommender.identify_well_clusters())
    print("=" * 80)

    recommended_clusters = recommender.identify_well_clusters()
    return recommended_clusters


if __name__ == "__main__":
    main()
