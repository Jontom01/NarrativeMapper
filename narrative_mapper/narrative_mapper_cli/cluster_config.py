import math
from dataclasses import dataclass

@dataclass
class ClusteringBaseline:
    name: str
    base_n_components: int
    base_n_neighbors: int
    base_min_cluster_size: int
    base_min_samples: int
    baseline_tokens: int
    baseline_avg_tokens: int
    min_dim: int = 5
    max_dim: int = 50

    def scale_params(self, total_tokens, avg_tokens, num_texts, verbose):
        # Scaling based on dataset size
        size_scale = max(1, (total_tokens / self.baseline_tokens) * 0.75)

        # Inverse scaling: longer texts = fewer clusters needed
        length_inverse_scale = max(0.6, min(1.0, self.baseline_avg_tokens / avg_tokens))

        # Scaling n_components with data size and verbosity
        verbosity_boost = min(1.5, avg_tokens / self.baseline_avg_tokens)
        dim_base = math.log2(num_texts) * verbosity_boost
        n_components_scaled = min(self.max_dim, max(self.min_dim, int(dim_base)))

        # Final param dictionary
        params = {
            "n_neighbors": int(self.base_n_neighbors * size_scale),
            "n_components": n_components_scaled,
            "min_cluster_size": int(self.base_min_cluster_size * size_scale * length_inverse_scale),
            "min_samples": int(self.base_min_samples * length_inverse_scale),
        }

        if verbose:
            print(f"[{self.name.upper()} PARAM SCALING]")
            print(f"Total tokens: {total_tokens}")
            print(f"Avg tokens per text: {avg_tokens:.2f}")
            print(f"Text count: {num_texts}")
            print(f"n_neighbors: {params['n_neighbors']}")
            print(f"min_cluster_size: {params['min_cluster_size']}")
            print(f"min_samples: {params['min_samples']}")
            print(f"n_components: {params['n_components']}")

        return params


# Define preset clustering modes
BASELINE_MODES = {
    "standard": ClusteringBaseline(
        name="standard",
        base_n_components=10,
        base_n_neighbors=20,
        base_min_cluster_size=50,
        base_min_samples=10,
        baseline_tokens=25000,
        baseline_avg_tokens=40
    ),
    "long_form": ClusteringBaseline(
        name="long_form",
        base_n_components=20,
        base_n_neighbors=15,
        base_min_cluster_size=30,
        base_min_samples=5,
        baseline_tokens=50000,
        baseline_avg_tokens=100
    ),
    "short_form": ClusteringBaseline(
        name="short_form",
        base_n_components=8,
        base_n_neighbors=10,
        base_min_cluster_size=10,
        base_min_samples=5,
        baseline_tokens=10000,
        baseline_avg_tokens=30
    )
}
