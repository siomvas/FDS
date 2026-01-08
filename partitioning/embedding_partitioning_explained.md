# Embedding-Based Partitioning Framework

## The Pipeline

The generation process follows five stages:

1.  **Extraction**: A foundation model converts images into high-dimensional semantic vectors.
2.  **Token Strategy**: Specific layers or tokens (CLS vs. Patches) are selected to represent the image.
3.  **Pooling**: Multi-layer or multi-token representations are aggregated into a single vector per image.
4.  **Preprocessing**: (Optional) Vectors are normalized or projected to a lower-dimensional manifold (UMAP).
5.  **Clustering & Mapping**: Algorithms group similar vectors. Each group becomes one Client.

---

## Key Design Choices & Parameters

### 1. Foundation Models (`--extractor`)
We use models pre-trained on massive datasets to capture universal features.
* **`dinov2`**: Self-supervised. Best for pure visual similarity and object geometry.
* **`clip`**: Text-image aligned. Best for capturing high-level semantic concepts.
* **`siglip`**: Improved CLIP formulation. Higher accuracy on zero-shot tasks.
* **`mae`**: Masked Autoencoder. Captures low-level pixel dependencies.

### 2. Token Extraction Strategy (`--tokens`)
Decides *what* information is pulled from the Vision Transformer.
* **`cls`**: Uses only the [CLS] classification token.
    * *Rationale:* Compact global summary.
* **`k_last_cls`**: Extracts [CLS] tokens from the `lastk_k` layers.
    * *Rationale:* Different layers capture different abstractions (texture vs. object parts vs. semantics). Concatenating them provides a richer hierarchical representation.
* **`all_last` / `k_last_all`**: Extracts *all* spatial patch tokens from the last/the last `lastk_k` layers, respectively.
    * *Rationale:* Preserves spatial information, better captures background latent information.

### 3. Aggregation Strategies
When extracting multiple layers or multiple tokens, we must reduce dimensions.

#### Layer Aggregation (`--layer_aggregation`)
*Used when `lastk_k > 1`*
* **`concat`**: Concatenates features from `lastk_k` layers.
    * *Rationale:* Preserves maximum information. Resulting vector is `last_k * embed_dim` large.
* **`weighted`**: Learned or heuristic weighted average.
    * *Rationale:* Emphasizes deeper (more semantic) layers while retaining some low-level signal.
* **`mean`**: Simple average across layers.

#### Token Pooling (`--pooling`)
* **`mean`**: Average all spatial tokens. Standard baseline.
* **`max`**: Element-wise maximum. Captures the most prominent feature in the map.
* **`gem` (Generalized Mean)**: $$x = (mean(|x|^p))^{1/p}$$.
    * *Rationale:* Acts as a tunable "soft max". Focuses on salient regions (objects) while ignoring background noise.
* **`spectral`**: Projects tokens onto the eigenvectors of their similarity graph.
    * *Rationale:* "Sharpens" the representation by focusing on the dominant structure of the image.

### 4. Preprocessing (`--dim_reduction`, `--normalize`)
* **L2 Normalization**: Self-explanatory.
* **UMAP**: Uniform Manifold Approximation and Projection.
    * *Rationale:* High-dimensional vectors (e.g., 1536d) suffer from the "curse of dimensionality." UMAP learns the local manifold topology and projects it to low dimensions (e.g., 50d), making density-based clustering significantly more stable.

### 5. Clustering (`--clustering`)
* **`hdbscan`**: Density-based.
    * *Pros:* Auto-detects number of clients. Handles noise.
    * *Use Case:* Natural discovery of semantic groups.
* **`kmeans`**: Centroid-based.
    * *Pros:* Fixed number of clients ($K$).
    * *Use Case:* Controlled benchmarks (e.g., "Split TinyImageNet into exactly 10 clients").
* **`knn` / `spectral`**: Graph-based. Captures complex non-convex cluster shapes.

### 6. Validation Split (`--valset_type`)
* **`global`**: All clients validate on a central, IID test set.
* **`local`**: **Semantic Validation Splitting**.
    * *Rationale:* If Client A trains on "Birds", they should not validate on "Cars". We project validation images into the training manifold (using the same UMAP reducer) and assign them to the nearest client centroid.

---

## Example Configurations

**1. The "Deep Hierarchical" Split (Rich Features)**
> Uses hierarchical features from the last 4 layers of DINOv2 to distinguish subtle classes.
* `--tokens k_last_cls` + `--lastk_k 4`
* `--layer_aggregation weighted`
* `--clustering hdbscan`

**2. The "Salient Object" Split (Focus on Foreground)**
> Uses GeM pooling to ignore background clutter in complex scenes.
* `--extractor dinov2-large`
* `--tokens all_last` (Spatial tokens)
* `--pooling gem` + `--gem_p 3.0`
* `--clustering kmeans`

**3. The "Standard Benchmark" (Fixed Budget)**
> Simple global semantics for reproducible FL benchmarking.
* `--extractor clip-base`
* `--tokens cls`
* `--clustering kmeans` + `--n_clusters 10`