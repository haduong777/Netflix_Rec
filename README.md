# Netflix_Rec

A comparative analysis of three machine learning approaches for building recommendation systems using Netflix data.

## Dataset

The Netflix dataset is one of the most well-known and widely used resources in recommendation system research. Released during the Netflix Prize competition (2006-2009), the dataset contains anonymized movie ratings provided by Netflix customers over a period spanning several years. It includes approximately 100 million ratings from around 480,000 randomly chosen users across nearly 18,000 movie titles.

Key characteristics of the dataset:
- Ratings range from 1 to 5 stars
- Date of rating is included
- Sparse matrix representation (most users rate only a small fraction of available titles)

## Data Pipeline

[Data preprocessing: parsing, transformation, compression, storage, streaming]

## ML Methods Overview

This project compares 3 Machine Learning approaches for recommendation systems:

### Matrix Factorization

Matrix Factorization (MF) is a collaborative filtering technique that works by decomposing the user-item interaction matrix into lower-dimensional latent factor matrices. The core theory behind MF is that both users and items can be represented in a shared, lower-dimensional latent space where their interactions can be modeled as the inner product of their respective latent representations.

The algorithm learns these latent factors by minimizing the reconstruction error between predicted and actual ratings, typically using stochastic gradient descent or alternating least squares optimization. Mathematically, for a rating matrix $R$ with users $u$ and items $i$, matrix factorization decomposes it into two lower-rank matrices $P$ (user factors) and $Q$ (item factors):

$R \approx P \times Q^T$

The predicted rating $\hat{r}_{ui}$ is computed as the dot product of user and item latent factors:

$\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i = \sum_{f=1}^{k} p_{u,f} \cdot q_{i,f}$

The optimization objective is to minimize the squared error with regularization:

$\min_{P,Q} \sum_{(u,i) \in \mathcal{K}} (r_{ui} - \mathbf{p}_u \cdot \mathbf{q}_i)^2 + \lambda(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)$

Popular variants include Singular Value Decomposition (SVD), Non-negative Matrix Factorization (NMF), and Probabilistic Matrix Factorization (PMF).

MF excels at capturing linear relationships between users and items and has proven highly effective for recommendation tasks with explicit feedback like ratings.

### Autoencoders

Autoencoders are neural network architectures that aim to reconstruct their input after passing it through a bottleneck layer, effectively learning a compressed representation of the data. In recommendation systems, autoencoders can be used to learn non-linear latent representations of users or items.

The core architecture consists of:
- An encoder network that compresses the input (user ratings/interactions) into a lower-dimensional representation
- A bottleneck layer that contains the learned latent factors
- A decoder network that reconstructs the original input from the bottleneck representation

For a user's rating vector $\mathbf{r}_u$, the autoencoder learns:

**Encoder:** $\mathbf{h}_u = f(\mathbf{W}_1 \mathbf{r}_u + \mathbf{b}_1)$

**Decoder:** $\hat{\mathbf{r}}_u = g(\mathbf{W}_2 \mathbf{h}_u + \mathbf{b}_2)$

Where:
- $\mathbf{h}_u$ is the latent representation
- $f$ and $g$ are activation functions (often sigmoid or ReLU)
- $\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2$ are learned parameters

The model is trained to minimize reconstruction error:

$\min_{\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2} \sum_{u} \| \mathbf{r}_u - \hat{\mathbf{r}}_u \|^2 + \lambda \cdot \text{regularization}$

Variants include denoising autoencoders (which add noise to inputs during training), variational autoencoders (which learn probabilistic latent representations), and deep autoencoders (with multiple hidden layers).

Autoencoders excel at capturing complex non-linear patterns in user-item interactions and can naturally handle missing values in the sparse interaction matrix.

### Graph Neural Networks (GNNs)

Graph Neural Networks approach the recommendation problem by modeling users and items as nodes in a bipartite graph, with edges representing interactions. This formulation allows the algorithm to exploit the graph structure of the recommendation problem and capture higher-order connectivity patterns.

The core mechanism of GNNs involves:
- Node feature initialization (embedding users and items)
- Message passing between nodes (aggregating information from neighbors)
- Node representation updates based on received messages
- Multi-layer propagation to capture higher-order connectivity

For a recommendation graph where $\mathcal{N}(i)$ represents the neighbors of node $i$, a GNN layer updates node representations as:

$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{AGGREGATE}\left(\{\mathbf{h}_j^{(l)} : j \in \mathcal{N}(i)\}\right)\right)$

Where:
- $\mathbf{h}_i^{(l)}$ is the representation of node $i$ at layer $l$
- $\text{AGGREGATE}$ is a function that combines neighbor information (e.g., mean, sum, or max)
- $\mathbf{W}^{(l)}$ is a learnable weight matrix
- $\sigma$ is a non-linear activation function

Final predictions for user $u$ and item $i$ can be computed as:

$\hat{r}_{ui} = \sigma(\mathbf{h}_u^{(L)} \cdot \mathbf{h}_i^{(L)})$

Popular GNN variants for recommendations include Graph Convolutional Networks (GCN), GraphSAGE, and Graph Attention Networks (GAT).

GNNs excel at capturing complex collaborative signals by leveraging the graph structure of user-item interactions and can naturally incorporate side information about users and items as node features.

## Results

[results here]
