import numpy as np
import matplotlib.pyplot as plt

def manual_pca(embeddings, n_components=2):
    """
    Step 15 (Visualization): Reduce embedding dimensions using PCA manually.
    Uses covariance matrix and eigen decomposition in pure NumPy.
    """
    # 1. Center the embeddings
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    
    # 2. Compute covariance matrix
    # Rowvar=False because variables (dimensions) are columns, observations (words) are rows.
    cov = np.cov(centered, rowvar=False)
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 4. Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Project onto top components
    components = eigenvectors[:, :n_components]
    projected = np.dot(centered, components)
    return projected

def plot_embeddings(embeddings, word_to_index, save_path="embeddings_pca.png"):
    """
    Plots the projected 2D embeddings using matplotlib.
    """
    projected = manual_pca(embeddings, n_components=2)
    
    plt.figure(figsize=(10, 8))
    for word, idx in word_to_index.items():
        x, y = projected[idx]
        plt.scatter(x, y, color='dodgerblue', edgecolors='k', s=100)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12, fontweight='semibold')
        
    plt.title("Manual PCA 2D Word Embeddings Projection", fontsize=14, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Embedding visualization saved to {save_path}")
