import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from wildlife_tools.data import FeatureDataset


def cosine_similarity(a, b, memory_limit: int | None = None, dtype: torch.dtype = torch.float32,
                      device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    Calculate cosine similarity between two sets of vectors.
    Pytorch Equivalent to `sklearn.metrics.pairwise.cosine_similarity`.
    """

    a = torch.as_tensor(a, dtype=dtype).to(device)
    b = torch.as_tensor(b, dtype=dtype).to(device)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected two 2D feature matrices.")

    # FIXME: The occasional RAM OOM error arises from the final result matrix sometimes being too large to fit in CPU 
    #   RAM. Check available memory and expected size of result matrix beforehand before computation (and before
    #   eventually storing it in CPU RAM via `.cpu()`)

    if memory_limit is None:
        similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
    else:
        try:
            # Try computing the full similarity matrix at once on GPU
            similarity = torch.empty((a.shape[0], b.shape[0]), dtype=dtype).to(device)
            similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
        except torch.cuda.OutOfMemoryError:
            # TODO log the OOM error
            # Limit peak memory by computing in blocks, store result part by part in CPU memory
            dtype_size = torch.tensor([], dtype=dtype).element_size()
            block_size = memory_limit // max(1, b.shape[0] * dtype_size)
            block_size = max(1, min(a.shape[0], block_size))  # Clamp to [1, row count of a]

            a_norm = torch.linalg.vector_norm(a, dim=1, keepdim=True).clamp_min_(1e-12)
            b_norm = torch.linalg.vector_norm(b, dim=1, keepdim=True).clamp_min_(1e-12)
            b_normalized = b / b_norm

            similarity = torch.empty((a.shape[0], b.shape[0]), dtype=dtype).to("cpu")
            
            for start in tqdm(range(0, a.shape[0], block_size), desc="Computing cosine similarity"):
                stop = min(start + block_size, a.shape[0])
                a_block = a[start:stop] / a_norm[start:stop]
                try:
                    mul = torch.matmul(a_block, b_normalized.T)
                    if mul.device != similarity.device:  # If mul is not on CPU
                        mul = mul.to(similarity.device)
                except torch.cuda.OutOfMemoryError:
                    # TODO: Log the OOM error
                    # Still OOM error, fallback to CPU computation entirely
                    device = torch.device("cpu")
                    a_block = a_block.to(device)
                    b_normalized = b_normalized.to(device)
                    mul = torch.matmul(a_block, b_normalized.T)
                similarity[start:stop] = mul

    return similarity.cpu().numpy()


class CosineSimilarity:
    """Wraps cosine similarity to be usable in SimilarityPipeline."""

    def __init__(self, memory_limit: int | None, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        self.memory_limit = memory_limit
        self.dtype = dtype
        self.device = torch.device(device)

    def __call__(self, query: FeatureDataset, database: FeatureDataset, **kwargs) -> np.ndarray:
        """
        Calculates cosine similarity given query and database feature datasets.

        Args:
            query (FeatureDataset): Query dataset of deep features.
            database (FeatureDataset): Database dataset of deep features.

        Returns:
            similarity (np.ndarray): 2D numpy array with cosine similarity.

        """

        return cosine_similarity(query.features, database.features, memory_limit=self.memory_limit, dtype=self.dtype,
                                 device=self.device)
