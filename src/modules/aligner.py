from typing import List
import numpy as np
from .client import TeiClient


class StepAligner:
    def __init__(
        self,
        tei_client: TeiClient,
        similarity_threshold: float = 0.7,
    ):
        """Initialize the StepAligner.

        Args:
            tei_client (TeiClient): Client for the TEI embedding service
            similarity_threshold (float): Threshold for considering steps as related (0-1)
        """
        self.tei_client = tei_client
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1 (List[float]): First vector
            vec2 (List[float]): Second vector

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_step_embeddings(self, steps: List[str]) -> List[List[float]]:
        """Get embeddings for all steps.

        Args:
            steps (List[str]): List of solution steps

        Returns:
            List[List[float]]: List of embeddings for each step
        """
        return self.tei_client.embed(steps)

    def __call__(self, steps: List[str]) -> List[List[int]]:
        """Analyze relationships between all pairs of steps.

        Args:
            steps (List[str]): List of solution steps

        Returns:
            List[List[int]]: Binary matrix where matrix[i][j] = 1 if steps i and j are related
        """
        # Get embeddings for all steps
        embeddings = self._get_step_embeddings(steps)
        n_steps = len(steps)

        # Initialize relationship matrix
        relationship_matrix = [[0 for _ in range(n_steps)] for _ in range(n_steps)]

        # Calculate relationships between all pairs
        for i in range(n_steps):
            for j in range(i + 1, n_steps):  # Only compute upper triangle
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                is_related = 1 if similarity > self.similarity_threshold else 0
                relationship_matrix[i][j] = is_related

            # Each step is related to itself
            relationship_matrix[i][i] = 1

        return relationship_matrix
