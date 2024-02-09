import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rvt.utils import dataset_liv


class TestGetLIVEmbeddings(unittest.TestCase):
        
    def setUp(self):
        # Mocking a demo object
        self.demo_example = [Mock(front_rgb=np.random.rand(32, 32, 3),
                                  wrist_rgb=np.random.rand(32, 32, 3)) for _ in range(10)]

        # Mocking the liv_model
        self.mock_model = Mock()
        self.mock_model.side_effect = lambda input, modality: torch.randn(input.size(0), 128)

    def test_embeddings_length(self):
        front_emb, wrist_emb = dataset_liv._get_liv_embeddings(
            self.demo_example, self.mock_model, "cpu")
        self.assertEqual(len(front_emb), 10)
        self.assertEqual(len(wrist_emb), 10)

    def test_embeddings_device(self):
        front_emb, wrist_emb = dataset_liv._get_liv_embeddings(
            self.demo_example, self.mock_model, "cpu")
        self.assertTrue(front_emb.device == torch.device("cpu"))
        self.assertTrue(wrist_emb.device == torch.device("cpu"))

    def test_batching(self):
        # Making the demo size 130 so we have 3 batches: 64, 64, 2
        demo_large = [Mock(front_rgb=np.random.rand(32, 32, 3),
                           wrist_rgb=np.random.rand(32, 32, 3)) for _ in range(130)]
        front_emb, wrist_emb = dataset_liv._get_liv_embeddings(
            demo_large, self.mock_model, "cpu", 64)
        self.assertEqual(len(front_emb), 130)
        self.assertEqual(len(wrist_emb), 130)


if __name__ == "__main__":
    unittest.main()
