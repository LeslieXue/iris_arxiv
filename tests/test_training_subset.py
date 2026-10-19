import unittest

import numpy as np

from iris.sampling import subsample_train_ids


class TrainingSubsetTest(unittest.TestCase):
    def test_stratified_one_third_subset_is_deterministic(self) -> None:
        ids = np.arange(90)
        labels = np.repeat(np.arange(3), 30)

        first = subsample_train_ids(ids, labels, fraction=1 / 3, seed=42)
        second = subsample_train_ids(ids, labels, fraction=1 / 3, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 30)
        selected_labels = labels[first]
        self.assertEqual(np.bincount(selected_labels).tolist(), [10, 10, 10])

    def test_full_fraction_preserves_training_ids(self) -> None:
        ids = np.array([7, 2, 9])
        labels = np.array([0, 1, 2])

        selected = subsample_train_ids(ids, labels, fraction=1.0, seed=42)

        np.testing.assert_array_equal(selected, ids)


if __name__ == "__main__":
    unittest.main()
