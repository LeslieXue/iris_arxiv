import unittest

import torch

from iris.loss import teacher_pairwise_ranking_loss


def make_loss(teacher_scores: torch.Tensor) -> torch.Tensor:
    queries = torch.tensor([[0.0, 0.0]], requires_grad=True)
    vectors = torch.tensor([[[0.1, 0.1], [2.0, 2.0]]])
    query_ids = torch.tensor([[0, 0]])
    attention_mask = torch.tensor([[True, True]])
    return teacher_pairwise_ranking_loss(
        queries=queries,
        vectors=vectors,
        query_ids=query_ids,
        teacher_scores=teacher_scores,
        attention_mask=attention_mask,
        temperature=1.0,
        min_score_gap=0.05,
    )


class TeacherPairwiseRankingLossTest(unittest.TestCase):
    def test_rewards_high_scored_chunk_being_closer(self) -> None:
        aligned = make_loss(torch.tensor([[3.0, 1.0]]))
        reversed_order = make_loss(torch.tensor([[1.0, 3.0]]))

        self.assertLess(aligned, reversed_order)

    def test_ignores_ties(self) -> None:
        tied = make_loss(torch.tensor([[2.0, 2.0]]))

        self.assertEqual(tied.item(), 0.0)

    def test_backpropagates_to_queries(self) -> None:
        queries = torch.tensor([[0.5, 0.5]], requires_grad=True)
        loss = teacher_pairwise_ranking_loss(
            queries=queries,
            vectors=torch.tensor([[[0.1, 0.1], [2.0, 2.0]]]),
            query_ids=torch.tensor([[0, 0]]),
            teacher_scores=torch.tensor([[3.0, 1.0]]),
            attention_mask=torch.tensor([[True, True]]),
            temperature=1.0,
            min_score_gap=0.05,
        )
        loss.backward()

        self.assertIsNotNone(queries.grad)
        assert queries.grad is not None
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertGreater(queries.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
