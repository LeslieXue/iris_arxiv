import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def teacher_pairwise_ranking_loss(
    queries: Tensor,
    vectors: Tensor,
    query_ids: Tensor,
    teacher_scores: Tensor,
    attention_mask: Tensor,
    temperature: float,
    min_score_gap: float,
) -> Tensor:
    """Align query distances with teacher preferences within each query."""
    losses = []
    num_queries = queries.shape[0]

    for query_id in range(num_queries):
        valid = (
            attention_mask
            & (query_ids == query_id)
            & torch.isfinite(teacher_scores)
        )
        if valid.sum() < 2:
            continue

        candidate_vectors = vectors[valid]
        candidate_scores = teacher_scores[valid]
        batch_indices = (
            torch.arange(vectors.shape[0], device=vectors.device)
            .unsqueeze(1)
            .expand_as(valid)[valid]
        )
        query = queries[query_id]
        retrieval_scores = -torch.mean(
            (candidate_vectors - query.unsqueeze(0)) ** 2,
            dim=-1,
        )

        teacher_difference = (
            candidate_scores.unsqueeze(1) - candidate_scores.unsqueeze(0)
        )
        same_document = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)
        preferred_pairs = same_document & (teacher_difference > min_score_gap)
        if not preferred_pairs.any():
            continue

        retrieval_difference = (
            retrieval_scores.unsqueeze(1) - retrieval_scores.unsqueeze(0)
        )
        losses.append(
            F.softplus(
                -retrieval_difference[preferred_pairs] / temperature
            ).mean()
        )

    if not losses:
        return queries.sum() * 0.0
    return torch.stack(losses).mean()


def query_similarity_penalty(queries, temperature):
    # queries = F.normalize(queries, p=2, dim=-1)
    distance_matrix = torch.cdist(queries, queries, p=2)
    similarity_matrix = torch.exp(-distance_matrix / temperature)
    return torch.norm(similarity_matrix, p="fro")


def query_distance_penalty(queries, threshold):
    out = F.normalize(queries, p=2, dim=1)
    distance_matrix = torch.cdist(out, out, p=2)
    return F.relu(distance_matrix - threshold).mean()


class QueryLoss(nn.Module):
    def __init__(
        self,
        temperature: float,
        ignore_index: int,
        teacher_weight: float = 0.0,
        teacher_pair_temperature: float = 1.0,
        teacher_min_score_gap: float = 0.05,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.temperature = temperature
        self.teacher_weight = teacher_weight
        self.teacher_pair_temperature = teacher_pair_temperature
        self.teacher_min_score_gap = teacher_min_score_gap

    def components(
        self,
        input: Tensor,
        label: Tensor,
        query: Tensor,
        vectors: Tensor | None = None,
        query_ids: Tensor | None = None,
        teacher_scores: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        classification = self.cross_entropy(input, label)
        penalty = query_similarity_penalty(query, self.temperature)
        teacher = query.sum() * 0.0
        if (
            self.teacher_weight > 0
            and vectors is not None
            and query_ids is not None
            and teacher_scores is not None
            and attention_mask is not None
        ):
            teacher = teacher_pairwise_ranking_loss(
                queries=query,
                vectors=vectors,
                query_ids=query_ids,
                teacher_scores=teacher_scores,
                attention_mask=attention_mask,
                temperature=self.teacher_pair_temperature,
                min_score_gap=self.teacher_min_score_gap,
            )
        weighted_teacher = self.teacher_weight * teacher
        total = classification + penalty + weighted_teacher
        return {
            "total": total,
            "classification": classification,
            "query_penalty": penalty,
            "teacher_ranking": teacher,
            "weighted_teacher": weighted_teacher,
        }

    def forward(
        self,
        input: Tensor,
        label: Tensor,
        query: Tensor,
        vectors: Tensor | None = None,
        query_ids: Tensor | None = None,
        teacher_scores: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self.components(
            input=input,
            label=label,
            query=query,
            vectors=vectors,
            query_ids=query_ids,
            teacher_scores=teacher_scores,
            attention_mask=attention_mask,
        )["total"]
