from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateRecord(StrictModel):
    """Complete internal record exported from IRIS retrieval."""

    document_id: int
    query_id: int
    chunk_id: int
    candidate_rank: int = Field(ge=1)
    vector_distance: float
    chunk_text: str
    label: int
    label_name: str
    source_path: str


class CandidateForScoring(StrictModel):
    """Safe record that may be sent to the teacher LLM."""

    document_id: int
    query_id: int
    chunk_id: int
    chunk_text: str


class TeacherScore(StrictModel):
    """Structured scores returned by the teacher LLM."""

    document_id: int
    query_id: int
    chunk_id: int

    ai_evidence: int = Field(ge=0, le=4)
    ds_evidence: int = Field(ge=0, le=4)
    pl_evidence: int = Field(ge=0, le=4)

    substantive_relevance: int = Field(ge=0, le=4)
    specificity: int = Field(ge=0, le=4)
    context_completeness: int = Field(ge=0, le=4)
    direct_label_disclosure: int = Field(ge=0, le=4)
    keyword_only_risk: int = Field(ge=0, le=4)

    brief_reason: str
