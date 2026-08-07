from pydantic import BaseModel


class RunPipelineRequest(BaseModel):
    profile_name: str
    limit: int = 20


class SearchResultItem(BaseModel):
    model_id: str
    chunk_text: str
    score: float
    pipeline_tag: str | None
    downloads: int | None
    hf_url: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
