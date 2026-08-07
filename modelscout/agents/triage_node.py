"""Triage Agent: cheap, local, zero-cost pre-filter using HF zero-shot
classification (facebook/bart-large-mnli, CPU). Called for every candidate
model by process_model_node (see graph.py) before it decides whether to
call the expensive Extractor Agent -- it exists specifically so that
Anthropic call only fires for models that pass here.

Classifies against the model's NAME + HF TAGS, not README prose. This was
not the first approach tried -- classifying cleaned README text (even after
stripping YAML frontmatter, badge markup, changelog lists, and code blocks)
produced unreliable, sometimes-backwards results verified empirically
against real model cards: bart-large-mnli is trained on short sentence-pairs
and degrades on long, heterogeneous documents regardless of how much noise
is stripped first. HF's own `tags` field is short, curated by the uploader,
and a much better match for the model's short-text training distribution.
Tradeoff worth knowing: a small number of low-quality community uploads
carry misleading tags copied from a base model (verified case: a GGUF
text-only "uncensored" merge tagged with "vision, multimodal" scored high
despite not being a real vision model) -- this is a real data-quality limit
of the free HF tag signal, not a bug in the classifier call itself.

Convention: interest_profile.triage.candidate_labels[0] is the "relevant"
hypothesis; any other labels are alternative/negative framings shown to the
classifier for contrast but NOT used to gate relevance directly. Classified
with multi_label=True so each label gets its own independent entailment
score instead of a forced softmax across labels -- with multi_label=False,
nearly everything scored as relevant regardless of actual content.
"""

from __future__ import annotations

import threading

from transformers import pipeline

from modelscout.agents.state import ModelState


_MODEL_NAME = "facebook/bart-large-mnli"

_classifier = None
_classifier_lock = threading.Lock()


def _get_classifier():
    # Loaded once per process, not once per model -- this matters even at
    # 20-model demo scale (model load dwarfs a single classification call).
    # LangGraph's Send-based fan-out runs triage_node in a thread pool, so a
    # bare @lru_cache singleton isn't safe here: multiple threads can all see
    # an empty cache before the first one finishes, and each independently
    # triggers a full ~1.6GB model load (verified empirically -- an early
    # version of this function produced 4+ concurrent "LOAD REPORT" logs for
    # a 7-candidate batch). Double-checked locking fixes it.
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = pipeline("zero-shot-classification", model=_MODEL_NAME, device=-1)
    return _classifier


def _classification_text(model_id: str, tags: list[str]) -> str:
    name = model_id.split("/")[-1].replace("-", " ").replace("_", " ")
    return f"{name}. Tags: {', '.join(tags)}."


def classify_relevance(model_id: str, tags: list[str], candidate_labels: list[str]) -> tuple[str, float, dict]:
    """Returns (target_label, target_label_score, full_label_to_score_dict).
    target_label is always candidate_labels[0] -- multi_label scoring means
    its score is independent of the other labels, not a ranking among them.
    """
    classifier = _get_classifier()
    text = _classification_text(model_id, tags)

    result = classifier(text, candidate_labels=candidate_labels, multi_label=True)
    labels_to_scores = dict(zip(result["labels"], result["scores"]))
    target_label = candidate_labels[0]
    target_score = labels_to_scores[target_label]
    return target_label, target_score, labels_to_scores


def triage_node(state: ModelState) -> dict:
    profile = state["interest_profile"]
    candidate_labels = profile["triage"]["candidate_labels"]
    threshold = profile["triage"].get("relevance_threshold", 0.6)

    target_label, target_score, all_scores = classify_relevance(
        state["model_id"], state.get("tags", []), candidate_labels
    )
    is_relevant = target_score >= threshold

    return {
        "triage_label": target_label,
        "triage_confidence": target_score,
        "is_relevant": is_relevant,
        "raw_labels": all_scores,
    }
