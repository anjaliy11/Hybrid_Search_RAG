"""Evaluation endpoint — score answers on demand."""

import logging
from fastapi import APIRouter

from src.evaluation.hallucination import HallucinationDetector
from src.evaluation.llm_judge import LLMJudge
from src.serving.schemas import EvalRequest, EvalResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_detector = None
_judge = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    return _detector


def _get_judge():
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest):
    """Evaluate an answer for quality and hallucinations."""
    context = "\n\n".join(request.contexts)

    hall = _get_detector().detect(request.answer, context)
    judge_result = _get_judge().judge(
        request.question, request.answer, request.ground_truth
    )

    return EvalResponse(scores=judge_result, hallucination=hall)