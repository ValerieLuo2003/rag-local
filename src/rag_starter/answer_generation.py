from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .schema import SearchResult


DEFAULT_LLM_MODEL = "gpt-5-mini"
REFUSAL_TEXT = "According to the provided evidence, the answer cannot be determined."


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    provider: str
    model: str
    evidence_count: int
    refused: bool = False
    citations_valid: bool = True
    citation_warning: str = ""


class MockAnswerGenerator:
    def __init__(self, model: str = "mock", max_context_chars: int = 8000) -> None:
        self.model = model
        self.max_context_chars = max_context_chars

    def generate(self, question: str, evidence: list[SearchResult]) -> AnswerResult:
        if not evidence:
            answer = f"{REFUSAL_TEXT}\n\nCitations: none"
            return AnswerResult(
                answer=answer,
                provider="mock",
                model=self.model,
                evidence_count=0,
                refused=True,
            )

        citations = " ".join(f"[{result.rank}]" for result in evidence[:3])
        preview = evidence[0].chunk.text.replace("\n", " ")[:420]
        answer = (
            "Mock answer: this placeholder is used to verify the RAG flow without a paid LLM API.\n\n"
            f"Question: {question}\n\n"
            f"Top evidence preview: {preview}\n\n"
            f"Citations: {citations}"
        )
        return AnswerResult(answer=answer, provider="mock", model=self.model, evidence_count=len(evidence))


class OpenAIResponsesAnswerGenerator:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_output_tokens: int = 700,
        max_context_chars: int = 8000,
        thinking: str = "default",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_output_tokens = max_output_tokens
        self.max_context_chars = max_context_chars
        self.thinking = thinking
        self.client = self._load_client()

    def generate(self, question: str, evidence: list[SearchResult]) -> AnswerResult:
        prompt = build_grounded_prompt(question, evidence, max_context_chars=self.max_context_chars)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
        )
        answer = getattr(response, "output_text", None) or str(response)
        return AnswerResult(
            answer=answer.strip(),
            provider="openai-responses",
            model=self.model,
            evidence_count=len(evidence),
        )

    def _load_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI generation requires openai. Install with: pip install -r requirements.txt") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env}. Set it first, or run with --llm-provider mock.")
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)


class OpenAICompatibleChatAnswerGenerator(OpenAIResponsesAnswerGenerator):
    def generate(self, question: str, evidence: list[SearchResult]) -> AnswerResult:
        prompt = build_grounded_prompt(question, evidence, max_context_chars=self.max_context_chars)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a careful RAG answer generator."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_output_tokens,
        }
        if self.thinking in {"enabled", "disabled"}:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
        response = self.client.chat.completions.create(**kwargs)
        answer = response.choices[0].message.content or ""
        return AnswerResult(
            answer=answer.strip(),
            provider="openai-compatible-chat",
            model=self.model,
            evidence_count=len(evidence),
        )


def build_grounded_prompt(question: str, evidence: list[SearchResult], max_context_chars: int = 8000) -> str:
    context = format_evidence(evidence, max_context_chars=max_context_chars)
    return f"""You are a strict grounded RAG assistant.

Rules:
1. Answer only from the provided evidence.
2. If the evidence is insufficient, answer exactly: "{REFUSAL_TEXT}"
3. Do not add facts from outside the evidence.
4. Use the same language as the user question when possible.
5. End with citations in this format: Citations: [1] [3]
6. Never cite an evidence number that was not provided.

Question:
{question}

Evidence:
{context}

Output format:
Answer:
Citations:"""


def format_evidence(evidence: list[SearchResult], max_context_chars: int = 8000) -> str:
    if not evidence:
        return "none"

    blocks = []
    used_chars = 0
    for result in evidence:
        header = (
            f"[{result.rank}] source={result.chunk.source}, "
            f"chunk_id={result.chunk.chunk_id}, score={result.score:.4f}"
        )
        text = result.chunk.text.strip()
        remaining = max_context_chars - used_chars - len(header) - 8
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "..."
        block = f"{header}\n{text}"
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def should_refuse(
    evidence: list[SearchResult],
    min_evidence: int = 1,
    min_top_score: float | None = None,
) -> tuple[bool, str]:
    if len(evidence) < min_evidence:
        return True, f"only {len(evidence)} evidence chunks found; require at least {min_evidence}"
    if min_top_score is not None and evidence:
        top_score = evidence[0].score
        if top_score < min_top_score:
            return True, f"top score {top_score:.4f} is below threshold {min_top_score:.4f}"
    return False, ""


def refusal_result(provider: str, model: str, evidence_count: int, reason: str) -> AnswerResult:
    answer = f"{REFUSAL_TEXT}\n\nCitations: none\nRefusal reason: {reason}"
    return AnswerResult(
        answer=answer,
        provider=provider,
        model=model,
        evidence_count=evidence_count,
        refused=True,
    )


def postprocess_answer(
    result: AnswerResult,
    evidence_count: int,
    require_citations: bool = True,
) -> AnswerResult:
    detected_refusal = is_refusal_answer(result.answer)
    if detected_refusal:
        return AnswerResult(
            answer=normalize_refusal_citations(result.answer),
            provider=result.provider,
            model=result.model,
            evidence_count=result.evidence_count,
            refused=True,
            citations_valid=True,
            citation_warning="",
        )

    if not require_citations or result.refused:
        return result

    cited = extract_citation_indices(result.answer)
    if not cited:
        warning = "No citations found. Expected references like [1]."
        return append_citation_warning(result, warning)

    invalid = [index for index in cited if index < 1 or index > evidence_count]
    if invalid:
        warning = f"Invalid citation ids: {invalid}. Valid range is 1..{evidence_count}."
        return append_citation_warning(result, warning)

    return AnswerResult(
        answer=result.answer,
        provider=result.provider,
        model=result.model,
        evidence_count=result.evidence_count,
        refused=result.refused,
        citations_valid=True,
        citation_warning="",
    )


def append_citation_warning(result: AnswerResult, warning: str) -> AnswerResult:
    answer = result.answer.rstrip() + f"\n\nCitation check: {warning}"
    return AnswerResult(
        answer=answer,
        provider=result.provider,
        model=result.model,
        evidence_count=result.evidence_count,
        refused=result.refused,
        citations_valid=False,
        citation_warning=warning,
    )


def extract_citation_indices(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[(\d+)\]", text)]


def is_refusal_answer(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    refusal = " ".join(REFUSAL_TEXT.lower().split())
    return refusal in normalized or "cannot be determined" in normalized


def normalize_refusal_citations(text: str) -> str:
    if re.search(r"citations?\s*:\s*none", text, flags=re.IGNORECASE):
        return text.strip()
    if re.search(r"citations?\s*:\s*$", text.strip(), flags=re.IGNORECASE):
        return re.sub(r"(citations?\s*:\s*)$", r"\1none", text.strip(), flags=re.IGNORECASE)
    if re.search(r"citations?\s*:", text, flags=re.IGNORECASE):
        return text.strip()
    return text.strip() + "\n\nCitations: none"


def estimate_tokens(text: str) -> int:
    # Cheap rough estimate for planning API calls. Real billing uses the provider tokenizer.
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + non_ascii_chars / 1.6))


def build_answer_generator(
    provider: str,
    model: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    max_output_tokens: int = 700,
    max_context_chars: int = 8000,
    thinking: str = "default",
):
    if provider == "mock":
        return MockAnswerGenerator(model=model or "mock", max_context_chars=max_context_chars)
    if provider == "openai":
        return OpenAIResponsesAnswerGenerator(
            model=model or DEFAULT_LLM_MODEL,
            api_key_env=api_key_env,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            max_context_chars=max_context_chars,
            thinking=thinking,
        )
    if provider == "openai-chat":
        return OpenAICompatibleChatAnswerGenerator(
            model=model or DEFAULT_LLM_MODEL,
            api_key_env=api_key_env,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            max_context_chars=max_context_chars,
            thinking=thinking,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
