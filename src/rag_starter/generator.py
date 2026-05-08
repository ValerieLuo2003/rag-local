from __future__ import annotations

from .schema import SearchResult


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = []
    for result in results:
        context_blocks.append(
            f"[{result.rank}] source={result.chunk.source}, chunk={result.chunk.chunk_id}\n"
            f"{result.chunk.text}"
        )

    context = "\n\n".join(context_blocks)
    return f"""你是一个严谨的知识库问答助手。请只基于给定上下文回答问题。
如果上下文中没有答案，请回答“根据现有资料无法确定”。
回答后列出引用编号。

问题：
{question}

上下文：
{context}

答案："""


def evidence_only_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "根据现有资料无法确定。"

    lines = [
        "当前 starter 还没有接入 LLM，先返回可用于生成答案的证据：",
        f"问题：{question}",
        "",
        "Top evidence:",
    ]
    for result in results:
        preview = result.chunk.text.replace("\n", " ")[:220]
        lines.append(
            f"[{result.rank}] score={result.score:.4f} source={result.chunk.source} "
            f"chunk={result.chunk.chunk_id}\n{preview}"
        )
    return "\n".join(lines)

