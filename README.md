# RAG Starter

这是一个给面试项目用的轻量 RAG 雏形。当前版本先实现一个不依赖 API 的 BM25 检索基线，用来跑通：

```text
load docs -> chunk -> retrieve -> cite sources -> evaluate retrieval
```

后续可以逐步加 embedding、FAISS/Chroma、hybrid retrieval、cross-encoder rerank 和 LLM 生成。

## 目录

```text
rag_starter/
  data/sample_docs/        示例文档
  docs/                    复盘文档和实验分析
  eval/eval_set.jsonl      小型检索评估集
  src/rag_starter/         核心代码
  experiments.md           实验结果记录
  requirements.txt         后续扩展依赖
```

## 运行示例

在 `rag_starter` 目录下执行：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.cli --docs data/sample_docs --question "RAG 的主流程是什么？" --top-k 3
```

运行评估：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --docs data/sample_docs --eval-file eval/eval_set.jsonl --top-k 3
```

## 导入公开数据集：BEIR SciFact

SciFact 是一个小型公开检索数据集，包含论文摘要语料、查询和相关文档标注。它适合用来做 BM25、embedding、hybrid retrieval、rerank 的指标对比。

导入命令：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.import_scifact --max-queries 300
```

如果 Python 下载时遇到证书问题，可以先在浏览器下载：

```text
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
```

然后执行：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.import_scifact --zip-path D:\你的下载目录\scifact.zip --max-queries 300
```

导入后会生成：

```text
data/scifact_docs/          转换后的 Markdown 文档库
eval/scifact_eval.jsonl     检索评估集
data/raw/scifact/           原始 BEIR 数据
```

然后可以直接跑 BM25 baseline：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1200 --chunk-overlap 100
```

跑 embedding 语义检索：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --retriever embedding --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz
```

第一次运行会下载 embedding 模型并编码全部 chunk，所以会慢一些。后续会复用 `--embedding-cache` 指定的缓存。
模型文件默认下载到项目内的 `model_cache/`，避免写到系统用户缓存目录。

单条问题也可以切换检索器：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.cli --retriever embedding --docs data/scifact_docs --question "0-dimensional biomaterials show inductive properties." --top-k 5 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz
```

跑 hybrid retrieval：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --retriever hybrid --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50
```

当前 hybrid 使用 RRF 融合 BM25 和 embedding 排名。RRF 不直接比较两种检索器的原始分数，而是根据各自排名累加分数：

```text
fused_score = sum(weight / (rrf_k + rank))
```

跑 rerank：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --retriever rerank --rerank-base hybrid --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --progress-every 50
```

rerank 会先用 `--rerank-base` 指定的检索器召回候选，再用 cross-encoder 对 `(query, chunk)` 成对打分并重新排序。默认 reranker 是：

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 接 LLM 生成答案

当前默认推荐 OpenAI `gpt-5-mini`：它适合这个项目的低成本、低延迟 RAG 生成场景。没有 API key 时，可以先用 `mock` 模式验证完整链路。

mock 模式：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.answer_cli --llm-provider mock --retriever rerank --rerank-base hybrid --docs data/scifact_docs --question "0-dimensional biomaterials show inductive properties." --top-k 5 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --show-evidence --verbose
```

OpenAI Responses API 模式：

```powershell
$env:OPENAI_API_KEY="你的 API key"
$env:PYTHONPATH="src"
python -m rag_starter.answer_cli --llm-provider openai --llm-model gpt-5-mini --retriever rerank --rerank-base hybrid --docs data/scifact_docs --question "0-dimensional biomaterials show inductive properties." --top-k 5 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --show-evidence --verbose
```

OpenAI-compatible Chat Completions 模式，例如 DeepSeek、SiliconFlow 等：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API key"
$env:PYTHONPATH="src"
python -m rag_starter.answer_cli --llm-provider openai-chat --api-key-env DEEPSEEK_API_KEY --openai-base-url https://api.deepseek.com --llm-model deepseek-v4-flash --retriever rerank --rerank-base hybrid --docs data/scifact_docs --question "0-dimensional biomaterials show inductive properties." --top-k 5 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --show-evidence --verbose
```

为了省 token，第一次先 dry-run，不调用 LLM：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.answer_cli --llm-provider openai-chat --api-key-env DEEPSEEK_API_KEY --openai-base-url https://api.deepseek.com --llm-model deepseek-v4-flash --thinking disabled --retriever rerank --rerank-base hybrid --docs data/scifact_docs --question "0-dimensional biomaterials show inductive properties." --top-k 3 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 30 --rerank-candidate-k 30 --max-context-chars 2500 --max-output-tokens 300 --dry-run --show-prompt --verbose
```

确认 `estimated_total_token_budget` 合理后，去掉 `--dry-run --show-prompt` 再正式调用：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.answer_cli --llm-provider openai-chat --api-key-env DEEPSEEK_API_KEY --openai-base-url https://api.deepseek.com --llm-model deepseek-v4-flash --thinking disabled --retriever rerank --rerank-base hybrid --docs data/scifact_docs --question "0-dimensional biomaterials show inductive properties." --top-k 3 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 30 --rerank-candidate-k 30 --max-context-chars 2500 --max-output-tokens 300 --show-evidence --verbose
```

生成层会把 top-k evidence 拼进 grounded prompt，要求模型只基于证据回答、证据不足时拒答，并在末尾输出引用编号。

代码层也加了 guardrail：

```powershell
--min-evidence 1
--min-top-score 0.5
--require-citations / --no-require-citations
```

`--min-evidence` 和 `--min-top-score` 会在调用 LLM 前决定是否拒答；引用检查会在 LLM 输出后检查是否有 `[1]` 这类合法引用编号。

## 生成侧评估

不调用真实 LLM 也可以先用 mock 跑通评估文件，输出每条 answer 供人工检查：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_generation --llm-provider mock --retriever rerank --rerank-base hybrid --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 5 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http --hybrid-candidate-k 50 --rerank-candidate-k 50 --max-examples 50 --output outputs/generation_eval.jsonl
```

这个脚本会统计：

```text
evidence_hit_rate      检索证据是否命中标准相关文档
citation_valid_rate    输出引用编号是否合法
refusal_rate           拒答比例
```

`answer_correctness` 需要人工标注或 judge model，不能只靠 mock 自动判断。

## 消融实验

消融实验就是：固定主流程，只改变一个因素，观察指标变化。例如只改 `chunk_size`、只改 `top_k`、只改检索器，判断到底是哪一步带来了提升。

当前实验分析见：`docs/ablation_analysis.md` 和中文复盘版 `docs/ablation_analysis_zh.md`

运行：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.run_ablation --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --retrievers bm25,embedding,hybrid --chunk-sizes 600,1200,1800 --top-ks 3,5,10 --max-examples 300 --output outputs/ablation_results.csv --progress
```

## 错误分析

对比两个方法，导出 A 命中 B 没命中、B 命中 A 没命中、都没命中的 case：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.analyze_errors --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --method-a bm25 --method-b hybrid --top-k 10 --chunk-size 1200 --chunk-overlap 100 --output outputs/error_analysis.jsonl
```

错误分析比单纯报指标更适合面试，因为你能讲清楚 BM25、embedding、hybrid、rerank 各自失败在哪里。

## FAISS 和 Chroma

当前 embedding 检索默认使用 NumPy 精确相似度搜索，适合几千条文档的小实验。FAISS 是向量相似度搜索库，更适合大规模向量检索。

可选安装：

```powershell
python -m pip install faiss-cpu
```

然后运行：

```powershell
$env:PYTHONPATH="src"
python -m rag_starter.eval_retrieval --retriever faiss --docs data/scifact_docs --eval-file eval/scifact_eval.jsonl --top-k 10 --chunk-size 1200 --chunk-overlap 100 --embedding-cache vector_store/scifact_all-MiniLM_chunk1200.npz --model-cache-dir model_cache_http
```

Chroma 是更像“轻量向量数据库”的工具，除了向量索引，还会管理 collection、metadata 和持久化存储。当前项目先保留为下一步工程化扩展。

## 你后面要加的优化

1. 向量库：当前 embedding 检索先用 NumPy 精确相似度，后续可换 FAISS 或 Chroma。
2. 拒答策略：检索分数过低或证据不足时输出“根据现有资料无法确定”。
3. 生成评估：统计回答正确率、引用命中率、拒答准确率。
4. 实验表格：对比 chunk size、overlap、top-k、是否 hybrid、是否 rerank。
