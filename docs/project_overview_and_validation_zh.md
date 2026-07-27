# CryptoSec Verifiable Agentic RAG：项目说明与验收指南

## 1. 这是什么项目

这是一个面向密码学与安全标准文档的、可验证的 Agentic RAG 系统。

它解决的不是“让大模型随便回答密码学问题”，而是下面几个更具体的问题：

1. 如何建立来源可信、版本明确的密码学文档库；
2. 如何把 PDF、RFC XML/TXT 解析成带页码和章节的检索单元；
3. 如何从多篇密码学规范中找全证据，而不是只命中一篇看起来相关的文档；
4. 如何处理标准版本替代、跨文档比较和中英文术语错配；
5. 如何让回答带可追溯引用，并在证据不足时拒答；
6. 如何用严格指标证明某项改进是否有效；
7. 如何记录数据、参数、随机种子、环境、延迟和结果，使实验可以复现。

因此，更准确的项目定位是：

> 一个面向密码学与安全标准文献的领域 RAG 检索、路由、引用和评测框架。

它目前更偏“算法与评测项目”，不是以 UI 为核心的聊天产品，也不是训练基础大模型的项目。

## 2. 项目要解决的痛点

普通 RAG demo 往往只完成：

```text
上传 PDF -> 向量检索 -> 拼接 Prompt -> 大模型回答
```

这种做法在密码学文档上有几个明显问题：

- 标准文档有版本和替代关系，旧 RFC 与现行 RFC 不能混为一谈；
- 密码学问题经常需要对照两份规范，例如 RFC 2104 与 FIPS 198-1；
- 参数、公式、算法编号和章节号非常重要，纯语义相似不一定可靠；
- PDF 公式、表格、竖排页边文字会导致解析噪声；
- 只检查引用格式不能证明引用内容真正支持答案；
- Top-K 命中任意一篇相关文档只是 Hit@K，不是真正的 Recall@K；
- 如果没有人工题集和错误分析，很难证明 Agent、Rerank 或 Hybrid 是否真的有用。

本项目把这些问题拆成“语料治理、结构化解析、检索、Agent 路由、生成、引用、拒答、评测”八个部分分别处理。

## 3. 整体架构

```text
人工 Manifest
    ↓
NIST / RFC 白名单下载与文件校验
    ↓
PDF / RFC XML / RFC TXT 结构化解析
    ↓
Document / Chunk 元数据传递
    ↓
按章节、页码切分
    ↓
BM25 / Embedding / Hybrid / Rerank / Agent 检索
    ↓
查询分解、主题路由、版本路由、跨文档证据留位
    ↓
回答或拒答
    ↓
引用格式、引用来源、人工语义指标评测
    ↓
实验参数、数据哈希、延迟、Token、成本记录
```

可以把系统理解成三个层次：

| 层次 | 主要职责 |
|---|---|
| 数据层 | 文档来源、版本、下载、解析、章节、页码和 chunk |
| 检索与 Agent 层 | 找证据、拆问题、选版本、跨文档覆盖、融合排序 |
| 生成与评测层 | 回答、拒答、引用、严格指标、错误分析和实验追踪 |

## 4. 采用的方法

### 4.1 人工 Manifest 与白名单下载

项目没有直接用不受控制的爬虫抓取整个互联网，而是人工维护：

```text
data/corpus_manifest.jsonl
```

每篇文档记录：

```text
doc_id
title
source_type
version
status
published_at
canonical_url
download_url
local_path
sha256
license
topics
supersedes
superseded_by
quality_review
```

这样做的原因是：

- 能明确知道每个答案来自哪份官方文档；
- 能记录标准版本和替代关系；
- 能防止下载器跳转到未授权域名；
- 能冻结某次实验使用的数据集合；
- 比“从某个网页临时抓文本”更容易复现。

下载器只允许：

- NIST 文档来自 `nvlpubs.nist.gov`；
- RFC 文档来自 `www.rfc-editor.org`；
- 只使用 HTTPS；
- NIST 响应必须是真 PDF；
- RFC 响应必须符合 XML 或 RFC TXT 的内容特征；
- 单文件大小不能超过限制；
- 下载后记录 SHA-256、文件大小、最终 URL 和检查时间。

### 4.2 结构化 Document / Chunk

`Document` 保存整篇文档的信息，`Chunk` 保存可检索证据片段的信息。

除文本外，每个 chunk 还带有：

```text
doc_id
title
source_type
version
status
canonical_url
topics
supersedes
superseded_by
section_id
section_title
page_start
page_end
start_char
end_char
chunk_id
```

这些字段让系统可以回答：

- 证据来自哪篇文档；
- 是哪个版本；
- 位于哪一章、哪一页；
- 是否已经被新版本替代；
- 属于 TLS、HMAC、HKDF、PQC 还是其他主题。

### 4.3 按章节和页码切分

项目不再把所有文档都当成连续字符串硬切。

不同格式采用不同策略：

- NIST PDF：提取页码和标题结构，在可靠时按章节切分；
- 标题识别不可靠的 PDF：主动退回按页切分；
- RFC XML：保留 XML 的层级章节；
- RFC TXT：识别带编号的正式章节标题；
- 单个章节过长时：再在章节内部按字符窗口切分，并保留章节元数据。

这样既能控制 chunk 长度，也不会丢失“这个片段属于哪一节、哪一页”的信息。

### 4.4 BM25 稀疏检索

BM25 是当前密码学开发集的基础检索器。

它适合：

- 算法名：AES、HKDF、Argon2；
- 文档编号：RFC 5869、FIPS 203；
- 参数名：nonce、salt、Label、Context；
- 精确符号和固定术语。

优点是快速、可解释、不需要下载额外模型。缺点是中文问题与英文规范之间存在措辞差异时，纯词法匹配可能失败。

### 4.5 Embedding、Hybrid 与 Rerank

项目还保留了通用检索组件：

- Embedding：把 query 和 chunk 编码为向量，按语义相似度检索；
- Hybrid：使用 RRF 融合 BM25 与 Embedding；
- Rerank：使用 Cross-Encoder 对候选 query-chunk 对重新打分。

它们主要用于 SciFact 消融和后续领域模型实验。当前密码学正式开发集选择 BM25 作为 Agent 的基础检索器，是因为：

- 领域术语和编号很多；
- BM25 本身已经有较高覆盖；
- 确定性 Agent 更容易解释每次提升来自哪里；
- 不需要把模型下载、显存或网络环境混入最小复现实验。

### 4.6 确定性 Agent 检索

Agent 不调用 LLM，而是生成一个可审计的 `QueryPlan`。

它实现：

1. 查询分解；
2. RFC/FIPS/SP 编号识别；
3. 主题路由；
4. 版本路由；
5. 跨文档比较证据留位；
6. RRF 多路结果融合；
7. 显式来源和主题来源加权。

例如：

```text
“HKDF 的 info 与 SP 800-108 的 Label/Context 有什么共同目的？”
```

系统需要同时覆盖：

- RFC 5869：HKDF；
- NIST SP 800-108：基于 PRF 的 KDF。

普通 BM25 可能让其他包含 KDF 的文档占满 Top-K。Agent 会识别 HKDF、SP 800-108 和“共同”这种比较意图，为两份来源分别保留证据位置。

### 4.7 回答、拒答与引用

生成层支持：

- Mock 生成器：无需 API key，验证工程链路；
- OpenAI Responses API；
- OpenAI-compatible Chat API。

系统会：

1. 检索证据；
2. 根据证据数量和可选阈值判断是否拒答；
3. 把证据编号后加入 Prompt；
4. 要求模型输出引用；
5. 解析最终 `Citations:` 行；
6. 检查引用编号是否存在、是否越界；
7. 记录引用来源、延迟、token usage 和成本。

需要注意：

> 引用编号合法，只说明 `[1]` 这样的格式正确，不说明第 1 条证据真的支持答案。

真正的引用正确性仍需人工或独立 judge 评估。

### 4.8 严格评测

检索指标在 source level 计算，同一文档返回多个 chunk 只计一个来源。

| 指标 | 含义 |
|---|---|
| Hit@K | Top-K 是否至少包含一个相关来源 |
| Recall@K | Top-K 找回的不同相关来源数 / 全部相关来源数 |
| MRR@K | 第一个相关来源排名的倒数 |
| nDCG@K | 综合考虑多个相关来源及其排序位置 |

简单例子：

```text
Gold sources = [A, B]
Top-3 sources = [X, A, A]
```

则：

- Hit@3 = 1，因为至少找到了 A；
- Recall@3 = 1/2，因为 B 没找到；
- MRR@3 = 1/2，因为第一个相关来源在第 2 位；
- 重复出现的 A 不会把 Recall 变成 2/2。

生成侧记录：

- 引用格式合法率；
- 引用来源精确率/召回率代理指标；
- 拒答 Precision、Recall、F1；
- Answer Correctness；
- Citation Correctness；
- Citation Completeness；
- Faithfulness；
- 延迟、token 和成本。

最后四个语义指标必须有人工或 judge 标签，没有标签时会显示 `N/A`。

## 5. 项目如何与密码学文档结合

### 5.1 语料内容

当前语料包含 44 篇官方标准/规范，覆盖：

- SHA-2、SHA-3、cSHAKE、KMAC；
- AES、GCM、ChaCha20-Poly1305、Ascon；
- HMAC、HKDF、SP 800-108 KDF；
- RSA、ECDSA、EdDSA、X25519、X448；
- TLS 1.3、HPKE、COSE、OpenPGP、MLS；
- Argon2、scrypt、PBKDF2；
- ML-KEM、ML-DSA、SLH-DSA；
- XMSS、LMS/HSS；
- hash-to-curve、VRF、OPRF、OPAQUE；
- 密钥管理、安全强度和算法迁移。

### 5.2 版本关系

密码标准会更新或被替代，例如：

```text
RFC 8446 -> RFC 9846
```

Manifest 用：

```text
supersedes
superseded_by
status
version
```

记录这种关系。

如果问题明确问“当前、最新、现行”，Agent 会降低旧版权重；如果问题是在核验旧版原文，则不会默认删除旧文档。

### 5.3 跨文档比较

密码学问题经常不是单文档问答：

- RFC 2104 与 FIPS 198-1 如何描述 HMAC；
- RFC 5869 的 HKDF 与 SP 800-108 的 KDF 有什么共同目标；
- RFC 8446 与 RFC 9846 的 TLS 1.3 版本关系；
- RFC 8439 的 ChaCha20-Poly1305 与 RFC 9053 的 COSE 配置有什么区别。

Agent 的证据留位与来源融合专门处理这种情况。

### 5.4 协议上下文优先

同一个密码原语可能出现在多个协议中。

例如：

- “Argon2”通常指向 RFC 9106；
- “OpenPGP Argon2 S2K”应优先查看 RFC 9580；
- “ChaCha20-Poly1305”通常指向 RFC 8439；
- “COSE 的 ChaCha20-Poly1305”应同时考虑 RFC 9053。

项目在错误分析中发现了这种误路由，并加入协议上下文规则，而不是只按算法关键词强制跳到唯一文档。

### 5.5 可验证引用

密码学答案不只返回文档名，还可以包含：

```text
文档标题 + 版本 + Section + Page
```

这让使用者能回到官方规范中检查：

- 参数是否抄对；
- 限制条件是否遗漏；
- 引用是否来自现行版本；
- 模型是否把两个协议的规定混在一起。

## 6. 已实现的主要功能

| 功能 | 主要文件 |
|---|---|
| Document / Chunk 数据结构 | `src/rag_starter/schema.py` |
| PDF、RFC XML/TXT、Manifest 加载 | `src/rag_starter/loaders.py` |
| 章节/页码感知切分 | `src/rag_starter/chunking.py` |
| 白名单下载与哈希状态 | `src/rag_starter/corpus/download.py` |
| 语料解析质量检查 | `src/rag_starter/corpus/inspect.py` |
| 结构化语料搜索辅助 | `src/rag_starter/corpus/search.py` |
| BM25 检索 | `src/rag_starter/retrieval.py` |
| Embedding / FAISS / Hybrid / Rerank | `src/rag_starter/*_retrieval.py` |
| Agent 查询计划与路由 | `src/rag_starter/agent_retrieval.py` |
| 回答、拒答与引用处理 | `src/rag_starter/answer_generation.py` |
| 检索严格指标 | `src/rag_starter/retrieval_metrics.py` |
| 生成与拒答指标 | `src/rag_starter/generation_metrics.py` |
| 评测集校验 | `src/rag_starter/validate_eval_set.py` |
| 检索实验 | `src/rag_starter/eval_retrieval.py` |
| 生成实验 | `src/rag_starter/eval_generation.py` |
| 实验参数与环境记录 | `src/rag_starter/experiment_tracking.py` |
| 单元测试 | `tests/` |
| CI | `.github/workflows/` |

## 7. 如何逐步验证项目

以下命令均在仓库根目录执行。

### 7.1 安装与环境

```powershell
python -m pip install -e .
$env:PYTHONPATH = "src"
```

解释：

- `pip install -e .` 以 editable 模式安装项目，修改源码后无需重复打包；
- `PYTHONPATH=src` 让 Python 直接从 `src/` 查找 `rag_starter`；
- 安装成功时不代表算法正确，只代表依赖和包结构可以加载。

验证导入：

```powershell
python -c "import rag_starter; print('import=passed')"
```

期望：

```text
import=passed
```

### 7.2 下载并校验语料

```powershell
python -B -m rag_starter.corpus.download `
  --manifest data/corpus_manifest.jsonl `
  --delay 0.4 `
  --timeout 45 `
  --retries 2
```

典型输出：

```text
[1/44] verified-existing NIST.FIPS.180-4 bytes=833315 sha256=0455b406d896
...
[44/44] verified-existing RFC.9807 bytes=274100 sha256=a0949b0d58b8
downloaded_or_verified=44
failed=0
state_file=data/corpus_state.json
```

逐项解释：

- `downloaded`：本次新下载；
- `verified-existing`：本地已有文件，本次重新检查内容类型并计算哈希；
- `bytes`：文件大小；
- `sha256`：哈希前 12 位，便于快速比对；
- `downloaded_or_verified=44`：44 篇全部成功；
- `failed=0`：没有下载或校验失败；
- `state_file`：完整哈希、URL、时间、响应头等记录位置。

验收标准：

- 必须 `failed=0`；
- 成功数应等于 manifest 条目数；
- 如果出现 `FAILED`，不能继续把缺失文档当成完整语料跑正式评测；
- `verified-existing` 不是“跳过检查”，而是确认本地文件仍然有效。

### 7.3 检查 20 篇文档的解析质量

```powershell
python -B -m rag_starter.corpus.inspect `
  --manifest data/corpus_manifest.jsonl `
  --limit 20 `
  --output-json outputs/corpus_quality_20.json `
  --output-md outputs/corpus_quality_20.md
```

当前代码和语料的参考输出：

```text
documents=20
status_counts={'pass': 13, 'review': 7}
characters=1895453
sections=983
chunks=2164
output_json=outputs/corpus_quality_20.json
output_md=outputs/corpus_quality_20.md
```

解释：

- `documents=20`：抽检 20 篇，不是只加载到 20 篇；
- `pass=13`：自动规则没有发现问题；
- `review=7`：文本可用，但存在需要人工关注的布局或字符告警；
- `review` 不等于失败；
- `characters`：提取出的总字符数；
- `sections`：识别出的章节/页级结构总数；
- `chunks`：使用当前切分参数得到的检索片段数。

常见 issue：

| Issue | 含义 | 如何处理 |
|---|---|---|
| `replacement_characters` | 公式中出现 Unicode 替换字符 | 不把损坏公式作为 gold evidence |
| `rotated_text` | PDF 存在旋转字符 | 检查表格或页边竖排文字 |
| `weak_section_detection` | 标题结构不够可靠 | 使用按页切分，保留真实页码 |
| `too_little_text` | 提取文本过少 | 通常应判定失败并更换解析器 |

验收标准：

- 不应出现 `fail`；
- `review` 文档应结合生成的 Markdown 和原 PDF 抽查；
- 不能只追求章节数量越多越好，错误标题会产生大量假章节；
- 最终应优先保证引用边界真实。

### 7.4 校验 100 条评测题

```powershell
python -B -m rag_starter.validate_eval_set `
  --eval-file eval/crypto_eval_100.json `
  --manifest data/corpus_manifest.jsonl `
  --min-examples 100 `
  --require-reference-answer `
  --require-verified `
  --validate-sections
```

期望输出：

```text
examples=100
query_types={'claim_verification': 16, 'definition': 14, 'multi_hop': 23, 'parameter_comparison': 32, 'unanswerable': 10, 'version_routing': 5}
answerability={'answerable': 90, 'unanswerable': 10}
validation=passed
```

解释：

- `examples=100`：组合清单确实加载了 20 + 80 条题；
- `query_types`：题型分布；
- `answerability`：90 条有语料证据，10 条应拒答；
- `validation=passed`：JSON 结构、来源 ID、章节 ID、gold evidence、参考答案和复核字段均通过。

它不能证明：

- 参考答案一定完全正确；
- gold evidence 一定是最佳证据；
- `verified` 已经完成独立第二人复核。

因此这是“结构与引用可用性校验”，不是最终学术正确性认证。

### 7.5 运行 BM25 基线

```powershell
python -B -m rag_starter.eval_retrieval `
  --config configs/retrieval_crypto_100_bm25.json
```

参考输出：

```text
retriever=bm25
examples=100
judged_queries=90
Hit@10=0.9667
Recall@10=0.9611
MRR@10=0.8588
nDCG@10=0.8821
mean_retrieval_latency_ms=5.7149
```

解释：

- `examples=100`：总题数；
- `judged_queries=90`：有相关来源标签、可以计算检索指标的题数；
- 另外 10 条不可回答题不进入检索指标分母；
- `Hit@10=0.9667`：90 条可回答题中约 96.67% 至少命中一个相关来源；
- `Recall@10=0.9611`：所有题的 gold source 平均找回约 96.11%；
- `MRR@10=0.8588`：第一个相关来源通常排得较靠前，但不是总在第 1 名；
- `nDCG@10=0.8821`：多来源覆盖与排序总体较好，仍有改进空间；
- 延迟会随机器状态变化，指标在数据和配置不变时应保持稳定。

报告文件：

```text
outputs/retrieval_crypto_100_bm25.json
```

主要字段：

```text
run                 参数、环境、Git commit、数据哈希
metrics.overall     总体指标
metrics.groups      按题型和复核组分组的指标
query_details       每题 gold、Top-K、得分和逐题指标
retrieval_latency_ms 延迟均值、P50、P95
```

### 7.6 运行 Agent 对照

```powershell
python -B -m rag_starter.eval_retrieval `
  --config configs/retrieval_crypto_100_agent.json
```

参考输出：

```text
retriever=agent
examples=100
judged_queries=90
Hit@10=1.0000
Recall@10=1.0000
MRR@10=0.9778
nDCG@10=0.9802
mean_retrieval_latency_ms=23.2217
```

与 BM25 对比：

| 方法 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.9667 | 0.9611 | 0.8588 | 0.8821 |
| Agent + BM25 | 1.0000 | 1.0000 | 0.9778 | 0.9802 |

可以解释为：

- Agent 找全了开发集 Top-10 中的所有 gold source；
- 首个相关来源更常排在第 1 名；
- 多来源题的整体排序更好；
- 代价是平均延迟约为 BM25 的 4 倍。

报告中的 `query_details[].agent_plan` 可以查看：

```text
subqueries
explicit_sources
routed_sources
comparison
current_version
```

这是验证 Agent 是否真的执行查询分解和路由的关键证据。

重要限制：

> 这 100 条题参与过错误分析和规则调优，所以 1.0000 是开发集结果，不是未见测试集泛化成绩。

正确的下一步是冻结规则，再制作一套隐藏测试集。

### 7.7 运行 Mock 生成、引用和拒答测试

```powershell
python -B -m rag_starter.eval_generation `
  --config configs/generation_crypto_100_mock.json
```

参考输出：

```text
queries=100
citation_format_valid_rate=1.0000
refusal_precision=0.6364
refusal_recall=0.7000
refusal_f1=0.6667
answer_correctness=N/A labeled=0
citation_correctness=N/A labeled=0
citation_completeness=N/A labeled=0
faithfulness=N/A labeled=0
```

解释：

- `citation_format_valid_rate=1.0000`：回答中的引用编号格式全部合法；
- 它不代表引用内容一定支持答案；
- `refusal_precision=0.6364`：系统拒答的题中约 63.64% 确实应该拒答；
- `refusal_recall=0.7000`：10 条应拒答题中识别出 7 条；
- `refusal_f1=0.6667`：单一 BM25 分数阈值的拒答能力一般；
- 四个语义指标为 `N/A`，因为 mock 结果没有人工评分。

当前混淆矩阵：

```text
正确拒答 TP = 7
错误拒答 FP = 4
漏拒答   FN = 3
正确回答 TN = 86
```

这个实验能证明：

- 100 题生成管线可以完整运行；
- 引用解析和结果写入正常；
- 拒答指标计算正常；
- 分组、延迟和 token 记录接口正常。

它不能证明：

- 模型答案正确；
- 引用语义正确；
- 系统已经具备生产级拒答能力。

输出：

```text
outputs/generation_crypto_100_mock.jsonl
outputs/generation_crypto_100_mock.summary.json
```

JSONL 每行是一道题，适合人工逐条复核；summary 文件保存汇总指标。

### 7.8 调用真实模型

```powershell
python -B -m rag_starter.eval_generation `
  --docs data/corpus_manifest.jsonl `
  --eval-file eval/crypto_eval_100.json `
  --retriever bm25 `
  --llm-provider openai-chat `
  --api-key-env YOUR_API_KEY_ENV `
  --openai-base-url YOUR_BASE_URL `
  --llm-model YOUR_MODEL `
  --input-cost-per-1m INPUT_USD `
  --output-cost-per-1m OUTPUT_USD
```

参数解释：

- `api-key-env` 填环境变量名，不是直接写 key；
- `openai-base-url` 填兼容服务地址；
- `llm-model` 填真实模型名；
- 两个 cost 参数用于计算本次实验成本。

真实模型运行后应检查：

- `input_tokens`、`output_tokens`、`total_tokens` 是否非空；
- `generation_latency_ms` 是否合理；
- `estimated_cost_usd` 是否按价格生成；
- 回答是否只使用检索证据；
- 引用是否对应相关陈述；
- 不可回答题是否拒答。

如果供应商不返回 usage，token 字段会保留为 `null`，项目不会用字符数冒充真实账单 token。

### 7.9 运行单元测试

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

当前期望：

```text
Ran 29 tests
OK
```

测试覆盖：

- 下载白名单和文件魔数；
- Document / Chunk 元数据；
- PDF、RFC XML/TXT 章节解析；
- Agent 版本、主题和协议上下文路由；
- Hit、Recall、MRR、nDCG；
- 引用解析；
- 拒答指标；
- 100 题组合清单与校验。

`OK` 表示这些确定性行为符合测试预期，但不能替代真实数据和模型的人工检查。

## 8. 如何判断整个项目“验收通过”

可以使用下面的验收表：

| 检查项 | 通过标准 | 当前状态 |
|---|---|---|
| Manifest | 44 个唯一 `doc_id` | 通过 |
| 文档下载 | `downloaded_or_verified=44`，`failed=0` | 通过 |
| 解析质量 | 20 篇中无 `fail` | 通过 |
| 题集结构 | `validation=passed` | 通过 |
| BM25 基线 | 指标可复现、报告可生成 | 通过 |
| Agent 对照 | 有明确提升和逐题 QueryPlan | 通过 |
| 引用链路 | 100 题格式检查可运行 | 通过 |
| 拒答链路 | 有混淆矩阵和 F1 | 通过，但效果仍需提升 |
| 语义生成评测 | 有人工评分接口 | 接口通过，尚未完成真实标注 |
| 单元测试 | 29 项全部 `OK` | 通过 |
| 未见测试集 | 冻结后独立评测 | 尚未完成 |

因此当前项目可以被称为：

> 已完成领域语料、结构化解析、Agent 检索和严格评测闭环的开发版本。

但还不应被称为：

> 已经在独立测试集和真实模型上证明生产可用的密码学问答系统。

## 9. 结果中最值得讲的部分

### 9.1 不是只做了一个 RAG 页面

项目真正有价值的工作是：

- 建立可追溯密码语料；
- 修正指标口径；
- 制作 100 条领域题；
- 做跨文档和版本路由；
- 保存逐题 QueryPlan；
- 做错误分析和负向消融；
- 证明哪些策略有效、哪些策略会回归。

### 9.2 有负向实验，不只展示最好数字

项目曾尝试“默认降低所有旧版 RFC 权重”，结果损害了需要核验旧版原文的问题，因此撤回。

这说明：

- 版本状态不能脱离查询意图；
- Agent 规则不是越多越好；
- 每项规则都需要对照实验。

### 9.3 指标解释严格

项目明确区分：

- Hit 与 Recall；
- 引用格式与引用正确性；
- Mock 链路与真实答案质量；
- 开发集结果与未见测试集结果；
- `review` 告警与解析失败。

这比单纯展示一个高分更能体现算法工程能力。

## 10. 面试时可以怎么介绍

### 30 秒版本

> 我做了一个面向密码学和安全标准文档的可验证 Agentic RAG。语料包括 44 篇 NIST 和 RFC 官方规范，我实现了白名单下载、版本和章节元数据、PDF/RFC 结构化切分、BM25 基线、查询分解、主题与版本路由、跨文档证据融合，以及严格的 Hit、Recall、MRR、nDCG、引用和拒答评测。开发集上 Agent 相比 BM25 把 Recall@10 从 0.9611 提升到 1.0000、MRR@10 从 0.8588 提升到 0.9778，同时保留了逐题 QueryPlan 和错误分析。

### 2 分钟版本

> 这个项目的目标不是简单做一个上传 PDF 的聊天页面，而是解决密码学标准文档中的可追溯检索问题。我先人工维护 44 篇 NIST/FIPS/SP 和 RFC 的 manifest，记录版本、主题、官方 URL 和替代关系，再用白名单下载器校验来源和文件内容。解析阶段对 PDF 保留页码和章节，对 RFC XML/TXT 保留层级结构，然后生成带版本、章节和页码的 chunk。
>
> 检索侧先建立 BM25 基线，再根据 100 条密码学开发题的错误分析加入确定性 Agent。Agent 会拆解复合问题、识别 RFC/FIPS/SP 编号、按主题路由、处理现行和旧版关系，并为跨文档比较保留多个来源。多路结果用 RRF 融合。最终开发集 Hit@10 和 Recall@10 达到 1.0，MRR@10 从 BM25 的 0.8588 提升到 0.9778。
>
> 我也专门区分了引用格式、引用来源和引用语义正确性，并实现拒答 Precision/Recall/F1、token、成本和延迟记录。项目还保留了失败实验，例如默认降权所有旧 RFC 会造成回归。当前结果属于开发集，下一步是冻结规则、做独立复核和隐藏测试集。

## 11. 下一步如何继续

优先级建议：

1. 请第二名复核者检查 100 题的 reference answer 与 gold evidence；
2. 冻结当前语料、切分和 Agent 规则；
3. 制作 50–100 条未见测试题；
4. 接入一个真实模型，人工标注四个语义指标；
5. 改进拒答：综合来源覆盖、分数分布、问题类型和证据一致性；
6. 加入 IACR ePrint 与会议论文，处理公式、定理和证明结构；
7. 对比领域 Embedding、领域 Reranker 与当前规则 Agent；
8. 最后再增加展示页面。

如果用于简历，这个项目下一阶段最重要的不是继续堆功能，而是得到：

- 独立复核的题集；
- 未见测试集结果；
- 真实模型的正确性与忠实性评分；
- 一组清晰的错误案例和消融结论。
