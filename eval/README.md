# 评测集规范

密码与安全文献评测题使用 JSONL，每行一题。较大的评测集可以用 JSON 清单组合多个 JSONL 文件。

当前文件：

- `crypto_pilot_20.jsonl`：20 条先导题；
- `crypto_expansion_80.jsonl`：80 条扩展题；
- `crypto_eval_100.json`：组合上述两部分，共 100 条。

## 单题字段

```json
{
  "query_id": "crypto-001",
  "question": "问题文本",
  "query_type": "definition",
  "answerable": true,
  "reference_answer": "经过核验的参考答案",
  "relevant_sources": ["NIST.FIPS.180-4"],
  "relevant_sections": ["5.1.1"],
  "gold_evidence": [
    {
      "source": "NIST.FIPS.180-4",
      "section_id": "5.1.1",
      "quote": "支持答案的短证据摘录"
    }
  ],
  "review_group": "expansion-reviewed",
  "reviewer": "reviewer-id",
  "review_status": "verified"
}
```

`query_id` 必须全局唯一。`answerable=false` 时，`relevant_sources`、`relevant_sections` 和 `gold_evidence` 应为空，并在参考答案中解释为什么语料不足。

当前覆盖的 `query_type`：

- `definition`：定义、符号和安全目标；
- `parameter_comparison`：同一方案或跨方案参数比较；
- `multi_hop`：需要综合两处或两个来源以上的证据；
- `claim_verification`：判断主张是否被来源支持；
- `version_routing`：区分现行、旧版和替代关系；
- `unanswerable`：当前语料没有充分证据，应拒答。

## 多文件清单

```json
{
  "schema_version": 1,
  "name": "crypto-eval-100",
  "parts": [
    "crypto_pilot_20.jsonl",
    "crypto_expansion_80.jsonl"
  ],
  "expected_examples": 100
}
```

相对路径以清单所在目录为基准。加载时会检查总题数和重复 `query_id`。

## 校验

```powershell
$env:PYTHONPATH = "src"
python -B -m rag_starter.validate_eval_set `
  --eval-file eval/crypto_eval_100.json `
  --manifest data/corpus_manifest.jsonl `
  --min-examples 100 `
  --require-reference-answer `
  --require-verified `
  --validate-sections
```

`--validate-sections` 会真实解析 manifest 中的文档，检查每个相关来源、章节和 gold evidence 是否存在，而不只检查 JSON 格式。

## 检索指标

指标均在 source level 计算：

- `Hit@K`：是否至少命中一个相关来源；
- `Recall@K`：命中的不同相关来源数 / 全部相关来源数；
- `MRR@K`：首个相关来源排名的倒数；
- `nDCG@K`：考虑所有相关来源和排名位置。

同一来源返回多个 chunk 不会重复增加 Recall 或 nDCG。不可回答题没有 qrels，单独记为 `unjudged_queries`，不进入检索指标分母。

## 生成指标边界

- `citations_valid` 只检查最终 `Citations:` 行中的 `[1]` 这类编号存在且未越界；
- `citation_source_precision` 只检查被引 chunk 中有多少来自 gold source；
- `citation_source_recall` 只检查 gold source 中有多少实际被引用；
- 上述自动指标不能证明引文内容蕴含答案，不能改名为 Citation Correctness 或 Faithfulness。

生成结果的 `evaluation` 字段允许填入 0 到 1 的人工或独立 judge 分数：

```json
{
  "evaluation": {
    "answer_correctness": 1.0,
    "citation_correctness": 1.0,
    "citation_completeness": 0.5,
    "faithfulness": 1.0,
    "reviewer": "reviewer-id",
    "notes": "缺少第二项限制条件"
  }
}
```

标注后可直接重新汇总，无需重复调用模型：

```powershell
python -B -m rag_starter.score_generation `
  --input outputs/generation_eval_reviewed.jsonl
```

拒答 Precision、Recall 和 F1 由 `answerable` gold 标签与系统 `refused` 预测计算。

## 数据治理

当前 100 题是开发集，已用于错误分析和规则调优。对外报告泛化能力前应：

1. 由另一名复核者检查问题、答案、来源和证据；
2. 冻结语料版本、切分配置、路由规则和阈值；
3. 新建未参与调优的测试集；
4. 只在测试集上运行一次正式对照实验。
