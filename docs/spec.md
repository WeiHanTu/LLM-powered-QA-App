# LLMQA evolution specification

Status: Phase 0 implemented; SciFact and reviewed project retrieval comparisons published
Last updated: 2026-08-29

## 1. Executive decision

LLMQA will evolve from a tutorial-style single-file chatbot into a measurable RAG system. The
priority order is:

1. trustworthy ingestion and provenance;
2. retrieval quality with an offline baseline;
3. component-level bias detection;
4. explicit, auditable mitigation with utility trade-off measurements;
5. production observability, scale, and deployment.

Adding FAISS without evaluation would only exchange one vector-store implementation for another.
This specification therefore couples every retrieval or fairness technique to a metric and release
gate.

## 2. Scope

### Goals

- Answer questions only from user-provided documents and expose the supporting passages.
- Make embedding, retrieval, reranking, generation, and evaluation independently testable.
- Detect corpus, retrieval, and outcome disparities on declared groups and controlled
  counterfactual cases.
- Mitigate measured retrieval exposure skew without retraining the embedding model.
- Preserve an offline path for tests and audits; provider calls must not be required for CI.

### Non-goals

- Claiming that the system or a foundation model is unbiased.
- Inferring race, gender, age, religion, disability, nationality, or other protected attributes from
  names, prose, images, or embeddings.
- Choosing a “fair” target distribution automatically.
- Deploying the current research controls in high-stakes allocation, employment, credit, housing,
  education, medical, or legal decisions.
- Implementing generator-logit mitigation against an API that does not expose stable class logits.
- Adding multi-provider runtime configuration in the current phase; OpenAI is the only live
  generation and embedding provider.

## 3. Evidence behind the design

- [FAISS](https://faiss.ai/) supplies local dense-vector indexing and supports exact and approximate
  similarity search. `IndexFlatIP` with normalized vectors is the correct transparent baseline for
  cosine similarity; approximate IVF/HNSW/PQ indexes should be introduced only after scale and
  recall measurements justify them.
- [Reciprocal Rank Fusion (SIGIR 2009)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
  combines independently ranked results as a sum of reciprocal ranks. It avoids pretending that
  BM25 and cosine scores share a calibrated numeric scale; LLMQA uses the paper's constant 60 as
  the explicit default and exposes component ranks and scores.
- [BEIR (NeurIPS Datasets and Benchmarks 2021)](https://arxiv.org/abs/2104.08663) found BM25 to be a
  robust zero-shot baseline and reranking/adaptation methods to improve average effectiveness at a
  computational cost. That supports measuring dense, lexical, fused, and reranked paths separately.
- [Does RAG Introduce Unfairness in LLMs? (COLING 2025)](https://aclanthology.org/2025.coling-main.669/)
  finds fairness problems in both retrieval and generation, which argues against treating a prompt
  as a system-wide mitigation.
- [Mitigating Bias in RAG: Controlling the Embedder (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.974/)
  shows that corpus, embedder, and generator biases interact. The result is not permission to
  “reverse-bias” an embedder blindly; it requires task-specific sensitivity and utility experiments.
- [Evaluating the Effect of Retrieval Augmentation on Social Biases (EACL 2026)](https://aclanthology.org/2026.eacl-long.233/)
  reports amplification of biased retrieved context across English, Japanese, and Chinese on gender,
  race, age, and religion cases. LLMQA therefore evaluates the retrieved slate before generation.
- [Fair RAG: End-to-End Fairness Across Retrieval and Generation (ACL Findings 2026)](https://aclanthology.org/2026.findings-acl.1358/)
  proposes a Fair Greedy Reranker (FGR), signed prefix-sensitive NDKL, and confidence-gated logit
  calibration. Phase 0 implements FGR and NDKL. It does not implement logit calibration because
  free-form QA through a closed model API does not expose the assumed classifier logits.
- [BBQ (ACL Findings 2022)](https://aclanthology.org/2022.findings-acl.165/) separates ambiguous
  cases, where abstention is appropriate, from disambiguated cases, where evidence should override
  stereotypes. Phase 2 adopts this distinction for generator evaluation.
- [NIST AI 600-1](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
  recommends documenting subgroup performance, counterfactual and low-context red teaming,
  benchmark limitations, data representativeness, and task-appropriate parity metrics. These become
  release-report requirements rather than optional dashboard decoration.
- [OpenAI's text generation guide](https://developers.openai.com/api/docs/guides/text) recommends the
  Responses API for direct model requests. LLMQA uses it with response storage disabled and keeps
  retrieved passages explicitly separated as untrusted context.

## 4. Threat model

| Stage | Failure | Detection | Mitigation |
|---|---|---|---|
| Corpus | Sources underrepresent or stereotype a group | Provenance inventory; group/intersection coverage | Curate or reweight sources; document exclusions |
| Embedding | Semantically equivalent group references retrieve different evidence | Counterfactual query pairs; score/rank deltas | Evaluate alternative embedders; controlled fine-tuning only after a baseline |
| Retrieval | Early ranks overexpose a labelled group | NDKL, signed NDKL, exposure by prefix | Retrieve `T > K`; FGR toward an explicit target |
| Generation | Context bias is amplified or evidence is ignored | BBQ-style ambiguous/disambiguated cases; citation and abstention metrics | Grounded instructions, abstention, context balancing, then model/prompt comparison |
| Outcome | A sensitive-attribute swap changes a consequential label or score | Counterfactual flip rate (CFR), mean absolute score difference (MASD) | Block release; inspect corpus, retriever, prompt, and model separately |
| Security | A document injects instructions or poisons retrieval | Injection fixtures; source anomaly and duplicate checks | Treat context as data, isolate uploads, validate provenance, add poisoning tests |

Fairness is contextual. Demographic parity, equal opportunity, counterfactual invariance, and source
diversity answer different questions and can conflict. A report must name the chosen criterion and
why it fits the use case.

## 5. Shipped architecture

```text
PDF/DOCX/TXT
    -> safe temporary extraction
    -> token windows with source/page metadata
    +-> OpenAI embeddings -> normalized vectors -> FAISS IndexFlatIP
    |      -> dense-only MMR baseline
    +-> Okapi BM25 lexical index
           -> dense + lexical reciprocal-rank fusion (default)
    -> candidate pool T
       -> direct top K
       -> optional research path: Fair Greedy reranking to K + NDKL before/after
    -> untrusted labelled passages [S1..SK]
    -> OpenAI Responses API
    -> answer + citation validation + visible passages
```

Key contracts:

- `Chunk`: stable ID, text, source, page, and JSON-compatible metadata.
- `EmbeddingProvider`: document and query embedding methods; replaceable by an offline fixture.
- `FaissRetriever`: normalized exact inner-product search, MMR, and non-pickle persistence.
- `BM25Retriever`: deterministic in-memory lexical baseline with explicit tokenization and BM25
  parameters.
- `HybridRetriever`: weighted RRF over dense and BM25 rankings with component diagnostics.
- `evaluate_rankings`: offline Recall@k, MRR, and graded NDCG@k over stable chunk IDs.
- `fair_greedy_rerank`: relevance-preserving selection within the most underexposed group at each
  prefix.
- `audit_exposure`: NDKL plus signed residual exposure for every target group.
- `audit_counterfactual_outcomes`: CFR and MASD over controlled pairs.
- `generate_grounded_answer`: source-labelled context, strict abstention, `store=False`, and citation
  validation.

## 6. Metrics and evaluation data

### Retrieval utility

- Recall@5 and Recall@10: whether labelled supporting chunks appear in the first results.
- MRR@10: how early the first supporting chunk appears.
- NDCG@10: graded relevance with rank discounting.
- Duplicate-context rate and source diversity.
- Index/embedding build time, p50/p95 retrieval latency, and index size.

### Generation utility and grounding

- Answer correctness on a human-reviewed QA set.
- Citation precision and recall at claim level.
- Faithfulness: supported claims divided by factual claims, with a documented grader and a manually
  checked sample.
- Correct abstention on unanswerable questions and false abstention on answerable questions.

### Fairness

- Label coverage: labelled candidate/displayed chunks divided by all chunks. NDKL is invalid when
  required labels are missing.
- NDKL and signed NDKL for prefix exposure against a declared target.
- Utility deltas before/after mitigation: Recall@k, MRR, NDCG, correctness, and latency.
- CFR and MASD on human-reviewed counterfactual pairs where the protected attribute should be
  causally irrelevant.
- For categorical consequential outcomes only: per-group risk difference and equal-opportunity gap.
- BBQ-style bias scores reported separately for ambiguous and disambiguated contexts.

Every report must include sample counts, slices, group intersections where support is adequate,
confidence intervals or paired tests, prompt/model/embedder/index versions, and benchmark
limitations. Tiny slices are reported as insufficient evidence, not as zero disparity.

### Public benchmark portfolio and evidence policy

Public benchmarks provide reproducible, transferable baselines. They do not validate quality or
fairness for LLMQA's eventual users, documents, languages, or policy targets.

| Order | Benchmark | Purpose | Integration decision |
|---|---|---|---|
| 1 | BEIR SciFact | Compare BM25, dense FAISS, dense + MMR, and hybrid RRF | Implement first: 5,183 abstracts, 300 test queries, 339 positive qrels |
| 2 | BRIGHT Robotics | Stress reasoning-intensive retrieval | Add after the SciFact harness is stable; report separately rather than averaging domains |
| 3 | BBQ | Diagnose generator bias in ambiguous versus disambiguated QA | Phase 2 adapter; never report it as retrieval fairness |
| 4 | CRAG | Evaluate factual answers, missing answers, and dynamic knowledge | Defer because its web/KG setup does not match the current uploaded-document product |

Current verified public comparison (2026-08-28): all four LLMQA retrievers ran on the full SciFact
test split (5,183 documents, 300 queries) at `k=10`. OpenAI `text-embedding-3-small` at 1536
dimensions backed the shared dense `IndexFlatIP` index. Dense produced the highest Recall@10
(`0.8536`); equal-weight hybrid RRF produced the highest MRR@10 (`0.6890`) and NDCG@10 (`0.7214`).
Dense + MMR at `lambda=0.75` regressed all three observed means versus plain dense. Paired
10,000-resample query-bootstrap intervals show dense and hybrid improvements over BM25 exclude zero
for all three metrics; the dense + MMR intervals versus BM25 cross zero. These statements apply to
SciFact only and do not establish production quality or fairness.

Dataset governance requirements:

- Download third-party data on demand into ignored local artifacts; do not commit or redistribute
  raw corpora.
- Pin the source URL and published checksum, verify before extraction, reject unsafe archive paths,
  and record the dataset version and original citation in every report.
- Record and honor the original license. SciFact is CC BY-NC 2.0; BRIGHT and BBQ are CC BY 4.0;
  CRAG is CC BY-NC 4.0. These benchmarks are research evidence, not automatically deployable
  product data.
- Keep benchmark adapters dependency-light and deterministic. Network and provider calls must be
  explicit; CI uses small local fixtures only.
- Run every retrieval method on the identical corpus, queries, qrels, cutoff, and embedding model.
  Do not compare scores copied from papers or leaderboards as if LLMQA produced them.
- Publish a README benchmark table or figure only after a complete reproducible run. The figure must
  identify the dataset, split, sample count, model, metric cutoffs, date, and known limitations.
- Retain the 100-case reviewed project set and its versioned qrels. Public results are an additional
  baseline, not a substitute.

## 7. Functional requirements

### FR-1 Ingestion and provenance

- Accept PDF, DOCX, and UTF-8 text.
- Strip path components from upload names and use an isolated temporary directory.
- Preserve source and PDF page number through retrieval and generation.
- Reject unsupported and empty documents with a user-visible error.

### FR-2 Retrieval

- Normalize both stored and query vectors and use FAISS inner product for cosine similarity.
- Return scores, displayed ranks, original ranks, and stable chunk IDs.
- Support a configurable candidate pool and MMR coefficient.
- Support BM25 and weighted RRF without comparing uncalibrated raw score magnitudes.
- Preserve component ranks and scores for every fused result.
- Persist index metadata as JSON and NumPy, never pickle.

### FR-2A Retrieval evaluation

- Read human-reviewed relevance judgments and ranked run outputs from versionable JSONL.
- Report Recall@k, reciprocal rank, and graded NDCG@k per answerable query and in aggregate.
- Reject duplicate/unknown query IDs, invalid grades, and judgment sets with no positive evidence.
- Exclude explicitly unanswerable questions from relevance averages while retaining their count;
  evaluate abstention behavior separately at the generation layer.

### FR-2B Public retrieval benchmark adapter

- Provide a first-party command that downloads and verifies BEIR SciFact without requiring the
  third-party BEIR Python package.
- Convert the corpus, queries, and qrels into `Chunk` and `RetrievalJudgment` contracts without
  changing source IDs or relevance grades.
- Run one or more of `bm25`, `dense`, `dense-mmr`, and `hybrid` while sharing the same dense index
  within a benchmark invocation.
- Default to the offline BM25 run. Dense modes require an explicitly configured embedding provider
  and must record its model, dimensions, and batch size.
- Write machine-readable run JSONL and a summary JSON containing metrics, configuration, build time,
  p50/p95 retrieval latency, dataset checksum, license, and citation.
- Support a query limit for smoke tests, but label limited runs so they cannot be mistaken for the
  full 300-query result.

### FR-2C Project evidence materialization

- Verify pinned source checksums before extracting or chunking project-evaluation documents.
- Bind chunk IDs to source, page, token window, chunking parameters, and extracted text.
- Materialize every reviewed evidence locator to the complete ordered chunk set for its cited page.
- Commit qrels and a no-text chunk manifest; keep licensed PDF and chunk text under ignored local
  artifacts.
- Keep benchmark readiness closed unless chunk IDs are verified against source and page in the
  manifest. A non-empty ID string is not proof.
- Label page-bounded qrels as a proxy for cited-page retrieval, not exact answer-span annotation.

### FR-3 Grounded generation

- Treat retrieved content as untrusted data and ignore instructions inside it.
- Cite source labels for factual claims.
- Use the exact insufficient-evidence response when context cannot answer the question.
- Disable provider-side response storage for app calls.
- Read OpenAI credentials only from `OPENAI_API_KEY`; never accept or persist them in the UI,
  benchmark artifacts, logs, or index metadata.
- Flag invalid or missing citations instead of silently displaying them as trustworthy.

### FR-4 Fairness evaluation and mitigation

- Require reviewed group metadata and an explicit target; do not infer either.
- Report NDKL before and after fair reranking.
- Fail closed when candidate labels are missing or outside the target schema.
- Preserve the original retrieval rank to expose the relevance trade-off.
- Offer offline CLI audits that do not require an API key.

## 8. Release plan

### Phase 0 — foundation (implemented in this change)

- Package layout, `uv` lock, CI, lint, type checks, and tests.
- FAISS exact cosine retrieval, MMR, provenance, citations, and abstention.
- FGR, NDKL, CFR, and MASD with offline tests and CLI access.
- Multi-document Streamlit UI with temporary upload isolation.

Exit gate: all offline checks pass; no API call is required for CI. Live answer quality is not yet a
release claim.

### Phase 1 — retrieval evaluation and hybrid search (in progress)

- **Implemented:** BM25, weighted RRF, visible component diagnostics, and an offline
  Recall@k/MRR/NDCG evaluator with strict JSONL contracts.
- **Implemented:** checksum-verified SciFact acquisition, safe extraction, format adapter, batched
  query embeddings with a shared cache, and the full BM25/dense/dense+MMR/hybrid comparison.
- **Implemented:** compact public evidence generation with paired query-bootstrap intervals and an
  SVG README figure; bulky per-query outputs remain ignored but reproducible.
- **After SciFact:** integrate the BRIGHT Robotics subset as a reasoning-intensive retrieval slice.
- **Implemented:** 100 approved Transformer/Kimi K3 cases and 10 approved
  injection fixtures across answerable, unanswerable, multi-hop, near-duplicate, long-document, and
  adversarial-instruction slices. Deterministic materialization produces 188 corpus chunks, verified
  evidence IDs, and versioned page-bounded qrels.
- **Implemented:** full BM25, dense FAISS, dense+MMR, and hybrid RRF comparison on the same 100 cases.
  Retrieval metrics cover 80 answerable cases grouped into 45 distinct positive relevance sets.
  BM25 leads all query-weighted means and the cluster-macro Recall@10/NDCG@10 means; hybrid's small
  cluster-macro MRR@10 lead is not conclusive against BM25.
- **Conditional:** add a cross-encoder reranker only if NDCG improves enough to justify latency and
  cost.
- **Pending:** select chunk size/overlap from the evaluation, not intuition.

Exit gate: select the evidence-backed default and document utility, latency, and cost. Replacing BM25
requires a meaningful paired improvement; this comparison does not establish one. Unanswerable-query
behavior remains a generation-layer gate.

### Phase 2 — end-to-end bias evaluation

- Build controlled counterfactual RAG cases for the intended domain and document which attributes
  should be irrelevant.
- Run BBQ and culturally appropriate variants as diagnostic benchmarks, not universal scores.
- Compare no-RAG, vanilla-RAG, MMR-RAG, and fair-reranked RAG at fixed model settings.
- Add paired bootstrap confidence intervals and regression thresholds.

Exit gate: no statistically or practically material regression on declared fairness slices; all
mitigations include utility deltas and a human review of failure clusters.

### Phase 3 — corpus governance and adversarial resilience

- Add document/version hashes, provenance manifests, deletion, tenant isolation, and audit logs that
  exclude raw secrets and unnecessary user content.
- Detect duplicates, suspicious instruction density, and retrieval poisoning candidates.
- Add prompt-injection, poisoned-corpus, and malicious-file regression suites.

Exit gate: deletion and tenant-boundary tests pass; red-team findings have owners and severity-based
release rules.

### Phase 4 — scale and deployment

- Benchmark `IndexFlatIP` against HNSW and IVF/PQ at realistic corpus sizes.
- Add a service boundary, authentication, rate limiting, tracing, budget controls, and deployment
  configuration.
- Rebuild indexes reproducibly from versioned manifests and measure recall loss after compression.

Exit gate: load, recovery, observability, privacy, and cost SLOs pass in a staging environment.

## 9. Phase 0 acceptance criteria

- `uv sync --locked --all-groups` succeeds on Python 3.12.
- Ruff, strict mypy, and pytest pass in CI.
- Exact search returns the known nearest fixture; MMR can select a less redundant fixture.
- Saved indexes round-trip without pickle metadata.
- FGR lowers NDKL on a skewed controlled slate and retains original ranks.
- Missing group metadata causes the fairness audit to fail rather than fabricate a score.
- Counterfactual audit reports CFR and MASD from paired fixtures.
- The generator call uses the Responses API with storage disabled.
- A cited answer passes validation; the exact abstention passes without a citation.

## 10. Phase 1 infrastructure acceptance criteria

- BM25 retrieves an exact lexical match independently of embeddings.
- RRF follows `sum(weight / (60 + rank))`, has deterministic tie-breaking, and retains component
  diagnostics.
- Hybrid retrieval rejects dense and sparse indexes built from different ordered corpora.
- Offline evaluation produces checked Recall@k, MRR, and graded NDCG@k values without provider
  calls.
- Documentation labels example data as schema fixtures, not a quality benchmark.
- Project evaluation materialization is byte-reproducible from checksum-pinned PDFs and locked
  dependencies.
- Validation rejects incomplete page mappings, unknown source/page assignments, and unverified
  chunk-ID strings.
- Project reports separate query-weighted means from macro means over distinct positive relevance
  sets, use cluster-level paired intervals, and label overlapping case-family slices as descriptive.
- Clean retrieval over prompt-injection-tagged questions is never reported as prompt-injection
  resistance.

## 11. Public SciFact benchmark acceptance criteria

- The downloader pins the official BEIR archive URL and MD5 and fails before extraction on a
  mismatch.
- Archive extraction rejects absolute paths and parent-directory traversal.
- The adapter preserves all 5,183 corpus IDs and all 300 test query IDs in a full run.
- The same query subset and cutoff feed every requested retriever.
- BM25 runs without an API key; dense modes share one embedding/index build and record provider
  configuration.
- Summary and run artifacts are deterministic except for measured timings and contain an explicit
  `limited_run` flag.
- Offline fixture tests cover acquisition failure, parsing, ranking, metric calculation, and report
  serialization; CI does not download SciFact or call a model provider.
- README tables or figures are generated only from a committed full-run summary, with the CC BY-NC
  2.0 restriction and lack of project-specific validity visible nearby.

## 12. Open decisions requiring domain input

- Intended users and decision stakes.
- Which corpus attributes are legitimate to label and audit.
- Who defines target exposure and affected-community review.
- Required languages and culturally appropriate benchmarks.
- Quality, latency, privacy, and monthly cost budgets.
- Retention and deletion policy if the system moves beyond in-memory local use.
