# CS4.406 Information Retrieval and Extraction — Monsoon 2026

A breadth course on the design, construction, and practice of real-world search and
recommender systems. It weaves together three threads — **systems** (storage, indexing,
scaling, latency budgets), **machine learning** (ranking, CTR/CVR, embeddings,
approximate nearest neighbors), and **practice** (evaluation, A/B testing, lifecycle,
industry architectures) — and closes with frontier topics.

**Design stance.** Conventional methods (BM25, PageRank, classic compression, matrix
factorization, …) are *introduced, not dwelt on*. Each is used to surface a transferable
**principle** — compression trades cheap compute for scarce bandwidth; BM25 exposes the
probabilistic ranking principle — rather than to drill mechanics. Every lecture states its
**major takeaway** explicitly and lists the specifics it would cover. Depth is reserved for
ideas that generalize and for building things.

## Format at a glance

- **22 lectures**, 1.5 hours each, **two per week**, grouped into **eight modules**.
- **Semester:** 1 August → 30 November.
- **Written tests (in order):** Quiz-1, Mid-Sem, Quiz-2, End-Sem.
- **Coursework:** a systems-integration build on a real industry challenge (Assignment-1), a
  from-scratch implementation (Assignment-2), a term project, and an **Optional-for-bonus track**
  (up to +5%). Assignments are meant to be *built with coding assistants* (Claude Code, etc.) —
  the exams then verify the student understood the code and theory.

### Reference material

Central texts, with per-module papers below:

1. `IIR` — *Introduction to Information Retrieval.* Manning, Raghavan, Schütze. (Free online.)
2. `SEIRP` — *Search Engines: Information Retrieval in Practice.* Croft, Metzler, Strohman.
3. `MMDS` — *Mining of Massive Datasets.* Leskovec, Rajaraman, Ullman. (Free online.)
4. `DDIA` — *Designing Data-Intensive Applications.* Kleppmann.
5. `PML` — *Personalized Machine Learning.* Julian McAuley. (Free online.)
6. `TOCE` — *Trustworthy Online Controlled Experiments.* Kohavi, Tang, Xu.

**Simulation harness (used in the bonus track and projects):** a reproducible sponsored-ads
marketplace testbed with swappable event stores, lexical/semantic/hybrid search backends,
GSP/VCG/first-price auctions, a funnel choice model, IR evaluation (NDCG/MRR/P@k/R@k),
seeded bootstrap A/B, and bounded-memory streaming KPIs; the repository link is shared in class.

---

## Module 1 — Systems foundations (L1–L3)

### L1. Course overview and the IR problem
- Course mechanics: syllabus, grading, the two assignments, the bonus track, and the term project.
- The two faces of every IR system: engineering (latency, throughput, cost) and functional (relevance, coverage, freshness).
- The two phases: indexing (crawl → process → index) and querying (parse → retrieve → rank → serve).
- Offline vs. online evaluation and why both are needed; deployment via A/B testing.
- How requirements bend by use-case: web search vs. e-commerce vs. news vs. ads.
- Crawling in brief: focused crawling, freshness / age-of-cache, data volume, duplicates.

*Example:* a single web-scale query touches an index of billions of pages yet must answer in
~200 ms — the whole course lives inside that relevance-under-a-latency-budget tension.
> **Takeaway:** IR is the disciplined management of a relevance-vs-cost trade-off under a
> latency budget — not a single algorithm.

**Readings**

- `IIR` Ch. 1 — Boolean retrieval and the IR problem.
- `SEIRP` Ch. 1 — search engines in practice.
- Brin & Page, "The Anatomy of a Large-Scale Hypertextual Web Search Engine" — a canonical end-to-end system overview.
- Olston & Najork, "Web Crawling" (survey) — crawling, focused crawling, and coverage.
- Cho & Garcia-Molina, "Effective Page Refresh Policies for Web Crawlers" — freshness and age-of-cache.

### L2. System design under constraints
- Wide-area systems: replication, partitioning / sharding, and why one box cannot scale.
- CAP: consistency vs. availability under a network partition; the latency corollary (PACELC).
- Anatomy of a request's latency budget: disk seek+read, cross-continent round trips, fan-out/fan-in.
- Little's law (L = λW) and utilization curves; why latency explodes as utilization → 1.
- Tail latency and fan-out amplification: one slow leaf drags the whole response.
- Case studies: local-master (Yahoo) vs. remote-master (Facebook) replication.

*Example:* a service fanning out to 100 shards, each with a 1-in-100 chance of a slow
response, is very likely to have *at least one* slow shard per query — so the aggregate p99 is
far worse than any single leaf, forcing hedged requests or tighter per-leaf budgets.
> **Takeaway:** There is no one-size-fits-all — every design is a chosen point on a
> consistency/availability/latency/cost surface, and you must name the trade-off you took.

**Readings**

- Brewer, "CAP Twelve Years Later: How the Rules Have Changed" — CAP and its nuances.
- Abadi, "Consistency Tradeoffs in Modern Distributed Database Design" — the PACELC refinement.
- Dean & Barroso, "The Tail at Scale" — fan-out amplification and tail-latency mitigation.
- Dean, "Achieving Rapid Response Times in Large Online Services" (Google talk) — latency in web-scale services.
- `DDIA` Ch. 5–6 — replication and partitioning/sharding.
- Barroso, Hölzle & Ranganathan, *The Datacenter as a Computer* (Ch. 1–2) — warehouse-scale context.

### L3. Storage and indexing hardware
- The storage hierarchy: cache → RAM → SSD → HDD → network, and their latency/cost orders of magnitude.
- Sequential vs. random access; why a ~10 ms HDD seek dwarfs a streaming read.
- B-Trees: in-place updates; strong reads and range scans; random-write cost.
- LSM-Trees: log-structured writes, compaction, write/read amplification, Bloom filters to skip.
- Choosing by workload: read- vs. write-heavy, point lookup vs. range scan.

*Example:* an inverted-index build is append-heavy, so an LSM-style log beats in-place B-Tree
updates; a user-profile store dominated by point lookups may prefer a B-Tree — the *workload*
picks the structure.
> **Takeaway:** Design to the hardware — sequential access beats random, and logs beat
> in-place updates when writes dominate.

**Readings**

- `DDIA` Ch. 3 — storage engines: B-Trees vs. LSM-Trees.
- O'Neil et al., "The Log-Structured Merge-Tree (LSM-Tree)" — the original LSM design.
- Chang et al., "Bigtable: A Distributed Storage System for Structured Data" — a production LSM-backed store.
- Bloom, "Space/Time Trade-offs in Hash Coding with Allowable Errors" — Bloom filters for skip decisions.
- Athanassoulis et al., "Designing Access Methods: The RUM Conjecture" — read/update/memory trade-offs.
- Dong et al., "Optimizing Space Amplification in RocksDB" — LSM compaction and amplification in practice.

## Module 2 — Building a queryable index (L4–L5)

### L4. Text processing, deduplication, and sketching
- Query as a noisy channel: typos, synonymy, and underspecification between intent and tokens.
- Document pre-processing: tokenization, case-folding, stemming/lemmatization, stopwords, phrase/n-gram detection.
- Information extraction: named entities, dates, prices, locations; field weighting (title vs. body).
- Corpus statistics: Zipf's law (term frequency) and Heaps' law (vocabulary growth) → estimating posting-list sizes and ordering conjunctions cheapest-first.
- Deduplication: shingling → sets, Jaccard similarity, and why all-pairs comparison is O(n²).
- Sketching: MinHash (unbiased Jaccard estimator, choosing signature length), LSH banding, SimHash, Bloom filters.

*Example:* two near-identical syndicated news copies share ~95% of their 5-shingles; a 200-hash
MinHash estimates that overlap to within a few percent at a fraction of the storage, and LSH
banding surfaces the pair without comparing it to every other document.
> **Takeaway:** At scale you approximate at sublinear cost — compact signatures replace
> quadratic pairwise work, at a controllable accuracy price.

**Readings**

- `IIR` Ch. 2 — tokenization, normalization, and the term vocabulary.
- `MMDS` Ch. 3 — finding similar items: shingling, MinHash, and LSH.
- Broder, "On the Resemblance and Containment of Documents" — MinHash and Jaccard estimation.
- Charikar, "Similarity Estimation Techniques from Rounding Algorithms" — SimHash.
- Manku, Jain & Das Sarma, "Detecting Near-Duplicates for Web Crawling" — SimHash at web scale.
- Sarawagi, "Information Extraction" (survey) — entity/field extraction foundations.
- Manning, Surdeanu et al., "The Stanford CoreNLP Natural Language Processing Toolkit" — a practical NLP/IE pipeline.

### L5. Inverted index, compression, and query processing *(merged)*
- Inverted index anatomy: dictionary + postings (docIDs, term frequencies, positions).
- Query operators over postings: AND/OR/NOT via merges; phrase/positional and field queries.
- The I/O-vs-compute gap: reading two 1 MB posting lists (~0.05 s) vs. merging them (~0.001 s).
- Compression: d-gap (delta) encoding, then unary, Elias-γ/δ (bit-aligned) and v-byte (byte-aligned); skip pointers for binary search over compressed gaps.
- Query-processing strategies: term-at-a-time vs. document-at-a-time; top-k thresholding (WAND/τ), early termination, result caching.
- Robustness: misspellings and query expansion (Rocchio / pseudo-relevance feedback).
- Query understanding and suggestion: intent classification, spelling correction at scale, query segmentation, entity linking, and autocomplete/typeahead (trie + ranking, session-aware suggestion).
- Near-real-time / streaming indexing: incremental index updates, Lucene-style NRT search, and change-data-capture freshness pipelines.

*Example:* v-byte can shrink a posting list ~4×, so it reads in a quarter of the time;
decompression costs far less than the I/O saved, so throughput rises — compute traded for
bandwidth, exactly the bottleneck that matters.
> **Takeaway:** Compression exists to trade cheap compute for scarce bandwidth; the right code
> depends on where your bottleneck sits in the memory hierarchy.

**Readings**

- `IIR` Ch. 4–6 — index construction, compression, and scoring.
- `SEIRP` Ch. 5 — ranking and query processing.
- `MMDS` Ch. 2 — MapReduce (for distributed index build).
- Broder et al., "Efficient Query Evaluation using a Two-Level Retrieval Process" — the WAND top-k algorithm.
- Ding & Suel, "Faster Top-k Document Retrieval Using Block-Max Indexes" — Block-Max WAND.
- Lemire & Boytsov, "Decoding Billions of Integers per Second Through Vectorization" — modern posting-list codecs.
- Cai & de Rijke, "A Survey of Query Auto Completion in Information Retrieval" — autocomplete/typeahead.
- `DDIA` Ch. 11 — stream processing and near-real-time updates.
- Bialecki et al., "Apache Lucene 4" — a production near-real-time inverted index.

## Module 3 — Relevance and ranking (L6–L7)

### L6. Ranking models: from heuristics to learning-to-rank
- Boolean retrieval and its all-or-nothing limitation.
- Vector-space model: tf–idf weighting and cosine similarity.
- Probabilistic ranking principle; BM25 with term-frequency saturation and length normalization.
- Language-model view: query/document likelihood with smoothing (Dirichlet, Jelinek-Mercer).
- Query-independent quality priors: PageRank and HITS.
- The pivot to learning: risk minimization; generalization error (irreducible / estimation / approximation); pointwise, pairwise (RankNet), listwise (LambdaMART) objectives; GBDTs as the workhorse.

*Example:* BM25 alone cannot tell that an authoritative page should outrank a keyword-stuffed
one; a learned ranker that *combines* BM25 + PageRank + click features can — which is precisely
why we move from a fixed formula to learning a combination of signals.
> **Takeaway:** Relevance is a modeling choice; classical scorers are special cases of a learned
> ranker, and learning-to-rank buys the ability to fuse many signals.

**Readings**

- `IIR` Ch. 6, 11, 21 — scoring, probabilistic IR, and link analysis.
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" — BM25 derivation and extensions.
- Zhai & Lafferty, "A Study of Smoothing Methods for Language Models Applied to Ad Hoc Retrieval" — Dirichlet/JM smoothing.
- Page et al., "The PageRank Citation Ranking: Bringing Order to the Web" — query-independent priors.
- Kleinberg, "Authoritative Sources in a Hyperlinked Environment" — HITS.
- Liu, "Learning to Rank for Information Retrieval" (survey) — pointwise/pairwise/listwise framing.
- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview" — the workhorse LTR objectives.
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" — GBDTs.
- Qin et al., "Are Neural Rankers Still Outperformed by Gradient-Boosted Decision Trees?" — a modern LTR baseline check.

### L7. Signals that feed ranking
- Feature families: query, document, query–document match, and user/context features.
- Behavioral signals and click models: position bias, cascade and examination hypotheses.
- CTR/CVR prediction: logistic regression → factorization machines → deep models (Wide & Deep, DeepFM).
- Sparse categorical features: embeddings and the hashing trick.
- Calibration: why a predicted 0.1 CTR must mean ~10% clicks for ad pricing/bidding to work.
- Head vs. tail regimes: count/behavioral features for frequent queries, semantic features for the cold tail.

*Example:* an item shown only in position 1 looks "high CTR," but that is position bias, not
quality; a click model corrects the estimate so a genuinely better item stuck at position 5 is
not unfairly penalized when it becomes training data.
> **Takeaway:** A ranker is only as good as its signals — and logged signals are biased by what
> was shown, so calibration and de-biasing are first-class concerns.

**Readings**

- Chapelle & Zhang, "A Dynamic Bayesian Network Click Model for Web Search Ranking" — position bias and the DBN model.
- Craswell et al., "An Experimental Comparison of Click Position-Bias Models" — cascade/examination hypotheses.
- Rendle, "Factorization Machines" — modeling sparse feature interactions.
- Cheng et al., "Wide & Deep Learning for Recommender Systems" — memorization + generalization.
- Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction."
- McMahan et al., "Ad Click Prediction: a View from the Trenches" — FTRL and production CTR lessons.
- Weinberger et al., "Feature Hashing for Large Scale Multitask Learning" — the hashing trick.
- Guo et al., "On Calibration of Modern Neural Networks" — calibration and temperature scaling.

## Module 4 — Semantic retrieval (L8–L9)

### L8. Embeddings and semantic search
- From lexical mismatch to semantic matching: "sofa" vs. "couch," cross-lingual queries.
- Dual-encoder (two-tower) models: encode query and document independently into a shared space.
- Training: contrastive loss, in-batch negatives, hard-negative mining.
- Bi-encoder (fast retrieval) vs. cross-encoder (accurate re-ranking) trade-off.
- Hybrid retrieval: fusing lexical (BM25) and dense scores.
- Learned-sparse and late-interaction retrieval: doc2query/docTTTTTquery expansion, SPLADE learned-sparse, and ColBERT/PLAID late interaction — bridging lexical and dense.
- Multimodal / cross-modal retrieval: CLIP-style joint embeddings and visual search (Google Lens, Pinterest Lens).

*Example:* the query "noise-cancelling earbuds for flights" should retrieve a product titled
"wireless ANC in-ear headphones" — zero token overlap, but neighbors in embedding space; a
two-tower retriever finds it where BM25 cannot.
> **Takeaway:** Embeddings turn retrieval into geometry — cold-tail recall comes from matching
> meaning, not tokens, at the cost of a training pipeline and a vector index.

**Readings**

- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" — the two-tower DPR.
- Huang et al., "Embedding-based Retrieval in Facebook Search" — dense retrieval in production.
- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
- Xiong et al., "Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)" — hard-negative mining.
- Izacard et al., "Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever)."
- Nogueira & Lin, "From doc2query to docTTTTTquery" — document expansion.
- Formal et al., "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking."
- Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction" (+ "ColBERTv2" and "PLAID").
- Gao & Callan, "Condenser / coCondenser" — retrieval-oriented pretraining.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" — cross-modal embeddings.

### L9. Approximate nearest neighbors at scale
- Why exact NN and kd-trees collapse in high dimensions (curse of dimensionality).
- IVF (coarse quantization) to prune the search space.
- Product quantization (PQ) to compress vectors and approximate distances in memory.
- HNSW graphs: navigable small-world graphs for near-logarithmic search.
- The recall–latency–memory frontier and how each knob (nlist/nprobe, M/efSearch) moves it.
- Vector databases as systems: filtered ANN (ANN + metadata predicates), index freshness/updates, and quantization at the DB layer (Milvus, pgvector, Weaviate).

*Example:* on a billion vectors, exact search is infeasible; IVF-PQ + HNSW returns ~95%
recall@10 in a few milliseconds using a fraction of the raw vectors' memory — trading a few
percent recall for orders of magnitude in speed and RAM.
> **Takeaway:** Recall, latency, and memory form one frontier — ANN is the knob that trades a
> little accuracy for orders-of-magnitude speed and footprint.

**Readings**

- Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs."
- Johnson, Douze & Jégou, "Billion-scale Similarity Search with GPUs (FAISS)."
- Jégou, Douze & Schmid, "Product Quantization for Nearest Neighbor Search."
- Indyk & Motwani, "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality" — LSH foundations.
- Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)."
- Wang et al., "Milvus: A Purpose-Built Vector Data Management System" (SIGMOD 2021).
- Aumüller, Bernhardsson & Faithfull, "ANN-Benchmarks: A Benchmarking Tool for ANN Algorithms."

## Module 5 — Evaluation, practice, and serving (L10–L12)

### L10. Evaluation: offline metrics, human judgments, and LLM-as-judge
- Set metrics: precision, recall, F1, MAP.
- Ranked metrics: MRR and nDCG (gain × discount), and the assumptions each encodes.
- Building test collections: pooling, graded relevance judgments, inter-annotator agreement (κ).
- The offline/online gap: why an nDCG win need not move online engagement.
- LLM-as-judge: pointwise vs. pairwise prompting; agreement with human labels.
- LLM-judge failure modes and fixes: position bias (randomize order), verbosity bias, self-preference bias, ensembling/calibration.

*Example:* swap the order of two candidate answers and an LLM judge often flips its preference
(position bias); averaging both orderings recovers a stable verdict — a vivid reminder that a
judge, human or model, is a biased instrument to be validated before you trust it.
> **Takeaway:** Every metric encodes an assumption about what users want; judges (human or LLM)
> are themselves biased instruments that must be validated before you trust them.

**Readings**

- `IIR` Ch. 8 — evaluation in information retrieval.
- Järvelin & Kekäläinen, "Cumulated Gain-based Evaluation of IR Techniques" — (n)DCG.
- Voorhees, "The Philosophy of Information Retrieval Evaluation" — TREC and pooling.
- Sanderson, "Test Collection Based Evaluation of Information Retrieval Systems" (survey).
- Chapelle et al., "Expected Reciprocal Rank for Graded Relevance" — a cascade-aware metric.
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."
- Thomas et al. (Microsoft), "Large Language Models can Accurately Predict Searcher Preferences."
- Faggioli et al., "Perspectives on Large Language Models for Relevance Judgment."

### L11. Retrieval lifecycle and online experimentation
- The retrieve → filter → rank → blend pipeline: candidate generation, light then heavy rankers, policy filters.
- Feature stores and event DBs; logging; training-serving skew and the data flywheel.
- A/B testing basics: hypothesis, power, minimum detectable effect (MDE), sample size/duration.
- Variance reduction (CUPED); pitfalls: peeking / sequential testing, multiple comparisons, novelty effects.
- Interference and network effects; switchback and cluster-randomized designs; guardrail metrics.
- Interleaving for ranking evaluation: team-draft / probabilistic interleaving — far more sensitive than A/B for ranking changes and standard in web search.

*Example:* an experiment that looks "significant" after being peeked at daily for two weeks is
mostly false positives; fixing the sample size in advance (or using a proper sequential test)
controls the error — the difference between a real win and noise you talked yourself into.
> **Takeaway:** The pipeline — not any single model — is the product; and only a well-designed
> online experiment can tell you whether a change actually helped.

**Readings**

- `TOCE` — Trustworthy Online Controlled Experiments (core text).
- Kohavi et al., "Online Controlled Experiments and A/B Testing" (KDD tutorial/survey).
- Deng et al., "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED)."
- Johari et al., "Peeking at A/B Tests: Why It Matters, and What to Do About It" — always-valid sequential testing.
- Chapelle et al., "Large-Scale Validation and Analysis of Interleaved Search Evaluation."
- Radlinski, Kurup & Joachims, "How Does Clickthrough Data Reflect Retrieval Quality?"
- Bakshy, Eckles & Bernstein, "Designing and Deploying Online Field Experiments (PlanOut)."
- Uber Michelangelo / Feast documentation — feature stores and training-serving skew.

### Lectures L12 and beyond

Other lecture details will be added in due time. They will tentatively cover topics the following areas based on lecture-hours available and student interests.

+ Serving at scale, Recommender foundations, Sequential, graph & knowledge-based recommenders, 
+ Industrial recommendation systems, Real-time & large-scale ranking infrastructure,
+ Open-sourced & search-ranking stacks, Applications: sponsored search, e-commerce & news,
+ Learning from logged bandit feedback, Interaction dynamics & responsible retrieval,
+ LLMs in retrieval & recommendation, Efficiency, cost & the economics of retrieval at scale 

---

## Semester calendar (1 Aug → 30 Nov)

Two lectures per week; the grid sequences the 22 lectures and the four written tests. Exact
weekday slots follow the room timetable; buffer weeks absorb holiday shifts.

| Week | Approx. dates | Lectures | Module & lecture titles | Assessment / milestone |
|------|---------------|----------|-------------------------|------------------------|
| 1 | Aug 1–7 | L1, L2 | *M1 Systems foundations* — L1 Course overview & the IR problem; L2 System design under constraints | Team formation begins; **Assignment-1 (component-1)** released |
| 2 | Aug 8–14 | L3, L4 | L3 Storage & indexing hardware; *M2 Building a queryable index* — L4 Text processing, deduplication & sketching | |
| 3 | Aug 15–21 | L5, L6 | L5 Inverted index, compression & query processing; *M3 Relevance & ranking* — L6 Ranking models: heuristics to learning-to-rank | **Assignment-2** released · **assignment groups declared (15 Aug)** |
| 4 | Aug 22–28 | L7, L8 | L7 Signals that feed ranking; *M4 Semantic retrieval* — L8 Embeddings & semantic search | **Quiz-1 (27 Aug) · A1 component-1 due (27 Aug)**; A1 component-2 released |
| 5 | Aug 29–Sep 4 | L9, L10 | L9 Approximate nearest neighbors at scale; *M5 Evaluation, practice & serving* — L10 Evaluation: offline metrics, human judgments & LLM-as-judge | **Project groups declared (30 Aug)** · **project proposal + initial literature survey due (31 Aug)** |
| 6 | Sep 5–11 | L11, L12 | L11 Retrieval lifecycle & online experimentation; L12 | **A1 component-2 due (10 Sep)** |
| 7 | Sep 12–18 | L13, L14 | *M6 Recommender systems* — L13  L14  | **A2 due (14 Sep)** |
| 8 | Sep 19–25 | (review + buffer) | — | **Mid-Sem (21 Sep)** |
| 9 | Sep 26–Oct 2 | L15, L16 | *M7 Industry architectures* — L15  L16 | **Project problem revised & finalized (30 Sep)** |
| 10 | Oct 3–9 | L17, L18 | L17 *M8 Applications & frontier* — L18 | |
| 11 | Oct 10–16 | L19, L20 | L19  L20  | |
| 12 | Oct 17–23 | L21, L22 | L21 L22 | **Quiz-2 (22 Oct)** (L13–L20) |
| 13 | Oct 24–30 | (review + buffer) | — | |
| 14 | Oct 31–Nov 6 | project work | — | |
| 15 | Nov 7–13 | project presentations | — | |
| 16 | Nov 14–20 | (revision) | — | **Project + optional-bonus build due (20 Nov)** |
| 17 | Nov 21–30 | — | — | **End-Sem (23 Nov)** (comprehensive, L1–L22) |

---

## Assessment structure

| Component | Date | Weight |
|-----------|------|-------:|
| Quiz-1 | 27 Aug | 10% |
| Assignment-1 · component-1 (lexical + semantic retrieval, individual) | 27 Aug | 5% |
| Assignment-1 · component-2 (click-log modelling, pairs) | 10 Sep | 5% |
| Assignment-2 | 14 Sep | 10% |
| Mid-Sem | 21 Sep | 15% |
| Quiz-2 | 22 Oct | 10% |
| Project | 20 Nov | 25% |
| End-Sem | 23 Nov | 20% |
| Optional for bonus (marketplace-simulation full build and research survey) | 20 Nov | up to +10% |
| **Total** | | **100% (+5% bonus)** |

The two assignments plus the project put 45% of the grade on building, and the optional bonus
track can add up to 5% more; the written tests carry the concepts, and a portion of every test
**probes the student's own assignment/project work**. Teams: **up to 2** per assignment (groups
declared by **15 Aug**), **up to 3** for the project (by **30 Aug**) — and even though work is
done in teams, __evaluation will always be individual__.

### Timeline (chronological)

- **1 Aug** — Semester begins; Assignment-1 component-1 released; project starts.
- **15 Aug** — Assignment-2 released (two weeks after Assignment-1); **assignment groups (≤2) declared**.
- **27 Aug** — Quiz-1 (10%); **A1 component-1 due** (5%, individual); A1 component-2 released.
- **30 Aug** — **Project groups (≤3) declared**.
- **31 Aug** — Project: initial proposal + literature survey due (30 days from start).
- **10 Sep** — **A1 component-2 due** (5%, pairs; two weeks after component-1).
- **14 Sep** — **Assignment-2 due** (10%; 30 days from its start).
- **21 Sep** — Mid-Sem (15%).
- **30 Sep** — Project: problem revised & finalized (further literature survey).
- **22 Oct** — Quiz-2 (10%).
- **20 Nov** — Project due (25%); Optional-for-bonus build due (up to +5%).
- **23 Nov** — End-Sem (20%).
- **30 Nov** — Semester ends.

---

## Assignments

Assignment-1 is a **systems-integration build on news recommendation over two datasets — the
RecSys 2024 EB-NeRD challenge and Microsoft's MIND** — in **two components**: component-1
(lexical + semantic retrieval, **individual**, due **27 Aug** at Quiz-1) and component-2
(click-log modelling, **groups of two**, due **10 Sep**). Assignment-2 is a **from-scratch ANN
implementation**, done in **pairs**, released **15 Aug** and due **14 Sep** (30 days).

**Teams and evaluation.** Assignment teams have **up to 2 members**; project teams **up to 3**.
**Assignment groups must be declared by 15 Aug; project groups by 30 Aug.** Even though the work
is done in teams, __evaluation will always be individual__ — via the signed component-ownership
matrix, per-member vivas, and the per-student assignment/project probes in every written test. An **Optional-for-bonus track** (up to
**+5%**) offers a single option — the **marketplace-simulation full build**, which rebuilds the
Assignment-1 spine on the simulation harness and additionally solves a problem requiring
counterfactuals only the simulation can reveal. Students may use Claude Code or similar assistants,
but the exams probe whether they understand the code paths and theory behind what they built. Each
submission includes a short (≤4 page) **design note**: *what did you build, which choices did you
make and why, and what did you observe?* Full rationale, evaluation gates, and the
viva/individual-effort protocol for Assignment-1 are in `revisions/assignment-1-revision.md`.

### Assignment-1 — News recommendation on EB-NeRD *and* MIND · two components · 10%
Students build a working, measured, reproducible pipeline on **both** news datasets: rank the
candidate articles in an impression by click likelihood, using the user's click history, session
context, and article content.

- **EB-NeRD** (RecSys 2024 challenge, Ekstra Bladet): ~2.7M users, 600M+ impression logs, 120k+
  articles with title/abstract/body and provided article embeddings; demo/small/large bundles let
  teams dial scale gradually.
- **MIND** (Microsoft News dataset, <https://msnews.github.io/>): ~1M users, 160k+ English news
  articles with title/abstract/body and entity annotations, 15M+ impression logs; MIND-small for
  fast iteration.

Both tasks natively exercise all three modelling axes — **lexical** (titles/bodies), **semantic**
(article/text embeddings), and **behavioural** (click history, recency/decay) — and rapid news
decay makes temporal splitting and freshness handling instructive. **Both leaderboards are up**,
so students can submit and verify their solutions against them; topping a board is not a
necessity — though always appreciated — and grading is **never** on leaderboard rank.

**Two components:**

1. **Component-1 — lexical & semantic retrieval · individual · due 27 Aug (at Quiz-1) · 5%.**
   Retrieval over both datasets using **lexical and semantic features**: the reproducible data
   pipeline with a temporal split, lexical (BM25) and embedding-based candidate generation, and
   the offline evaluation harness. Done **individually**.
2. **Component-2 — learning from click-logs · groups of two · due 10 Sep · 5%.** Building on
   component-1: **behavioural signals from the click-logs** — click-history/session features,
   the re-ranker, baseline reproduced then beaten with an ablation, and the serving/scale
   analysis. Done in **groups of two** (two weeks after component-1).

**The common spine (across the two components):**

1. **Reproducible data pipeline** *(C1)* — download → clean → **temporal train/val/test split**
   (never random for interaction data) → a small feature store; one command rebuilds from raw files.
2. **Two-stage retrieve-then-rank** — candidate generation (lexical and/or ANN) to a few hundred
   candidates *(C1)*, then a **re-ranker** (GBDT or a small neural ranker) over engineered
   behavioural features *(C2)*.
3. **Offline evaluation harness** *(C1, extended in C2)* — the official metrics (AUC, MRR,
   nDCG@{5,10}, plus beyond-accuracy: diversity, novelty, coverage) with at least one slice
   (cold-start vs. warm, head vs. tail) and **bootstrap confidence intervals**.
4. **Baseline reproduced, then beaten** *(C2)* — reproduce the official/starter baseline, then one
   principled improvement with an **ablation**; claimed gains must ship a paired bootstrap 95% CI
   that excludes zero.
5. **Serving & scale analysis** *(C2)* — index memory, p99 retrieval latency, and a back-of-envelope
   cost/QPS at a target SLA (a measured local benchmark plus a scaling argument suffices).
6. **Design note (≤4 pages)** *(both, one per component)* — what you built, choices and
   alternatives, observations, and where it breaks at 10×.

**Anti-gaming:** report metrics **with and without features unavailable at serving time** (an
organizer requirement); enforce the behaviour-window boundary; verify no future-click leakage.

**Rubric (10%):** reproducible pipeline + correct temporal evaluation 30% · two-stage system with
all three axes 25% · baseline reproduced + ablated improvement 20% · serving/scale/cost analysis
15% · design-note clarity 10%. Individual effort is checked via a signed component-ownership
matrix, a short per-member viva (explain-and-modify on the live repo), and per-team exam probes.


### Assignment-2 — Approximate nearest neighbors from scratch · pairs · released 15 Aug · due 14 Sep · 10%
Where Assignment-1 wires existing systems, Assignment-2 has student pairs **implement one core
method themselves** to internalize its mechanics: an ANN method (e.g., HNSW *or* IVF + product
quantization) for a given large vector dataset. Benchmark recall@k vs. latency vs. memory against
a brute-force baseline and a reference library (FAISS), and map how the frontier moves as index
parameters vary. Deliverable includes the code, the benchmark, and the design note. 

### Optional for bonus · up to +10% · individually

The bonus track has a **single option**: the **marketplace-simulation full build**, worth **up to
+5%**, done solo or in groups of up to two. It follows the same standards as the assignments
(reproducible repo + short design note), is due **20 Nov**, and may not double-count project work.
Optionally, one can take up writing research survey on a chosen topic.

**Part 1 — rebuild Assignment-1 on the simulation harness.** Teams must explicitly rebuild
everything in the Assignment-1 spine, but on the marketplace simulation instead of the EB-NeRD
logs: the reproducible data pipeline, a two-stage retrieve-then-rank over
lexical/semantic/behavioural signals (event-store + lexical backend, query corpus + qrels, funnel
metrics), the offline evaluation harness with bootstrap CIs, a semantic/hybrid backend swap with a
seeded bootstrap A/B, and a defended operating point on the recall–latency–memory–cost frontier
under a serving SLA.

**Part 2 — solve a problem that needs counterfactuals.** In addition, teams pick and solve one
problem that requires observing **counterfactuals** — outcomes under actions the logging policy
never took — which the simulation makes directly observable but Assignment-1's fixed logs never
can. Example ideas:

- **Off-policy estimators vs. ground truth** — deploy a target ranking policy in the simulator to
  observe its *true* reward, then measure how far IPS, self-normalized IPS, and doubly-robust
  estimates computed from logs alone deviate from it, as a function of log size and how much the
  target policy diverges from the logging policy. *(L20.)*
- **Position bias & click-model validation** — serve the *same* result list under swapped or
  randomized positions to measure the true position effect, then test whether click models
  (PBM, cascade) recover it correctly from logs alone. *(L7, L10.)*
- **Feedback loops & popularity bias** — retrain the ranker on its own logged clicks over many
  rounds, and compare against the counterfactual world where an exploration policy generated the
  data; quantify popularity amplification and lost catalog coverage. *(L21.)*
- **The true cost of exploration** — replay *identical* user streams under ε-greedy vs.
  Thompson-sampling exploration to measure realized regret and long-run catalog coverage — the
  same users under two policies, observable only in simulation. *(L20.)*
- **Paired A/B on identical sessions** — run control and treatment on the *same* simulated
  users and sessions to observe per-session treatment effects directly, and quantify how much
  variance (and sample size) a standard between-user A/B or interleaving design wastes relative
  to this paired ground truth. *(L10–L11.)*

---

## Term project

Teams of **up to 3** (groups declared by **30 Aug**) pursue a problem with reasonable _engineering or science complexity_ (possibly a research question) 
in retrieval or recommendation. The project does not have to be an end-to-end system, nor serve a concrete use-case — 
what matters is a sharp, defensible question and a rigorous investigation of it. 

Teams may **extend the marketplace simulation harness** (e.g., the multi-turn dialog or cross-platform-competition scenarios the
harness scaffolds), bring their own dataset and application (product search, news/feed recsys,
scholarly search, ad ranking), or pursue a primarily empirical or methodological question.

- **Initial proposal (by 31 Aug — 30 days from start):** a first **literature survey** plus the
  candidate research question and why it is worth answering, dataset/testbed, baseline,
  evaluation plan, success metrics, and (where applicable) a target SLA + cost envelope.
- **Problem finalization (by 30 Sep):** a **deeper literature survey**; the problem statement
  **revised and finalized** against it — with the gap in prior work identified precisely — plus a
  working pipeline or experimental setup where feasible.
- **Final (due 20 Nov):** the completed investigation — implementations of the competing
  solutions, ablations, an offline or simulated A/B evaluation where applicable, and a discussion
  of scaling, cost, and failure modes; report + demo + presentation (presentations in the
  preceding weeks).

The project need not span every course thread. Instead, it must take on one hard, possibly
open-ended question — which may sit entirely within a single thread (systems, ML, or
evaluation) — and explore and evaluate several competing solutions to it in depth. The bar is the
difficulty of the question and the rigor of the comparison, not breadth of coverage.

### Project ideas

Scoped prototypes; evaluation plan agreed with the instructor first:

- **Conversational search** — multi-turn retrieval with query rewriting, clarifying questions, and
  session state; evaluate on a conversational IR benchmark (e.g., TREC CAsT/iKAT). 
- **Agentic memory** — a long-term memory layer for a search/recommendation agent (episodic store,
  retrieval, consolidation); measure the personalization lift over a memoryless agent. 
- **Scouting agents** — standing-query web monitors in the style of Yutori
  (<https://yutori.com/>): persistent user intents, change detection over recrawls, and ranked
  notifications; evaluate alert precision and timeliness. 
- **Generative retrieval (DSI-lite)** — train a small model mapping queries to docIDs on a modest
  corpus; compare against BM25/dense and study behaviour under index updates. 
- **RAG evaluation harness** — chunking + hybrid retrieval + rerank + generate, with RAGAS-style
  grounding and answer-quality evaluation. 

---
