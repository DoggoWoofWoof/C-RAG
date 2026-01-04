# C-RAG Technical Implementation Plan

## Goal
Build a scalable "Cognitive Graph-RAG" system for 100k nodes that balances global search (via partitioning) and local reasoning (via agentic traversal).

## User Review Required
> [!IMPORTANT]
> **Alignment Strategy**: We are choosing a CLIP-style contrastive approach to align Queries to Graph Partitions.
> **ColBERT Optimization**: We cannot fully cache ColBERT scores (as they depend on the query), but we will cache document token tensors to remove the encoding step during runtime.

## Proposed Changes

### 1. Data & Preprocessing
#### [NEW] `src/data/ingest.py`
- Loads Wikipedia dump.
- Filters for target domains (Sports, Politics, etc.).
- Outputs: `nodes.jsonl`, `edges.jsonl`.

#### [NEW] `src/graph/builder.py`
- Builds `NetworkX` graph object.
- Stores node text (First Paragraph) in `SQLite` or `LMDB` for fast random access.

### 2. Partitioning & Indexing (Offline)
#### [NEW] `src/graph/partitioning.py`
- **Embedding**: Use `intfloat/e5-base-v2` to embed all node texts.
- **Clustering**:
    - **Baseline**: `METIS` (via `pymetis`). Fixed `k`. Balanced sizes. Time: Fast.
    - **Experiment A**: `Leiden` (Community Detection).
        - *Behavior*: Adapts `k` naturally based on graph structure (modularity). Can be hierarchical to approximate `k`.
        - *Benefit*: High "purity". A partition is a true topic, reducing noise for the agent.
- **Representation**: Compute "Partition Centroids" (mean of node embeddings).
### 2b. The "Hybrid" Partitioning Algorithm (Implemented)
To address the "Giant Component vs. Island Dust" topology of real-world knowledge graphs (e.g., Wiki/CoraFull), we designed a two-phase refinement strategy:

#### Phase 1: Semantic Merging (Strict Capacity)
*   **Action**: Iteratively merge small "Islands" (<100 nodes) into their nearest semantic neighbor.
*   **Constraint**: STRICT capacity check. Only merge if `Size(Island) + Size(Neighbor) <= 200`.

#### Phase 2: Recursive Splitting (The Hammer)
*   **Target**: Any partition with size $> 200$ nodes.
*   **Action**: Recursively apply Leiden until all sub-components are $\le 200$ nodes.

#### Phase 3: Cleanup & Orphan Clustering (Semantic Refinement)
*   **Problem**: Splitting creates fragments. Bin-Packing them blindly (by ID) merges unrelated concepts (e.g., Physics + Movies).
*   **New Action**: **Semantic Greedy Packing**.
    *   Compute Centroids for all remaining orphan islands.
    *   Iteratively pick an orphan and find its **Nearest Cosine Neighbors**.
    *   Pack them into a new partition until `Size=200`.
*   **Result**: 100% of partitions are valid size (1-200). 0 Giants. **Consistent Semantic Purity** (even for orphans).
*   **Result**: 100% of partitions are valid size (1-200). 0 Giants.
*   **Connectivity Integrity**: **NO edges are added.** Connectivity is achieved by grouping isolated nodes into connected clusters.

> [!NOTE]
> **The "Partition 0" Phenomenon (Wiki)**:
> Analysis revealed Partition 0 contained ~68k nodes with 0.0 semantic similarity but high structural connectivity.
> These are **Dates**, **Stubs**, and **Generic Hubs**. They are central but semantically "Stopwords".
> **Strategy**: The Runtime Agent treats Partition 0 as a global exclusion list.

### 2c. Partitioning Methods Supported
1.  **Hybrid** (Recommended): The strategy above. Best for RAG.
2.  **Leiden**: Pure modularity optimization. Good quality but unbalanced sizes.
3.  **METIS**: Structural min-cut. Strictly balanced sizes but often breaks semantic topics.

### 3. Query Alignment (The "CLIP" Layer)
#### [NEW] `src/model/alignment_mlp.py` (Unified Base)
- **Architecture**: Bi-Encoder (Query <-> Partition).
- **Unified Loss**: Now handles both `InfoNCE` (Contrastive) and `BCE` (Binary Cross Entropy) via `loss_type` argument.
- **Inheritance**: `GCN`, `SAGE`, and `GIN` modules now inherit from this base class to support both loss types seamlessly.
- **Training**:
    - Models are saved within their respective partition capsule (e.g., `data/wiki/graph/leiden/model/`).
    - Experiments run independently on each capsule.

### [DONE] Edge Extraction (For GNNs)
- Implemented `src/graph/extract_edges.py`.
- Extracted `data/wiki/graph/leiden/partition_edges.pt`.
- Enabled GCN/SAGE/GIN models to use real structure.

### 3.5. Evaluation Framework
#### [NEW] `src/evaluate.py`
- Implements strict P@1, R@5, R@10 metrics.
- Loads models dynamically based on capsule structure.

### 3.4. Synthetic Data Generation [DONE]
#### [UPDATED] `src/data/generator.py`
- **Hybrid Strategy**: Generates both Intra-partition (local) and Inter-partition (multi-hop) queries (50/50 split).
- **Quality Fix**: Prioritizes `Title` over `Text` to prevent paragraph dumping. extracts `regarding Topic_X` for context.
- **Optimization**: Re-uses `partition_edge_index` for fast coarse graph traversal.
- **CLI**: Added `--size` argument to `generate_data` (default: 100k).

#### [FUTURE] Topic-Aware Multi-Hop Generation
- **Idea**: To improve multi-hop reasoning, explicitly sample "Bridge Topics" (e.g., "Physics" connects to "Mathematics") rather than random edge walks.
- **Implementation**:
    - Build a "Topic Graph" (coarser than partitions).
    - Sample queries like: "How does [Topic A Concept] relate to [Topic B Concept]?"
    - *User Suggestion*: Add explicit topic tags to prompt context.

### 3.5. Evaluation Framework (Ongoing)
#### [NEW] `src/evaluate.py`
- Implements strict P@1, R@5, R@10, MRR metrics.
- Uses `Float32` precision for stability.

#### [DONE] Unified CLI (`src/main.py`)
- New `evaluate` command:
    1. Scans all capsules (`leiden`, `metis`).
    2. Auto-generates held-out "Test Set" (1000 samples) if missing.
    3. Evaluates all trained models on both **Train** and **Test** splits.
    4. Compares performance across methods.

### 4. Runtime Agent (Dynamic Graph Stitching)
#### [NEW] `src/agent/walker.py`
The "Runtime Agent" does *not* walk the full 100k-node graph. It constructs a **Dynamic Subgraph** per query.

**1. Think (Global Navigation)**
*   **Input**: Query $Q$.
*   **Action**: Use Aligner (Leiden/METIS) to score all $N_{parts}$ partitions.
*   **Filter**: Select Top-K partitions (e.g., $K=5$) or Score $> \tau$ (e.g., 0.5).
*   **Stitch**:
    *   Load all nodes/edges from the selected partitions.
    *   Load pre-computed **Bridge Edges** that connect these specific partitions (e.g., $P_1 \leftrightarrow P_5$).
    *   **Result**: A small, focused `Subgraph` (~500-2000 nodes).

**2. Act (Parallel Local Search & Teleport)**
*   **Landing (The "Virtual Edge")**:
    *   When the Agent selects a Partition, it does *not* strictly enter via a bridge edge.
    *   **Action**: It retrieves the full partition (max 200 nodes).
    *   **Scan**: It computes a lightweight score (Dot Product or ColBERT) for *all* nodes in the partition against the query.
    *   **Teleport**: The "Walker" spawns at the **Top-3 Scoring Nodes** (Landing Pads).
    *   *Benefit*: This groups disconnected "Islands" into a single logical search space. If the answer is on an island, the scoring step will find it immediately.
*   **Walk (Expansion)**:
    *   From the Landing Pads, the Walker traverses physical edges to neighbors.
    *   Mechanism: Score neighbors -> Move to best -> Repeat (Depth 2-3).
    *   Maintain "Visited Set" to avoid loops.

**3. Observe (Stop & Answer)**
*   **Check**: If a node's text similarity to $Q$ is $> 0.9$, stop.
*   **Backtrack**: If stuck (scores drop), jump back to the Hub or try the next best path (Beam Search width=3).

### 5. RAG Generation
#### [NEW] `src/rag/generator.py`
- Lineurize the path: `Hub -> Node A -> Node B -> Answer Node`.
- Prompt LLM: "Reason through this path to answer: {Query}".

### 6. Why is this "GraphRAG" and not just Vector Search?
If we "Teleport" into partitions using Vector Search, aren't we just doing RAG?
**No.** Vector Search is only the **Entry Ticket**.

| Method | Mechanism | What it Finds | Limitation |
| :--- | :--- | :--- | :--- |
| **Vector Search** | Similarity($Q$, $Doc$) | "Apples" $\to$ "Apple Pie Recipe" | Misses context. Cannot find "Cider" if it doesn't mention "Apple". |
| **GraphRAG** | Entry $\to$ **Walk** | "Apples" $\to$ "Apple Pie" $\to$ **"Cinnamon"** | Finds **Dissimilar but Connected** concepts (Multi-Hop). |

**The Workflow**:
1.  **Vector Entry**: Teleport to the "Apple" Partition.
2.  **Graph Expansion**: Walk edges to find "Cider", "Orchards", and "Cinnamon" (nodes that might have low vector similarity to "Apple" but are structurally critical).
3.  **Result**: The LLM gets a *connected subgraph* of context, not just a bag of similar keywords.

### 7. Research Novelty & Significance
User Question: *"Is this just HNSW Bi-Encoder? Can I publish this?"*

#### Differentiation from HNSW (Vector Search)
| Feature | HNSW / Vector Store | C-RAG (Hybrid Partitioning) |
| :--- | :--- | :--- |
| **Unit of Retrieval** | **Single Node** (Paragraph). | **Subgraph** (Partition + Neighbors). |
| **Context Window** | Stuffs Top-K isolated chunks. | Provides a **Connected Topology** for reasoning. |
| **Entry Point** | The final answer (hopefully). | The **Starting Point** for a multi-hop walk. |
| **Scalability** | O(log N) - but ignores structure. | O(1) (Partitions are constant size 200). |

#### Differentiation from Microsoft GraphRAG (2024)
| Feature | MS GraphRAG (Summarization) | C-RAG (Agentic Walk) |
| :--- | :--- | :--- |
| **Core Idea** | **Map-Reduce**. "Summarize every community first." | **Agentic**. "Walk the raw graph at runtime." |
| **Mechanism** | Retrieval = Matching query to a *Static Text Summary*. | Retrieval = Matching query to a *Mathematical Centroid* $\to$ Walking Edges. |
| **Cost** | **Extremely High**. Requires LLM calls for 100k communities offline. | **Low**. Only requires Embedding (Math). No LLM offline. |
| **Reasoning** | **Static**. Can only answer what is in the summary. | **Dynamic**. Can traverse edges to find connections the summarizer missed. |

#### The "Hybrid" Contribution (Publication Worthy)
The GraphRAG field is split into:
1.  **Global Summaries** (e.g., Microsoft GraphRAG): "Summarize the whole graph". Expensive.
2.  **Local Retrieval** (e.g., Hippoco): "Given node, find 1-hop neighbors".

**C-RAG fills the gap (Mid-Level Reasoning):**
*   **Novelty 1: Hybrid Partitioning**. Existing methods (Leiden/Metis) fail on real-world power-law graphs (Wiki). You designed a system that balances **Semantic Coherence** (Leiden) with **Computational Loads** (Metis/Bin Packing).
*   **Novelty 2: The "Partition 0" Solution**. Identifying and handling "Structural Glue" (Dates/Stubs) as a separate class of nodes (Teleport Exclusion) is a significant system design contribution.

**Target Venues**:
*   **SIGIR** (Short Paper): Focus on the Retrieval Efficiency of Hybrid Partitioning.
*   **CIKM** (Applied Track): Focus on the System Architecture and Scalability.
### 8. Future Directions

#### 8.1. Local LLM / SLM Support
To make the Agent fully private and offline-capable:
1.  **Architecture Change**: Abstract the `LLMClient` class.
    *   Current: Hardcoded to `OpenAI` / `Gemini` via API.
    *   Target: Supports `Ollama`, `vLLM`, or `Llama.cpp` endpoints.
2.  **Model Choice**:
    *   **Phi-3 (Mini)**: For "Thinking" (Router/Filter decisions).
    *   **Llama-3 (8B)** or **Gemma-2**: For "Synthesis" (Final Answer Generation).
3.  **Strict JSON Mode**: The Agent relies on JSON for `Thought/Action`. Local SLMs often struggle with schema enforcement, so we will implement `Guidance` or `LMQL` constrained decoding.

#### 8.2. Knowledge Graph (Hypergraph) Adaptation
Currently, C-RAG assumes a Homogeneous Graph (A -> B).
For true Knowledge Graphs (Hypergraphs):
1.  **Schema Awareness**:
    *   Modify `src/graph/engine.py` to store `edge_type` alongside `edge_index`.
    *   Update GCN layers to use `RGCNConv` (Relational GCN) instead of `GCNConv`.
2.  **Hyperedge Representation**:
    *   If Node A, B, and C share a relation (e.g., "Co-Authors"), standard graphs require $N^2$ edges.
    *   **Solution**: Introduce "Virtual Nodes" for hyperedges.
    *   Transformation: `(A, B, C)` -> `A --(is_author)--> [Node: Paper X] <--(is_author)-- B`.
    *   This converts the Hypergraph into a Standard Bipartite Graph that our existing alignment models (Metis/MLP) can handle without kernel modification.

### 9. Real-World Benchmark (HotpotQA)
To prove the Agent works on "Real Reasoning" (not just synthetic walks), we will use **HotpotQA** (Distractor Setting) in Phase 5.

#### The Setup
HotpotQA provides `(Question, Context)`.
The `Context` consists of:
*   **Gold Paragraphs** (2 nodes, e.g., "Apple" and "Cider").
*   **Distractor Paragraphs** (8 nodes, e.g., "Orange", "Banana").

#### The "Graph-ification"
1.  **Ingest**: Treat the 10 paragraphs as a small, disconnected graph of 10 Nodes.
2.  **Edges**: We do *not* have explicit links.
    *   *Option A*: Fully Connected (Clique) - Let the Agent decide.
    *   *Option B (Novel)*: Use **Entity Linking** to draw edges if Node A mentions "Cider" and Node B is "Cider".
3.  **Task**:
    *   Start at a random node.
    *   Goal: "Walk" to the 2 Gold Nodes.
    *   Success: If the Agent retrieves the 2 Gold Nodes in its "Stitched Subgraph".

#### Why this matters?
Synthetic data (CoraFull) tests "Can I find the partition?".
HotpotQA tests "Can I reason that A connects to B?".
If C-RAG solves HotpotQA, it proves the **"Think-Stitch-Walk"** architecture generalizes to generic multi-hop QA.
