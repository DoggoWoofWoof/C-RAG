# C-RAG: Cognitive Graph Retrieval-Augmented Generation

**C-RAG** is a scalable system designed to perform Retrieval-Augmented Generation (RAG) on large-scale Knowledge Graphs (100k+ nodes). It solves the scalability problem by partitioning the graph into semantic clusters and using an **Agentic Walker** to navigate only the relevant subgraphs at runtime.

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/your-repo/crag.git
cd crag

# Install dependencies (Torch, PyG, Transformers)
pip install -r requirements.txt
```

### 2. The "One-Click" Pipeline (CoraFull)
To see the system in action on a smaller dataset (CoraFull), run the full pipeline using the **Winner Strategy (Metis)**:
```bash
# Runs: Ingest -> Partition (Metis) -> Edge Extraction -> Data Gen -> Train MLP -> Evaluate
python -m src.main corafull --step all --method metis --num_parts 100
```

---

## 🛠️ Step-by-Step Guide

### Phase 1: Data Ingestion
Download and process the raw Wikipedia dump into `nodes.jsonl` and `edges.jsonl`.
```bash
# Valid datasets: 'wiki', 'corafull'
python -m src.main ingest --path data/dump.xml
```

### Phase 2: Graph Partitioning
We support three strategies. **Metis** is recommended for best alignment performance.

**Option A: Metis (Recommended)**
Strictly balanced sizes. Preserves global structure (Min-Cut).
```bash
python -m src.main partition --dataset corafull --method metis --num_parts 100
```

**Option B: Hybrid (Semantic)**
Recursive splitting + bin packing. Good for interpretability but lower retrieval recall.
```bash
python -m src.main partition --dataset corafull --method hybrid --num_parts 100
```

**Option C: Leiden (Native)**
Pure modularity. Creates Giant Components unsuited for RAG.
```bash
python -m src.main partition --dataset corafull --method leiden --num_parts 100
```

### Phase 3: Synthetic Data Generation
Generate "Question/Answer" pairs aligned to the graph structure.
```bash
# --size: Number of samples (default 100k)
# --method: Must match the partition method you used
python -m src.main generate_data --dataset corafull --method metis --size 100000
```

### Phase 4: Model Training
Train the Alignment Model (Bi-Encoder) to map queries to graph partitions.

**Train Single Model (Winner Config)**
```bash
# --model_type: mlp (Best), gcn, sage, gin
# --loss_type: infonce (Best), bce (Failed)
python -m src.main train --dataset corafull --method metis --model_type mlp --loss_type infonce --epochs 5
```

**Run Experiment Matrix (All Models)**
Use the helper script to train all 8 variations (MLP/GCN/SAGE/GIN x InfoNCE/BCE):
```bash
python run_experiments.py --dataset corafull --method metis
```

### Phase 5: Evaluation
Compare trained models using P@1, R@5, R@10, and MRR metrics.
```bash
# Scans all trained models in data/corafull/graph/*/model/
python -m src.main evaluate --dataset corafull --method all
```

---

## 🔧 Advanced: Manual Workflow / Reproduction
Use these commands to manually reproduce specific parts of the pipeline (e.g., resizing datasets or fixing specific methods).

### 1. Manual Edge Extraction
**Crucial for GNNs**. If you partition manually, you must extract edges explicitly so GNNs can see connections between partitions.
```bash
python -m src.main extract_edges --dataset corafull --method metis --output_path data/corafull/graph/metis/partition_edges.pt
```

### 2. Manual Data Generation (Custom Sizes)
Standardize training to 5k and testing to 10k samples.
```bash
# Generate 5,000 Train Samples
python -m src.main generate_data --dataset corafull --method metis --size 5000 --out data/corafull/graph/metis/train.json

# Generate 10,000 Test Samples
python -m src.main generate_data --dataset corafull --method metis --size 10000 --out data/corafull/graph/metis/test.json
```

### 3. Manual Training (Single Model)
Train a specific configuration (e.g., Debugging GCN+BCE).
```bash
python -m src.main train --dataset corafull --method metis --model_type gcn --loss_type bce --epochs 5
```

---

## 📂 Project Structure & Scripts

### Core CLI (`src/main.py`)
The central entry point.
*   `ingest`: Process raw XML.
*   `partition`: Run Leiden/Metis/Hybrid algorithms.
*   `generate`: Create synthetic training data.
*   `train`: Train alignment models.
*   `evaluate`: Benchmark performance.
*   `corafull`: Shortcut for the dataset pipeline.
*   `verify_pipeline`: Dry-run to check code health.

### Helper Scripts
*   `run_experiments.py`: Orchestrates the 8-model training matrix.
*   `debug/check_hybrid.py`: Validates partition stats (Giant/Island counts).
*   `debug/check_connectivity.py`: Analyzes graph topology (GCC size).
*   `run_smoke_tests.py`: Quick 1-epoch test to verify code changes don't break training.

### Key Directories
*   `src/graph/partitioning.py`: Implementation of Recursive Leiden + Semantic Merge.
*   `src/data/generator.py`: Synthetic query generation logic.
*   `src/model/`: PyTorch modules (MLP, GCN, SAGE, GIN).

---

## 📚 Documentation Index (Where to find explanations)
To keep files clean, we organized details as follows:

1.  **Benchmarks & Logs**: See `benchmarks.md`.
    *   Contains: Smoke Test Logs, Training Dynamics (Loss interpretation), and Model Comparisons.
2.  **Hybrid Partitioning & Binning**: See `ARCHITECTURE.md` (Section 6) or `IMPLEMENTATION.md` (Section 2b).
    *   explains: Recursive Split, Semantic Merge, and Orphan Bin Packing.
3.  **Runtime Agent (GraphRAG)**: See `ARCHITECTURE.md` (Section 7).
    *   Explains: The "Think-Stitch-Walk" algorithm and why it's different from Vector Search.

---

## 📂 Comprehensive File Reference (Total Catalog)

### Root Directory
*   `run_experiments.py`: **Experiment Orchestrator**. Runs the full 8-model matrix (MLP/GNN x InfoNCE/BCE).
*   `run_smoke_tests.py`: **Wiki Smoke Test**. fast 1-epoch check on Wiki data.
*   `run_smoke_tests_corafull.py`: **CoraFull Smoke Test**. Fast 1-epoch check on CoraFull data.
*   `src/main.py`: **CLI Entry Point**. The central command router.

### `src/` (Source Code)
#### Data & Graph
*   `src/data/ingest.py`: **XML Parser**. Converts Wiki XML -> JSONL.
*   `src/data/generator.py`: **Query Generator**. Creates synthetic `(Query, Path)` training pairs via random walks.
*   `src/data/corafull_adapter.py`: **Benchmark Adapter**. Loads standard CoraFull dataset.
*   `src/graph/engine.py`: **Graph DB**. Handles IO for the large `torch_geometric` Data object.
*   `src/graph/partitioning.py`: **The Partition Engine**. Implements `Leiden`, `Metis`, and `Hybrid` (Recursive + Semantic Merge) logic.
*   `src/graph/extract_edges.py`: **Edge Mapper**. Finds edges *between* partitions for GNNs.

#### Models (`src/model/`)
*   `alignment_mlp.py`: **Base Class**. MLP Bi-Encoder + Loss Logic (InfoNCE/BCE).
*   `alignment_gcn.py`: **GCN**. Adds GraphConv layers.
*   `alignment_sage.py`: **GraphSAGE**. Adds SAGEConv layers.
*   `alignment_gin.py`: **GIN**. Adds GINConv layers.

#### Core Logic
*   `src/train.py`: **Training Loop**. Specialized loop for Partition Alignment (Text -> Centroids).
*   `src/evaluate.py`: **Evaluator**. Computes P@1, R@5, R@10, MRR.

### `debug/` (Analysis & Dev Tools)
*   **Analysis**:
    *   `analyze_partitions.py`: **Key Script**. Calculates Stratification (Size/Coherence) and Lift metrics. Used for `BENCHMARKS.md`.
    *   `check_connectivity.py`: Validates Giant Connected Component (GCC) size.
    *   `check_hybrid.py`: Verifies Hybrid constraints (No Giants, No Islands).
    *   `check_partition_zero.py`: Deep dive into "The Giant" (Partition 0) on Wiki.
*   **Inspection**:
    *   `inspect_node_content.py`: Prints raw text of nodes in a specific partition.
    *   `inspect_data.py`: Quick look at `nodes.jsonl`.
    *   `inspect_graph.py`: Quick look at `.pt` graph stats.
    *   `debug_tensors.py`: Checks tensor shapes for mismatches.
*   **Verification**:
    *   `count_samples.py` / `count_metis_samples.py`: Verifies line counts in generated JSONs.
    *   `test_metrics_logic.py`: Unit test for P@1 / MRR math.
    *   `test_metis.py`: Minimal reproduction of Metis clustering.
    *   `check_scatter.py`: Tests `torch_scatter` installation.

### `legacy/` (Archive)
*   `generate_samples.py` & `train_jigsaw_model.py`: Old prototypes (Jigsaw Approach). replaced by `src/`.

---

## 📜 Command Cheat Sheet (Analysis & Debugging)
*Commands we used to generate the findings in Benchmarks.md.*

### 1. Benchmark Analysis
```bash
# Calculate Partition Size Distribution, Coherence, and Random Lift
python debug/analyze_partitions.py
```

### 2. Topology Checks
```bash
# Check how connected the partitions are (GCC)
python debug/check_connectivity.py --dataset corafull --method metis

# Verify that Partition 0 contains Dates/Stubs (Wiki)
python debug/check_partition_zero.py
```

### 3. Data Verification
```bash
# Print first 5 nodes of Partition 10
python debug/inspect_node_content.py --dataset corafull --method metis --partition 10 --limit 5

# Check tensor shapes of the graph file
python debug/debug_tensors.py
```

### 4. Full Experiment Loop
```bash
# 1. Partition
python -m src.main partition --dataset corafull --method metis --num_parts 100

# 2. Extract Edges (Required for GCN)
python -m src.main extract_edges --dataset corafull --method metis

# 3. Generate Data (100k samples)
python -m src.main generate_data --dataset corafull --method metis --size 100000

# 4. Run All Models
python run_experiments.py --dataset corafull --method metis

# 5. Evaluate
python -m src.main evaluate --dataset corafull --method all
```



