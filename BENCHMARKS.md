# C-RAG Benchmarks & Evaluation Log

**Last Updated**: 2026-01-04
**Hardware**: CPU (Local)
**Dataset**: CoraFull (Validation), Wikipedia (Structure Analysis)

---

## 1. Executive Summary & Leaderboard
The goal is to align a user query to a relevant graph partition (Cluster of ~200 nodes). This is the **"Entry Ticket"** step before the Agent starts walking the graph.

### The Leaderboard (Test Set - 10,000 Samples)
*Ranking Criteria: MRR (Mean Reciprocal Rank) & Recall@10*

| Partitioning | Model | Loss | P@1 | R@5 | R@10 | MRR | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Leiden** | **GCN** | InfoNCE | **17.93%** | **36.75%** | **50.41%** | **0.3202** | 🏆 **Robust**. Best at ranking the *general area* correctly (Top-5). |
| **Leiden** | **MLP** | InfoNCE | 15.06% | 34.01% | 51.44% | 0.3038 | **High Coverage**. Best Recall@10. "Bag of Words" works. |
| **Metis** | **MLP** | InfoNCE | **11.87%** | **28.53%** | 39.25% | **0.2273** | 🥈 **Balanced Winner**. Global Min-Cut beats Hybrid. |
| **Metis** | GCN | InfoNCE | 5.19% | 15.35% | 25.41% | 0.1327 | **Mixed**. Structure helps slightly more than Hybrid GCN but still lags MLP. |
| **Hybrid** | **MLP** | InfoNCE | 8.84% | 22.49% | 33.92% | 0.1808 | **Honest**. Lower scores due to "Blind Bin-Packing" of orphans. |
| **Hybrid** | GCN | InfoNCE | 4.24% | 12.37% | 22.14% | 0.1083 | **Oversmoothed**. Structure hurts alignment in finely fragmented graphs. |
| **Leiden** | GIN | InfoNCE | 5.41% | 8.28% | 16.57% | 0.1269 | **Weak**. |
| **Leiden** | SAGE | InfoNCE | 3.53% | 7.15% | 15.30% | 0.1076 | **Weak**. |
| **Metis** | GIN | InfoNCE | 4.13% | 9.14% | 18.01% | 0.1031 | **Weak**. |
| **Metis** | SAGE | InfoNCE | 2.99% | 7.32% | 13.86% | 0.0748 | **Weak**. |
| **Hybrid** | GIN | InfoNCE | 2.50% | 7.38% | 14.92% | 0.0729 | **Weak**. Struggles to capture signal. |
| **Hybrid** | SAGE | InfoNCE | 1.54% | 4.36% | 9.12% | 0.0501 | **Weak**. |
| **Metis** | MLP | BCE | 2.01% | 5.02% | 10.35% | 0.0608 | **Failed**. Class imbalance kills BCE training. |
| **Metis** | GCN | BCE | 1.19% | 3.10% | 7.42% | 0.0321 | **Failed**. |
| **Metis** | GIN | BCE | 0.53% | 3.50% | 9.55% | 0.0357 | **Failed**. |
| **Metis** | SAGE | BCE | 0.53% | 3.43% | 8.45% | 0.0318 | **Failed**. |
| **Hybrid** | MLP | BCE | 1.59% | 4.02% | 8.84% | 0.0496 | **Failed**. |
| Any | Any | BCE | < 2% | < 5% | < 10% | < 0.05 | **Failed**. Binary classification cannot handle 1-vs-100 class imbalance. |

> **Interpretation**:
> *   **Leiden scores are inflated**: It has typical "Giant Components" which act as massive gravity wells. Predicting the Giant gives you high accuracy but low utility.
> *   **Hybrid scores are realistic**: It forces small, equal-sized partitions (max 200 nodes). Random chance is 0.9%. Achieving **8.84% P@1 (10x Lift)** and **34% Recall@10** is excellent for a retrieval system.
> *   **Recommendation**: Use **Hybrid + MLP + InfoNCE** for honest partitioning, or **Metis + MLP + InfoNCE** for pure performance.

---

## 2. Partitioning & Topology Analysis

### Stratification Analysis (CoraFull & Wiki)
*How the different algorithms break down the graph.*

| Method | Parts | Size Range | Avg Size | Coherence | Lift (vs Rand) | Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Leiden** | **403** | **2 - 1,859** | 49.1 | **0.985** | **+0.076** | **High Fidelity**. Captures true semantic clusters. Giants remain a blocker. |
| **Metis** | **100** | **192 - 203** | 197.9 | 0.961 | +0.052 | **Forced Balance**. Breaks semantic boundaries for equal sizing. |
| **Hybrid** | **107** | **110 - 200** | 185.0 | 0.962 | +0.053 | **Balanced & Clean**. Matches Metis coherence. Semantic Packing (V2) showed negligible lift (+0.001). |
| **Wiki/Leiden**| **9102** | **1 - 68,700**| 41.9 | 0.996 | +0.296 | **Unusable Giants**. 98% of users in Partition 0. |
| **Wiki/Hybrid**| **1906** | **29 - 399** | 200.3 | 0.868 | +0.168 | **Semantically Diluted**. Previous run (0.869) confirms that Semantic Packing of orphans yields **zero net gain**. |
| **Wiki/Metis** | **2120** | **156 - 185** | **180.0** | 0.868 | +0.168 | **Perfect Balance**. Dynamically tuned (K=2120) to hit 180-node target. Coherence matches Hybrid exact. |

### Analysis: The Orphan Paradox (Effect of Semantic Packing)
We hypothesized that switching from "Blind Bin-Packing" to "Semantic Greedy Packing" for Hybrid Orphans would improve coherence.
**Result**: **Zero Gain** (0.869 vs 0.868).

**Why?**
1.  **Leiden First**: The Hybrid algorithm starts with `Recursive Leiden`. Leiden *inherently* extracts all highly coherent structures (Islands) first.
2.  **Leftovers are Trash**: By definition, the "Orphans" are nodes/clusters that Leiden *rejected* from every other community. They are structurally and semantically isolated.
3.  **No Good Neighbors**: When we try to "Semantically Pack" them, we are searching for neighbors in a pool of other isolated rejects. Their similarity to *anything* is low.
4.  **Conclusion**: Binning vs. Semantic Packing doesn't matter because the signal was already extracted by the primary Leiden pass.

### Comparative Analysis: Why Metis > Hybrid? (11.8% vs 8.8%)
Despite similar partition counts (100 vs 107) and sizes, Metis outperforms Hybrid.
1.  **Global Optimization**: Metis solves for the **Global Min-Cut**. It finds the mathematically optimal way to slice the graph with minimal broken edges. In a citation graph, edges are semantic links, so Metis preserves global topic structure better.
2.  **Hybrid's Weakness**: Our Hybrid approach uses **Greedy Bin-Packing** for orphans. If an island doesn't fit, it gets thrown into a "Bin" based on ID order. This dilutes the centroid of that bin with unrelated content, making it harder to retrieve.
3.  **Takeaway**: Global Optimization (Metis) > Greedy Heuristic (Hybrid) for alignment.

### The "Partition 0" Phenomenon (Wikipedia)
*Why standard Community Detection (Leiden) fails for RAG.*

We analyzed the structure of Wikipedia (Subset) using Leiden.
*   **Total Nodes**: ~70,000
*   **Result**: 
    *   **Partition 0 (The Giant)**: 68,700 nodes (97% of graph).
    *   **Other Partitions**: 8,991 tiny islands (1-5 nodes).

**Analysis of Partition 0 Nodes (Pure Noise ~0.00 Sim)**:

| Node A | Node B | Relation | Why are they connected? |
| :--- | :--- | :--- | :--- |
| "2020 in Film" | "List of Insects" | None | Both link to "2020" or "List". |
| "Category:Physics" | "Category:History" | None | Both link to "Category:Fields". |
| "Albert Einstein" | "Taylor Swift" | None | Connected via "21st Century" or "USA". |

**The Problem**: **Structural Glue**. Standard Graph Algorithms treat "High Degree" nodes (Dates, Categories, Lists) as core hubs. They attract *everything*.
**The Solution**: **Hybrid Partitioning** detects Giants (>200 nodes) and recursively shatters them, while bin-packing the islands.

### The Semantic Bottleneck (Text Analysis)
*Why is the baseline similarity so high? Why did BCE fail?*

We inspected the training data (`src/data/corafull_adapter.py`) and found that CoraFull does not provide raw text. Instead, we generate synthetic text:
> "Scientific research paper {i} regarding Topic_{label}."

1.  **High Similarity**: Since 90% of the sentence is identical boilerplate, **all nodes are semantically close** (Cosine Sim > 0.9).
2.  **Hard Mode**: The model must ignore the boilerplate and learn to "Zoom In" exclusively on the `Topic_{label}` token.
3.  **Why InfoNCE Works**: It is a **Contrastive Loss**. It forces the model to find the *one* feature (Topic ID) that separates the positive from the 100 negatives, even if they look 99% the same.
4.  **Why BCE Failed**: The "No" class is too easy to predict due to the high noise-to-signal ratio.

> **Implication**: If C-RAG can align this generic data, it will perform significantly better on real-world data (Wiki/Docs) where text is distinct.

---

## 3. Architectural Decisions & FAQ

### Why MLP > GCN for Alignment?
> *"GraphRAG needs edges! Why is the Graph Neural Network failing?"*

**Diagnosis**: The Alignment Task is **Global Matching**, not Local Smoothing.
1.  **MLP Input**: `[Query Embedding]` vs `[Partition Centroid]`.
    *   The Centroid is the *Average* of all nodes in the partition. It represents the "Topic".
    *   MLP matches Topic to Topic. **High Precision**.
2.  **GCN Input**: `[Partition Centroid]` + `[Neighbor Partitions]`.
    *   GCN *averages* the neighbors into the representation.
    *   If "Physics" is connected to "History" (via a date), GCN *mixes* them.
    *   The embedding becomes "Physics+History".
    *   The Query "Quantum Mechanics" no longer matches as cleanly.
    *   **Result**: **Oversmoothing**.

**Decision**: Use **MLP** for the "Entry Ticket" (Alignment). Use **Graph Walk** (Edges) *after* landing.

### Why InfoNCE > BCE?
> *"Why does Binary Cross Entropy fail (Loss ~0.3)?"*

**Diagnosis**: The **Class Imbalance Trap**.
*   **Scenario**: 1 Correct Partition, 100 Wrong Partitions.
*   **BCE Strategy**: "If I guess 'No' everytime, I am 99% accurate."
*   **Result**: The model optimizes for *Accuracy* (rejecting negatives) rather than *Recall* (finding the positive). Loss drops to 0.3 and stays there. Gradient vanishes.
*   **InfoNCE Strategy**: "I must rank the Positive *higher* than the Negatives."
*   **Result**: Even if the score is low, as long as `Pos > Neg`, the loss decreases. It forces the model to learn the *structure* of the space.

---

## 4. Final Conclusions & Recommendations

### 🏆 The Winner: Metis + MLP + InfoNCE
After comprehensive testing on CoraFull (synthetic text, high connectivity), **Metis** partitioning with **MLP** alignment and **InfoNCE** loss is the most balanced and scalable solution.

*   **P@1**: 11.87% (Best scalable result)
*   **Recall@10**: 39.25%
*   **Coherence**: >0.96 (Equivalent to Hybrid)

### ⚔️ MLP vs. GCN (The "Structure Paradox")
*   **MLP (Winner)**: Treats partitions as "Bag of Topic Words". This is robust. It aligns the *query centroid* to the *partition centroid* directly.
*   **GCN (Loser)**: Averaging neighbors dilutes the signal. If a "Physics" partition is linked to a "History" partition (via a date), GCN blends them. The query "Quantum" then fails to match the blended embedding.
*   **Recommendation**: **Disable GCN for Alignment**. Use graph traversal (Edges) *only* after proper entry.

### 🔮 Implications for Wikipedia (Phase 4)
Results from CoraFull directly inform our Wiki strategy:
1.  **Drop Leiden**: It creates Giant Components (98% of graph) that act as black holes.
2.  **Use Metis (K=2000)**: We proved on Wiki that Metis can force perfect 180-node chunks. This is mathematically optimal for RAG context windows.
3.  **Expect Higher Scores**: Wiki has *real* distinct text (unlike CoraFull's boilerplate). InfoNCE performance should jump from ~15% to >40% because the semantic separation between partitions will be much sharper.

---

## Appendix A: Full Training Logs (Hybrid - CoraFull)
*Detailed 5-Epoch Logs showing convergence behavior.*

```text
🎯 PARTITION METHOD: HYBRID

------------------------------------------------------------
▶ Experiment A (MLP + InfoNCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=4.67, P1=1.90%, R5=4.63%
Epoch 2/5: L=4.63, P1=3.14%, R5=6.50%
Epoch 3/5: L=4.50, P1=6.46%, R5=10.60%
Epoch 4/5: L=4.42, P1=7.72%, R5=12.75%
Epoch 5/5: L=4.35, P1=8.52%, R5=14.73%
Model saved to data\corafull\graph\hybrid\model\alignment_mlp_infonce.pt
✅ Success (Steady Learning)

------------------------------------------------------------
▶ Experiment B (MLP + BCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=0.36, P1=2.06%, R5=4.32%
Epoch 2/5: L=0.34, P1=2.34%, R5=4.52%
Epoch 3/5: L=0.34, P1=2.02%, R5=4.61%
Epoch 4/5: L=0.34, P1=1.74%, R5=4.27%
Epoch 5/5: L=0.34, P1=2.00%, R5=4.63% (Flatline)
Model saved to data\corafull\graph\hybrid\model\alignment_mlp_bce.pt

------------------------------------------------------------
▶ Experiment C (GCN + InfoNCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=4.67, P1=1.84%, R5=3.96%
Epoch 2/5: L=4.67, P1=2.48%, R5=5.84%
Epoch 3/5: L=4.55, P1=3.28%, R5=7.51%
Epoch 4/5: L=4.47, P1=3.92%, R5=8.52%
Epoch 5/5: L=4.43, P1=4.60%, R5=8.93% (Slower than MLP)
Model saved to data\corafull\graph\hybrid\model\alignment_gcn_infonce.pt

------------------------------------------------------------
▶ Experiment D (SAGE + InfoNCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=4.67, P1=2.12%
Epoch 2/5: L=4.67, P1=1.84%
Epoch 3/5: L=4.67, P1=2.00%
Epoch 4/5: L=4.67, P1=2.04%
Epoch 5/5: L=4.67, P1=2.28% (Stuck)
Model saved to data\corafull\graph\hybrid\model\alignment_sage_infonce.pt

------------------------------------------------------------
▶ Experiment E (GIN + InfoNCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=4.67, P1=1.98%
Epoch 2/5: L=4.67, P1=1.90%
Epoch 3/5: L=4.67, P1=1.54%
Epoch 4/5: L=4.67, P1=1.86%
Epoch 5/5: L=4.61, P1=2.56% (Very slow start)
Model saved to data\corafull\graph\hybrid\model\alignment_gin_infonce.pt

------------------------------------------------------------
▶ Experiment F (GCN + BCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=0.36, P1=1.54%, R5=3.89%
Epoch 2/5: L=0.34, P1=1.44%, R5=3.48%
Epoch 3/5: L=0.33, P1=1.32%, R5=3.59%
Epoch 4/5: L=0.33, P1=1.54%, R5=3.53%
Epoch 5/5: L=0.33, P1=1.32%, R5=3.42%
Model saved to data\corafull\graph\hybrid\model\alignment_gcn_bce.pt

------------------------------------------------------------
▶ Experiment G (SAGE + BCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=0.37, P1=1.86%, R5=4.42%
Epoch 2/5: L=0.34, P1=2.22%, R5=4.58%
Epoch 3/5: L=0.34, P1=1.86%, R5=3.98%
Epoch 4/5: L=0.33, P1=1.72%, R5=4.10%
Epoch 5/5: L=0.33, P1=1.58%, R5=4.06%
Model saved to data\corafull\graph\hybrid\model\alignment_sage_bce.pt

------------------------------------------------------------
▶ Experiment H (GIN + BCE) [hybrid]
------------------------------------------------------------
Epoch 1/5: L=0.36, P1=1.60%, R5=3.73%
Epoch 2/5: L=0.34, P1=1.60%, R5=3.46%
Epoch 3/5: L=0.34, P1=1.84%, R5=3.64%
Epoch 4/5: L=0.34, P1=1.38%, R5=3.53%
Epoch 5/5: L=0.33, P1=1.72%, R5=3.52%
Model saved to data\corafull\graph\hybrid\model\alignment_gin_bce.pt
```

## Appendix B: Full Training Logs (Leiden - CoraFull)
*Detailed 5-Epoch Logs showing convergence behavior.*

```text
🎯 PARTITION METHOD: LEIDEN

------------------------------------------------------------
▶ Experiment A (MLP + InfoNCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=5.80, P1=13.04%, R5=18.33%
Epoch 2/5: L=5.20, P1=21.74%, R5=25.40%
Epoch 3/5: L=5.01, P1=13.04%, R5=29.27%
Epoch 4/5: L=4.93, P1=21.74%, R5=29.41%
Epoch 5/5: L=4.89, P1=30.43%, R5=34.92% (Strongest Learning)
Model saved to data\corafull\graph\leiden\model\alignment_mlp_infonce.pt

------------------------------------------------------------
▶ Experiment B (MLP + BCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=0.35, P1=0.00%, R5=0.00%
Epoch 2/5: L=0.32, P1=0.00%, R5=1.75%
Epoch 3/5: L=0.32, P1=0.00%, R5=0.00%
Epoch 4/5: L=0.32, P1=0.00%, R5=0.00%
Epoch 5/5: L=0.32, P1=0.00%, R5=0.00%
Model saved to data\corafull\graph\leiden\model\alignment_mlp_bce.pt

------------------------------------------------------------
▶ Experiment C (GCN + InfoNCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=5.62, P1=13.04%
Epoch 2/5: L=5.12, P1=4.35%
Epoch 3/5: L=4.99, P1=17.39%
Epoch 4/5: L=4.93, P1=30.43%
Epoch 5/5: L=4.90, P1=4.35% (Oscillating but generally down)
Model saved to data\corafull\graph\leiden\model\alignment_gcn_infonce.pt

------------------------------------------------------------
▶ Experiment D (SAGE + InfoNCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=5.44, P1=0.00%, R5=6.38%
Epoch 2/5: L=5.08, P1=4.35%, R5=11.11%
Epoch 3/5: L=5.05, P1=8.70%, R5=23.19%
Epoch 4/5: L=5.05, P1=4.35%, R5=10.17%
Epoch 5/5: L=5.03, P1=4.35%, R5=13.73%
Model saved to data\corafull\graph\leiden\model\alignment_sage_infonce.pt

------------------------------------------------------------
▶ Experiment E (GIN + InfoNCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=5.20, P1=17.39%, R5=25.00%
Epoch 2/5: L=5.03, P1=13.04%, R5=22.00%
Epoch 3/5: L=5.02, P1=4.35%, R5=17.54%
Epoch 4/5: L=5.02, P1=4.35%, R5=13.64%
Epoch 5/5: L=5.02, P1=21.74%, R5=21.88%
Model saved to data\corafull\graph\leiden\model\alignment_gin_infonce.pt

------------------------------------------------------------
▶ Experiment F (GCN + BCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=0.35, P1=0.00%, R5=0.00%
Epoch 2/5: L=0.33, P1=0.00%, R5=0.00%
Epoch 3/5: L=0.32, P1=0.00%, R5=0.00%
Epoch 4/5: L=0.32, P1=0.00%, R5=0.00%
Epoch 5/5: L=0.32, P1=0.00%, R5=0.00%
Model saved to data\corafull\graph\leiden\model\alignment_gcn_bce.pt

------------------------------------------------------------
▶ Experiment G (SAGE + BCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=0.35, P1=0.00%, R5=3.45%
Epoch 2/5: L=0.32, P1=0.00%, R5=1.67%
Epoch 3/5: L=0.32, P1=0.00%, R5=0.00%
Epoch 4/5: L=0.32, P1=0.00%, R5=0.00%
Epoch 5/5: L=0.32, P1=0.00%, R5=0.00%
Model saved to data\corafull\graph\leiden\model\alignment_sage_bce.pt

------------------------------------------------------------
▶ Experiment H (GIN + BCE) [leiden]
------------------------------------------------------------
Epoch 1/5: L=0.35, P1=13.04%, R5=22.64%
Epoch 2/5: L=0.32, P1=17.39%, R5=15.22%
Epoch 3/5: L=0.32, P1=13.04%, R5=12.31%
Epoch 4/5: L=0.32, P1=4.35%, R5=9.62%
Epoch 5/5: L=0.32, P1=0.00%, R5=7.69%
Model saved to data\corafull\graph\leiden\model\alignment_gin_bce.pt
```

## Appendix C: Full Training Logs (Metis - CoraFull)
*Detailed 5-Epoch Logs showing convergence behavior.*

```text
🎯 PARTITION METHOD: METIS

------------------------------------------------------------
▶ Experiment A (MLP + InfoNCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=4.60, P1=2.80%, R5=5.40%
Epoch 2/5: L=4.52, P1=5.76%, R5=10.19%
Epoch 3/5: L=4.36, P1=9.84%, R5=15.09%
Epoch 4/5: L=4.28, P1=11.98%, R5=17.68%
Epoch 5/5: L=4.22, P1=13.62%, R5=19.58%
Model saved to data\corafull\graph\metis\model\alignment_mlp_infonce.pt

------------------------------------------------------------
▶ Experiment B (MLP + BCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=0.36, P1=2.82%, R5=5.22%
Epoch 2/5: L=0.34, P1=1.86%, R5=5.10%
Epoch 3/5: L=0.34, P1=2.12%, R5=4.76%
Epoch 4/5: L=0.34, P1=2.22%, R5=4.94%
Epoch 5/5: L=0.34, P1=2.50%, R5=5.25%
Model saved to data\corafull\graph\metis\model\alignment_mlp_bce.pt

------------------------------------------------------------
▶ Experiment C (GCN + InfoNCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=4.60, P1=2.12%, R5=5.68%
Epoch 2/5: L=4.52, P1=3.76%, R5=7.68%
Epoch 3/5: L=4.37, P1=4.28%, R5=9.61%
Epoch 4/5: L=4.32, P1=4.68%, R5=10.69%
Epoch 5/5: L=4.27, P1=5.14%, R5=11.42%
Model saved to data\corafull\graph\metis\model\alignment_gcn_infonce.pt

------------------------------------------------------------
▶ Experiment D (SAGE + InfoNCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=4.60, P1=2.22%, R5=5.39%
Epoch 2/5: L=4.60, P1=2.04%, R5=5.00%
Epoch 3/5: L=4.59, P1=2.66%, R5=5.58%
Epoch 4/5: L=4.59, P1=2.76%, R5=5.62%
Epoch 5/5: L=4.58, P1=2.90%, R5=6.08%
Model saved to data\corafull\graph\metis\model\alignment_sage_infonce.pt

------------------------------------------------------------
▶ Experiment E (GIN + InfoNCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=4.60, P1=2.20%, R5=5.56%
Epoch 2/5: L=4.59, P1=2.88%, R5=5.93%
Epoch 3/5: L=4.59, P1=2.44%, R5=6.34%
Epoch 4/5: L=4.58, P1=3.06%, R5=6.54%
Epoch 5/5: L=4.46, P1=3.64%, R5=7.79%
Model saved to data\corafull\graph\metis\model\alignment_gin_infonce.pt

------------------------------------------------------------
▶ Experiment F (GCN + BCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=0.36, P1=0.60%, R5=2.07%
Epoch 2/5: L=0.34, P1=0.70%, R5=1.88%
Epoch 3/5: L=0.34, P1=0.50%, R5=2.02%
Epoch 4/5: L=0.34, P1=0.60%, R5=1.89%
Epoch 5/5: L=0.34, P1=0.70%, R5=1.93%
Model saved to data\corafull\graph\metis\model\alignment_gcn_bce.pt

------------------------------------------------------------
▶ Experiment G (SAGE + BCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=0.36, P1=0.76%, R5=3.15%
Epoch 2/5: L=0.34, P1=0.66%, R5=2.64%
Epoch 3/5: L=0.34, P1=0.60%, R5=2.77%
Epoch 4/5: L=0.34, P1=0.54%, R5=2.93%
Epoch 5/5: L=0.34, P1=0.54%, R5=2.84%
Model saved to data\corafull\graph\metis\model\alignment_sage_bce.pt

------------------------------------------------------------
▶ Experiment H (GIN + BCE) [metis]
------------------------------------------------------------
Epoch 1/5: L=0.37, P1=0.52%, R5=2.01%
Epoch 2/5: L=0.34, P1=0.48%, R5=1.98%
Epoch 3/5: L=0.34, P1=0.48%, R5=1.90%
Epoch 4/5: L=0.34, P1=0.78%, R5=1.88%
Epoch 5/5: L=0.34, P1=0.58%, R5=1.96%
Model saved to data\corafull\graph\metis\model\alignment_gin_bce.pt
```
