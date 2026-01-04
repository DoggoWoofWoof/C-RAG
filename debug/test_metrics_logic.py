import torch

def calc_recall_at_k(k, topk_indices, multi_labels):
    batch_hits = 0
    batch_total_targets = 0
    current_topk = topk_indices[:, :k]
    
    print(f"--- Calculating R@{k} ---")
    print(f"TopK Slice:\n{current_topk}")
    
    for i in range(len(multi_labels)):
            m_labels = multi_labels[i]
            true_set = m_labels[m_labels != -1].tolist()
            pred_set = current_topk[i].tolist()
            
            hits = len(set(true_set) & set(pred_set))
            print(f"Sample {i}: True={true_set}, Pred={pred_set} -> Hits={hits}")
            
            batch_hits += hits
            batch_total_targets += len(true_set)
    return batch_hits, batch_total_targets

def test():
    # Batch of 2
    # Sample 0: Target=5. Preds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (Hit at pos 5, i.e., rank 6)
    # Sample 1: Target=0. Preds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (Hit at pos 0, i.e., rank 1)
    
    topk_indices = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ])
    
    multi_labels = torch.tensor([
        [5, -1, -1],
        [0, -1, -1]
    ])
    
    # Expected:
    # R@1:
    #   Sample 0: Pred=[0]. Target=5. Miss.
    #   Sample 1: Pred=[0]. Target=0. Hit.
    #   Total Hits=1. Total Targets=2. R@1 = 50%
    
    # R@10:
    #   Sample 0: Pred=[0..9]. Target=5. Hit.
    #   Sample 1: Pred=[0..9]. Target=0. Hit.
    #   Total Hits=2. Total Targets=2. R@10 = 100%
    
    print("Testing R@1...")
    h1, t1 = calc_recall_at_k(1, topk_indices, multi_labels)
    r1 = h1/t1
    print(f"R@1 Result: {r1:.2%}")
    
    print("\nTesting R@10...")
    h10, t10 = calc_recall_at_k(10, topk_indices, multi_labels)
    r10 = h10/t10
    print(f"R@10 Result: {r10:.2%}")
    
    if r1 == r10:
        print("\n❌ BUG CONFIRMED: R1 == R10")
    else:
        print("\n✅ LOGIC IS CORRECT: R1 != R10")

if __name__ == "__main__":
    test()
