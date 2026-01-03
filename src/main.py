import argparse
import os
from pathlib import Path
from src.graph.engine import GraphEngine
from src.data.generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(description="C-RAG: Cognitive Graph-RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 1. Ingest
    parser_ingest = subparsers.add_parser("ingest", help="Ingest Wikipedia dump")
    parser_ingest.add_argument("--path", type=str, required=False, help="Path to dump file")
    
    # 2. Partition
    parser_part = subparsers.add_parser("partition", help="Partition the graph")
    parser_part.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"], help="Target Dataset")
    parser_part.add_argument("--num_parts", type=int, default=500, help="Number of partitions")
    parser_part.add_argument("--method", type=str, default="leiden", choices=["leiden", "metis", "hybrid"], help="Partitioning Strategy")
    
    # ... (generate_data)


    parser_gen = subparsers.add_parser("generate_data", help="Generate synthetic training data")
    parser_gen.add_argument("--out", type=str, required=False, help="Output file (Optional)")
    parser_gen.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"], help="Target Dataset")
    parser_gen.add_argument("--method", type=str, default="leiden", choices=["leiden", "metis", "hybrid"], help="Partitioning Strategy")
    parser_gen.add_argument("--size", type=int, default=100000, help="Number of samples to generate")
    
    # 4. Train
    parser_train = subparsers.add_parser("train", help="Train alignment model")
    parser_train.add_argument("--epochs", type=int, default=10)
    parser_train.add_argument("--batch_size", type=int, default=32)
    parser_train.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "gcn", "sage", "gin"], help="Model Architecture")
    parser_train.add_argument("--loss_type", type=str, default="infonce", choices=["infonce", "bce"], help="Loss Function")

    parser_train.add_argument("--graph_path", type=str, default=None, help="Path to Partitioned Graph")
    parser_train.add_argument("--train_path", type=str, default=None, help="Path to Training Data")
    parser_train.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"], help="Target Dataset (if paths not provided)")
    parser_train.add_argument("--method", type=str, default="leiden", choices=["leiden", "metis", "hybrid"], help="Partition Method (if paths not provided)")
    parser_train.add_argument("--max_steps", type=int, default=None, help="Stop after N batches (Smoke Test)")
    
    # 5. Extract Edges
    parser_edges = subparsers.add_parser("extract_edges", help="Extract Inter-Partition Edges for GNN")
    parser_edges.add_argument("--graph_path", type=str, default=None, help="Input Graph")
    parser_edges.add_argument("--output_path", type=str, default=None, help="Output Edges")
    parser_edges.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"])
    parser_edges.add_argument("--method", type=str, default="leiden", choices=["leiden", "metis", "hybrid"])

    # 6. Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate trained models")
    parser_eval.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"], help="Target Dataset")
    parser_eval.add_argument("--method", type=str, default="all", choices=["leiden", "metis", "hybrid", "all"], help="Partition Method to evaluate")
    parser_eval.add_argument("--size", type=int, default=1000, help="Test Set Size (if creating new)")
    parser_eval.add_argument("--split", type=str, default="both", choices=["train", "test", "both"], help="Data Split to evaluate on")

    # 7. CoraFull Pipeline
    parser_cora = subparsers.add_parser("corafull", help="Run CoraFull Pipeline")
    parser_cora.add_argument("--step", type=str, default="all", choices=["ingest", "partition", "extract_edges", "generate", "train", "all"])
    parser_cora.add_argument("--method", type=str, default="leiden", choices=["leiden", "metis"], help="Partitioning Strategy")
    parser_cora.add_argument("--num_parts", type=int, default=100, help="Number of Partitions")

    # 7. System Verification
    parser_verify = subparsers.add_parser("verify_pipeline", help="Run internal pipeline verification (Dry Run)")
    parser_verify.add_argument("--method", type=str, default="leiden", choices=["leiden", "metis", "hybrid"])
    parser_verify.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "corafull"], help="Target Dataset")
    parser_verify.add_argument("--num_parts", type=int, default=10, help="Number of Partitions (Dry Run)")

    args = parser.parse_args()
    
    if args.command == "ingest":
        from src.data.ingest import download_dump, process_dump, RAW_DIR
        
        # Determine path
        if args.path:
            target_path = args.path
        else:
            # Auto-download if no path provided
            if not (RAW_DIR / "simplewiki.xml.bz2").exists():
                target_path = download_dump()
            else:
                target_path = RAW_DIR / "simplewiki.xml.bz2"
        
        process_dump(target_path)
        
    elif args.command == "partition":
        from src.graph.partitioning import GraphPartitioner
        # Output directory depends on method
        out_dir = f"data/{args.dataset}/graph/{args.method}"
        data_dir = f"data/{args.dataset}/processed"
        print(f"Partitioning Method: {args.method.upper()} -> {out_dir}")
        
        gp = GraphPartitioner(data_dir=data_dir, output_dir=out_dir)
        gp.run_pipeline(num_parts=args.num_parts, method=args.method)
        print("Partitioning complete.")
        
    elif args.command == "generate_data":
        ge = GraphEngine()
        
        # Dynamic Path Construction
        base_dir = f"data/{args.dataset}/graph/{args.method}"
        graph_path = f"{base_dir}/full_graph.pt"
        
        # Default output path if not specified
        if not args.out:
            args.out = f"{base_dir}/train.json"
            
        print(f"Generating Data for {args.dataset.upper()} ({args.method})...")
        print(f"  Graph: {graph_path}")
        print(f"  Out:   {args.out}")
        
        if not os.path.exists(graph_path):
             print(f"Error: {graph_path} not found. Run 'partition' first.")
             return
             
        ge.load(graph_path)
        
        # Ensure text map is loaded
        text_map_path = f"data/{args.dataset}/processed/nodes.jsonl"
        ge.load_text_map(text_map_path)
        
        gen = SyntheticDataGenerator(ge)
        gen.generate_pairwise_alignment_data(output_path=args.out, total_samples=args.size)
        print("Data generation complete.")

    elif args.command == "train":
        from src.train import train_alignment_model
        
        # Dynamic Path Resolution
        if not args.graph_path:
             args.graph_path = f"data/{args.dataset}/graph/{args.method}/full_graph.pt"
        if not args.train_path:
             args.train_path = f"data/{args.dataset}/graph/{args.method}/train.json"

        print(f"Training Alignment Model ({args.method})...")
        print(f"  Graph: {args.graph_path}")
        print(f"  Train: {args.train_path}")
        train_alignment_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_type=args.model_type,
            loss_type=args.loss_type,
            graph_path=args.graph_path,
            train_path=args.train_path,
            max_steps=args.max_steps
        )
        
    elif args.command == "extract_edges":
        from src.graph.extract_edges import extract_coarse_edges
        
        if not args.graph_path:
             args.graph_path = f"data/{args.dataset}/graph/{args.method}/full_graph.pt"
        if not args.output_path:
             args.output_path = f"data/{args.dataset}/graph/{args.method}/partition_edges.pt"
             
        extract_coarse_edges(graph_path=args.graph_path, output_path=args.output_path)

    elif args.command == "evaluate":
        import torch
        from src.evaluate import load_model, evaluate
        
        print(f"🚀 Starting Evaluation (Dataset: {args.dataset.upper()}, Method: {args.method.upper()})")
        
        data_root = Path(f"data/{args.dataset}")
        graph_root = data_root / "graph"
        
        if args.method == "all":
            methods = [d.name for d in graph_root.iterdir() if d.is_dir() and d.name != "embeddings"]
        else:
            methods = [args.method]
            
        results = []
        
        for method in methods:
            print(f"\n============================================================")
            print(f"🎯 Evaluating Method: {method.upper()}")
            print(f"============================================================")
            
            capsule_dir = graph_root / method
            model_dir = capsule_dir / "model"
            
            if not model_dir.exists():
                print(f"  ⚠️ No model directory for {method}")
                continue
                
            # 1. Prepare Data Paths
            paths_to_evaluate = []
            
            # Train Set
            if args.split in ["train", "both"]:
                train_path = capsule_dir / "train.json"
                if train_path.exists():
                    paths_to_evaluate.append(("TRAIN", train_path))
                else:
                    print(f"  ⚠️ Train set missing: {train_path}")
            
            # Test Set
            if args.split in ["test", "both"]:
                test_path = capsule_dir / "test.json"
                if not test_path.exists():
                    print(f"  generating Test Set ({args.size} samples)...")
                    full_graph_path = capsule_dir / "full_graph.pt"
                    nodes_path = data_root / "processed/nodes.jsonl"
                    
                    if full_graph_path.exists():
                        ge = GraphEngine()
                        ge.load(str(full_graph_path))
                        ge.load_text_map(str(nodes_path))
                        
                        # Optimization: Load coarse edges if available
                        edges_path = capsule_dir / "partition_edges.pt"
                        p_edges = None
                        if edges_path.exists():
                             p_edges = torch.load(edges_path, map_location="cpu", weights_only=False)
                             print(f"  Loaded Partition Edges (Fast Gen).")
                        
                        # Generate exact number of samples requested
                        print(f"  Generating {args.size} samples (50% Intra, 50% Inter)...")
                        
                        gen = SyntheticDataGenerator(ge)
                        gen.generate_pairwise_alignment_data(
                            output_path=str(test_path), 
                            total_samples=args.size,
                            partition_edge_index=p_edges
                        )
                    else:
                        print(f"  ❌ Cannot generate test set. Graph missing: {full_graph_path}")
                
                if test_path.exists():
                    paths_to_evaluate.append(("TEST", test_path))

            # 2. Evaluate Models
            models = list(model_dir.glob("*.pt"))
            if not models:
                 print("  ⚠️ No models found.")
                 continue
                 
            for model_file in models:
                try:
                    name = model_file.stem
                    parts = name.split("_")
                    if len(parts) >= 3:
                         m_type = parts[1]; l_type = parts[2]
                    else:
                         continue
                         
                    print(f"\n  ▶ Model: {name}")
                    model, centroids = load_model(str(model_file), m_type, l_type, str(capsule_dir / "full_graph.pt"))
                    
                    for split_name, data_path in paths_to_evaluate:
                        # print(f"    Evaluating on {split_name}...")
                        metrics = evaluate(model, str(data_path), l_type, centroids, str(capsule_dir / "full_graph.pt"))
                        
                        results.append({
                            "Method": method, "Model": m_type.upper(), "Loss": l_type.upper(),
                            "Split": split_name, "P@1": metrics["P@1"], "R@5": metrics["R@5"], "R@10": metrics["R@10"], "MRR": metrics["MRR"]
                        })
                        print(f"    [{split_name}] P@1: {metrics['P@1']:.2%} | R@5: {metrics['R@5']:.2%} | R@10: {metrics['R@10']:.2%} | MRR: {metrics['MRR']:.4f}")
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")

        # Final Table
        print(f"\n\n🏆 EVALUATION LEADERBOARD")
        print(f"{'Method':<10} | {'Model':<6} | {'Loss':<8} | {'Split':<6} | {'P@1':<8} | {'R@5':<8} | {'R@10':<8} | {'MRR':<8}")
        print("-" * 85)
        results.sort(key=lambda x: (x["Split"], x["P@1"]), reverse=True)
        for r in results:
            print(f"{r['Method']:<10} | {r['Model']:<6} | {r['Loss']:<8} | {r['Split']:<6} | {r['P@1']:.2%} | {r['R@5']:.2%} | {r['R@10']:.2%} | {r['MRR']:.4f}")

    elif args.command == "verify_pipeline":
        print(f"🚀 Verifying Data Pipeline (Dataset: {args.dataset}, Method: {args.method})...")
        from src.graph.partitioning import GraphPartitioner
        from src.graph.extract_edges import extract_coarse_edges
        
        # 1. Pipeline Paths
        data_root = f"data/{args.dataset}"
        out_dir_root = f"{data_root}/graph/{args.method}"
        full_graph_path = f"{out_dir_root}/full_graph.pt"
        edges_path = f"{out_dir_root}/partition_edges.pt"
        test_train_path = f"{out_dir_root}/train_dry_run.json"
        
        # 2. Partition
        print("\n[1/3] Testing Partitioning (Dry Run)...")
        # Ensure input data exists
        processed_dir = f"{data_root}/processed"
        if not os.path.exists(processed_dir):
            print(f"❌ Missing processed data in {processed_dir}. Run ingest/corafull pipeline first.")
            sys.exit(1)
            
        gp = GraphPartitioner(data_dir=processed_dir, output_dir=out_dir_root)
        gp.run_pipeline(num_parts=args.num_parts, method=args.method) 
        
        if not os.path.exists(full_graph_path):
            print(f"❌ Partitioning Failed. Missing {full_graph_path}")
            sys.exit(1)
            
        # 3. Extract Edges
        print("\n[2/3] Testing Edge Extraction...")
        extract_coarse_edges(graph_path=full_graph_path, output_path=edges_path)
        
        if not os.path.exists(edges_path):
            print(f"❌ Edge Extraction Failed. Missing {edges_path}")
            sys.exit(1)
            
        # 4. Generate Data
        print("\n[3/3] Testing Data Generation (Dry Run)...")
        ge = GraphEngine()
        ge.load(full_graph_path)
        ge.load_text_map(f"{data_root}/processed/nodes.jsonl")
        gen = SyntheticDataGenerator(ge)
        gen.generate_pairwise_alignment_data(output_path=test_train_path)
        
        if not os.path.exists(test_train_path):
             print(f"❌ Data Generation Failed. Missing {test_train_path}")
             sys.exit(1)
        
        # Cleanup
        os.remove(test_train_path)
        print("\n✅ Verification Successful! Pipeline is robust.")
        
    elif args.command == "corafull":
        from src.data.corafull_adapter import ingest_corafull
        from src.graph.partitioning import GraphPartitioner
        from src.graph.extract_edges import extract_coarse_edges
        from src.train import train_alignment_model
        
        print(f"🚀 Running CoraFull Pipeline (Method: {args.method.upper()}) (Step: {args.step})")
        
        # Dynamic Paths for Method
        base_dir = f"data/corafull/graph/{args.method}"
        
        full_graph_path = f"{base_dir}/full_graph.pt"
        edges_path = f"{base_dir}/partition_edges.pt"
        train_path = f"{base_dir}/train.json"
        
        if args.step in ["ingest", "all"]:
            # Check if exists
            if args.step == "all" and os.path.exists("data/corafull/processed/nodes.jsonl"):
                print("  ✓ Skipping Ingest (Found nodes.jsonl)")
            else:
                ingest_corafull()
            
        if args.step in ["partition", "all"]:
             if args.step == "all" and os.path.exists(full_graph_path):
                 print(f"  ✓ Skipping Partition (Found {full_graph_path})")
             else:
                 print(f"\nStep 2: Embedding & Partitioning ({args.method.upper()})...")
                 gp = GraphPartitioner(data_dir="data/corafull/processed", output_dir=base_dir)
                 gp.run_pipeline(num_parts=args.num_parts, method=args.method)
             
        if args.step in ["extract_edges", "all"]:
             if args.step == "all" and os.path.exists(edges_path):
                 print(f"  ✓ Skipping Edge Extraction (Found {edges_path})")
             else:
                 print(f"\nStep 3: Extracting Partition Structure...")
                 extract_coarse_edges(graph_path=full_graph_path, output_path=edges_path)
             
        if args.step in ["generate", "all"]:
             if args.step == "all" and os.path.exists(train_path):
                 print(f"  ✓ Skipping Data Gen (Found {train_path})")
             else:
                 print(f"\nStep 4: Generating Training Data (Queries)...")
                 ge = GraphEngine()
                 ge.load(full_graph_path)
                 ge.load_text_map("data/corafull/processed/nodes.jsonl")
                 gen = SyntheticDataGenerator(ge)
                 gen.generate_pairwise_alignment_data(output_path=train_path)
             
        if args.step in ["train", "all"]:
             # Always train or check model? Usually we want to retrain if requested, but let's be idempotent.
             model_path = f"data/corafull/model/alignment_mlp_infonce.pt" 
             # Wait, model path logic in train.py is tied to graph_path parent -> parent -> model
             # graph_path = data/cora/graph/metis/full_graph.pt
             # parent = data/cora/graph/metis
             # parent = data/cora/graph
             # model_dir = data/cora/model
             # output = data/cora/model/alignment_mlp_infonce.pt
             # This means METIS and LEIDEN models will overwrite each other if names are same!!
             # We need to change save name in train.py to include method? 
             # Or save in data/cora/graph/metis/model?
             # My logic in train.py was:
             # graph_dir = os.path.dirname(graph_path) # e.g. data/wiki/graph/leiden
             # data_root = os.path.dirname(graph_dir) # e.g. data/wiki/graph
             # model_dir = os.path.join(data_root, "model") # e.g. data/wiki/model
             # This is WRONG if I want separation.
             
             # CORRECTION:
             # If I want total separation, models should live in `data/wiki/graph/leiden/model/`?
             
             # Or I update train.py to respect the deeper structure.
             # data/wiki/graph/leiden -> parent is graph. parent is wiki.
             
             print("\nStep 5: Training MLP Baseline on Cora...")
             train_alignment_model(
                 graph_path=full_graph_path,
                 train_path=train_path, 
                 model_type="mlp", 
                 loss_type="infonce",
                 epochs=3
             )

if __name__ == "__main__":
    main()
