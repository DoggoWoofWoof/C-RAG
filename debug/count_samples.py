
import json
import os

def check_counts():
    files = [
        "data/corafull/graph/metis/train.json",
        "data/corafull/graph/leiden/train.json",
        "data/corafull/graph/hybrid/train.json",
        "data/corafull/graph/leiden/test.json",
        "data/corafull/graph/hybrid/test.json"
    ]
    
    for f in files:
        if os.path.exists(f):
            try:
                print(f"Reading {f}...")
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    print(f"  ✅ Count: {len(data)} samples")
            except Exception as e:
                print(f"  ❌ Error reading {f}: {e}")
        else:
            print(f"  ⚠️ File not found: {f}")

if __name__ == "__main__":
    check_counts()
