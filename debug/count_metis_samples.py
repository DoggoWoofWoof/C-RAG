import json
import os

target_file = "data/corafull/graph/metis/train.json"

def count_samples():
    print(f"Checking file: {target_file}")
    
    if not os.path.exists(target_file):
        print("❌ File not found.")
        return

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            count = len(data)
            print(f"✅ File is a valid JSON Array.")
            print(f"🔢 Total Samples: {count}")
            
            if count > 0:
                print("\nSample [0]:")
                print(json.dumps(data[0], indent=2))
        else:
            print(f"⚠️ Unexpected content type: {type(data)}")

    except Exception as e:
        print(f"❌ Error reading JSON: {e}")

if __name__ == "__main__":
    count_samples()
