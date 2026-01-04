print("Start...")
try:
    import torch
    print("Torch ok")
    import torch_scatter
    print("Scatter ok")
except Exception as e:
    print(f"Error: {e}")
print("Done")
