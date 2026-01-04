import pymetis
import numpy as np

print("Testing Pymetis...")
try:
    # Tiny graph: 0-1, 1-2
    xadj = [0, 1, 2, 2]
    adjncy = [1, 0]
    n_cuts, membership = pymetis.part_graph(2, xadj=xadj, adjncy=adjncy)
    print(f"Success! Cuts: {n_cuts}, Membership: {membership}")
except Exception as e:
    print(f"Error: {e}")
