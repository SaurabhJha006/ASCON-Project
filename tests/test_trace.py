from ascon.state import random_state
from ascon.permutation import ascon_permutation_trace

s = random_state()

trace = ascon_permutation_trace(s, rounds=4)

print(f"Number of states: {len(trace)}")

for i, st in enumerate(trace, start=1):
    print(f"Round {i}")
    print(st)

assert len(trace) == 4

print("Trace test passed")