from ascon.state import random_state, AsconState
from ascon.permutation import ascon_p12
from ascon.utils import hamming_weight

MASK64 = 0xFFFFFFFFFFFFFFFF


def state_hamming_distance(a: AsconState, b: AsconState) -> int:
    dist = 0
    for x, y in zip(a.as_list(), b.as_list()):
        dist += hamming_weight((x ^ y) & MASK64)
    return dist


s1 = random_state()

s2 = s1.copy()
s2.x0 ^= 1

out1 = ascon_p12(s1)
out2 = ascon_p12(s2)

distance = state_hamming_distance(out1, out2)

print("Hamming distance:", distance)

assert distance > 100

print("Avalanche test passed")