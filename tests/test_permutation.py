from ascon.state import random_state
from ascon.permutation import (
    ascon_p2,
    ascon_p3,
    ascon_p4,
    ascon_p5,
    ascon_p6,
    ascon_p8,
    ascon_p12,
)

s = random_state()
print("Input state:")
print(s)

outputs = [
    ascon_p2(s),
    ascon_p3(s),
    ascon_p4(s),
    ascon_p5(s),
    ascon_p6(s),
    ascon_p8(s),
    ascon_p12(s),
]

for i, out in enumerate(outputs, start=2):
    print(f"Output after {i if i <= 6 else (8 if i == 7 else 12)} rounds:")
    print(out)

assert outputs[0].as_list() != s.as_list()
assert outputs[-1].as_list() != s.as_list()
assert outputs[0].as_list() != outputs[-1].as_list()

print("Permutation tests passed")