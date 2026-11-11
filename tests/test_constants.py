from ascon.state import AsconState
from ascon.constants import ROUND_CONSTANTS, add_round_constant

assert len(ROUND_CONSTANTS) == 12

s = AsconState(0, 0, 0, 0, 0)
out = add_round_constant(s.copy(), ROUND_CONSTANTS[0])

assert out.x2 == 0xF0
assert out.x0 == 0
assert out.x1 == 0
assert out.x3 == 0
assert out.x4 == 0

print("Round constants test passed")