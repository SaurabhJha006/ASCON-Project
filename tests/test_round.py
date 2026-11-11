from ascon.state import AsconState
from ascon.constants import ROUND_CONSTANTS
from ascon.round import ascon_round

s = AsconState(
    0x0123456789ABCDEF,
    0x1111111111111111,
    0x2222222222222222,
    0x3333333333333333,
    0x4444444444444444,
)

out = ascon_round(s, ROUND_CONSTANTS[0])

print(out)

assert isinstance(out, AsconState)
assert out.as_list() != s.as_list()

print("ASCON round test passed")