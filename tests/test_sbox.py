from ascon.state import AsconState
from ascon.sbox import sbox_layer

s = AsconState(0, 0, 0, 0, 0)
out = sbox_layer(s)

print(out)

assert isinstance(out, AsconState)
for w in out.as_list():
    assert 0 <= w <= 0xFFFFFFFFFFFFFFFF

print("S-box test passed")