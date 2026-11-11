from ascon.state import AsconState
from ascon.diffusion import diffusion_layer

s = AsconState(
    0x0123456789ABCDEF,
    0x1111111111111111,
    0x2222222222222222,
    0x3333333333333333,
    0x4444444444444444,
)

out = diffusion_layer(s)
print(out)

assert isinstance(out, AsconState)
assert len(out.as_list()) == 5

assert out.as_list() != s.as_list()

print("Diffusion test passed")