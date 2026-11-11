from ascon.state import random_state

s = random_state()
print(s)

assert len(s.as_list()) == 5
for word in s.as_list():
    assert 0 <= word <= 0xFFFFFFFFFFFFFFFF

print("State test passed")