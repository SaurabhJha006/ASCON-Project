from ascon.utils import rotr, rotl, hamming_weight

x = 0x0123456789ABCDEF

assert rotr(x, 8) == 0xEF0123456789ABCD
assert rotl(x, 8) == 0x23456789ABCDEF01
assert hamming_weight(0xFFFFFFFFFFFFFFFF) == 64
assert hamming_weight(0) == 0

print("Utility tests passed")