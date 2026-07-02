"""Triangle multiplicative update + triangle attention (AF3 supplement, Alg. 12-15).

The most error-prone part of the trunk. Give the einsum indices real care and
back them with a hand-computed reference test.
"""
