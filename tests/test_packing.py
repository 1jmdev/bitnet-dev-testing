import torch

from qwen_bitnet_qat.packing import pack_bits, pack_trits_base3, unpack_bits, unpack_trits_base3


def test_pack_bits_roundtrip():
    bits = torch.randint(0, 2, (101,), dtype=torch.uint8)
    packed, n = pack_bits(bits)
    out = unpack_bits(packed, n)
    assert torch.equal(bits, out)


def test_pack_trits_roundtrip():
    trits = torch.randint(0, 3, (103,), dtype=torch.uint8)
    packed, n = pack_trits_base3(trits)
    out = unpack_trits_base3(packed, n)
    assert torch.equal(trits, out)
