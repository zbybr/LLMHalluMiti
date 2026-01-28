from pathlib import Path

path = "qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv"
out  = "qwen3_32b_mutation_outputs_qwen3_32b_dataset20251225_utf8_sig_responses_fixed_newlines.csv"

data = Path(path).read_bytes()

print("Before:", data.count(b"\x84"))

# 删除所有 0x80 字节
clean = data.replace(b"\x84", b"")

Path(out).write_bytes(clean)

print("After:", clean.count(b"\x84"))