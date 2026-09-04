"""The same baseline run, with the kernel grafted in place of attention.

Diff its output against `reference/baseline.py`. Identical text means the
custom attention block changed nothing.

    python reference/grafted.py                   # graft every layer, compressed
    python reference/grafted.py 10                # graft only layer 10
    python reference/grafted.py --dense           # no quantisation

`--dense` is the one that answers "did my block change anything": with
compression off, any difference from the baseline is plumbing -- a RoPE
convention, a head layout, the W_o fold -- and not the cache format.
"""

import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from kernel.adapters.torch_llama import graft

MODEL = "models/Llama-3.2-1B"
PROMPT = "The history of computing hardware spans centuries, from mechanical"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()

args = [a for a in sys.argv[1:] if a != "--dense"]
dense = "--dense" in sys.argv[1:]
layers = [int(a) for a in args] or list(range(model.config.num_hidden_layers))
graft(model, layers=layers, compressed=not dense)
print(f"grafted layers {layers} ({'dense' if dense else 'compressed'})", file=sys.stderr)

inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
