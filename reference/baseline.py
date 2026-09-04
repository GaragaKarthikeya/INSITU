"""The stock Hugging Face inference snippet, unmodified.

Run it before grafting and after. Same prompt, greedy decoding, so the same
text either way -- if the continuation changes, the custom attention block
changed something.

    python reference/baseline.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/Llama-3.2-1B"
PROMPT = "The history of computing hardware spans centuries, from mechanical"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
