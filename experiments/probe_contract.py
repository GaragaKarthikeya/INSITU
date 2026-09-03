"""What a real decoder layer hands its attention module. Run before grafting."""
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()

seen = {}
orig = model.model.layers[10].self_attn.forward
def spy(hidden_states, *a, **kw):
    seen["args"] = [type(x).__name__ for x in a]
    seen["kwargs"] = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
                      for k, v in kw.items()}
    seen["hidden"] = tuple(hidden_states.shape)
    out = orig(hidden_states, *a, **kw)
    seen["out"] = tuple(out[0].shape) if isinstance(out, tuple) else tuple(out.shape)
    seen["out_type"] = type(out).__name__
    return out
model.model.layers[10].self_attn.forward = spy

ids = tok("The history of computing hardware spans centuries.", return_tensors="pt")
with torch.no_grad():
    model(**ids)
for k, v in seen.items():
    print(f"{k:10} {v}")
