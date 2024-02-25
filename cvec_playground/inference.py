import sys

sys.path.append(".")
import os
import argparse

from contextlib import nullcontext
import torch

from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
)
import numpy as np


# adapted from generate_cifs.py
def generate(model_dir, seed, device, dtype, num_gens, max_new_tokens, temperature, top_k, chunk_of_prompts):
    # init torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init tokenizer
    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    print(f"initializing model from {model_dir} on {device}...")
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    # model = torch.compile(model)  # requires PyTorch 2.0

    generated = []
    with torch.no_grad():
        with ctx:
            for id, prompt, cvec in chunk_of_prompts:
                start_ids = encode(tokenizer.tokenize_cif(prompt))
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                cvec_tensor = None
                if cvec is not None:
                    cvec_tensor = torch.tensor(cvec, dtype=ptdtype, device=device)[None, ...]
                gens = []
                for _ in range(num_gens):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                       cvec=cvec_tensor)
                    output = decode(y[0].tolist())
                    gens.append(output)
                generated.append((id, gens))
    return generated


parser = argparse.ArgumentParser(prog='inference')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='location of checkpoint')
parser.add_argument('-i', '--id', type=str, required=True,
                    help='id of input')
parser.add_argument('-t', '--text', type=str, required=True,
                    help='text of input')
parser.add_argument('-c', '--cvec', type=float, nargs='+',
                    help='cvec of input')
args = parser.parse_args()

cvec = None
if args.cvec is not None:
    cvec = np.array(args.cvec)
chunk_of_prompts = [(args.id, args.text, cvec)]
out = generate(args.model, 0, 'cuda', 'float16', 2,
               100, 0.1, 4, chunk_of_prompts)
print(out)