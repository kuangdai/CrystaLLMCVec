"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py
"""
import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from crystallm import CIFTokenizer


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 371
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    ##################################
    # for conditioning vector (cvec) #
    ##################################
    # main switch
    activate_cvec: bool = False
    # length of cvec
    n_input_cvec: int = 2
    # number of prefix virtual tokens
    n_prefix_cvec: int = 4
    # hidden size of CVecEncoder
    n_hidden_cvec: int = 128
    # dropout of CVecEncoder
    dropout_cvec: float = 0.1
    # layer-wise strategy
    # True: all blocks share the same key-value features
    # False: each block uses independent key-value features;
    #        the size of CVecEncoder will become `n_layer` times larger
    kv_shared_by_layers_cvec: bool = False


class CVecEncoder(torch.nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # size of output
        kv_size = config.n_prefix_cvec * config.n_embd * 2  # here 2 for key and value
        if not config.kv_shared_by_layers_cvec:
            kv_size *= config.n_layer
        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(config.n_input_cvec, config.n_hidden_cvec),
            torch.nn.Tanh(),
            torch.nn.Linear(config.n_hidden_cvec, kv_size),
            torch.nn.Tanh()
        )
        # dropout
        self.dropout = torch.nn.Dropout(config.dropout_cvec)

    def forward(self, x: torch.Tensor):
        # [batch_size, n_input_cvec] => [batch_size, kv_size]
        x = self.fnn(x)
        # [batch_size, kv_size] => [n_layer, batch_size, n_prefix_cvec, n_embd * 2]
        if self.config.kv_shared_by_layers_cvec:
            x = x.view(-1, self.config.n_prefix_cvec, self.config.n_embd * 2)
            # duplicate for layers
            x = x.unsqueeze(0).expand(self.config.n_layer, -1, -1, -1)
        else:
            x = x.view(-1, self.config.n_layer, self.config.n_prefix_cvec, self.config.n_embd * 2)
            x = x.transpose(0, 1)
        # dropout on last dimension
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool):
        """
        Initialize the LayerNorm module.

        :param ndim: dimensionality of the input tensor
        :param bias: whether to add a learnable bias to the output
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_prefix_cvec = config.n_prefix_cvec
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise RuntimeError("Flash attention (PyTorch >= 2.0) is required.")

    def forward(self, x: Tensor, kv_cvec: Tensor = None) -> Tensor:
        """
        Applies causal self-attention to the given tensor,
        with a mask to prevent attention to future positions.

        :param x: tensor of shape (batch size, sequence length, embedding dimension)
        :param kv_cvec: tensor of shape (batch size, n_prefix_cvec, embedding dimension * 2)
        :returns: result of applying the causal self-attention operation
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        attn_mask = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0).to(q.device)

        # conditioning vector
        if kv_cvec is not None:
            P = self.n_prefix_cvec
            k_cvec, v_cvec = kv_cvec.split(self.n_embd, dim=-1)
            k_cvec = k_cvec.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, P, hs)
            v_cvec = v_cvec.view(B, P, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, P, hs)
            k = torch.cat((k_cvec, k), dim=2)  # (B, nh, P + T, hs)
            v = torch.cat((v_cvec, v), dim=2)  # (B, nh, P + T, hs)
            cvec_mask = torch.ones(T, P, dtype=torch.bool).to(q.device)
            attn_mask = torch.cat((cvec_mask, attn_mask), dim=1)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


def gelu(x: Tensor) -> Tensor:
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function, as used in the Google BERT and
    OpenAI GPT models. See: "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415

    :param x: the tensor to which the GELU activation function will be applied
    :returns: the result tensor after applying the GELU activation function,
              possessing the same shape as the input tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, kv_cvec: Tensor = None) -> Tensor:
        """
        Forward pass for the Transformer Block module. A Block module includes causal self-attention,
        layer normalization, and MLP, and residual connections.

        :param x: input to the transformer block
        :param kv_cvec: key-value features of cvec
        :returns: output of the transformer block, with the same shape as in the input
        """
        x = x + self.attn(self.ln_1(x), kv_cvec)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # conditioning vector
        self.cvec_encoder = None
        if config.activate_cvec:
            self.cvec_encoder = CVecEncoder(config)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :returns: the number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cvec=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        if self.cvec_encoder is None:
            assert cvec is None, "Model cannot take in CVEC input."
            for block in self.transformer.h:
                x = block(x, None)
        else:
            assert cvec is not None, "Model requires CVEC input."
            kv_cvec = self.cvec_encoder(cvec)
            for i_block, block in enumerate(self.transformer.h):
                x = block(x, kv_cvec[i_block])

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: "transformer.wte.weight" and "lm_head.weight" are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, keyed by "transformer.wte.weight", below.
        # so let's manually remove "lm_head.weight" from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 cvec=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        tokenizer = CIFTokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        prev_id = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, cvec=cvec)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # a sequence of two newlines indicates the end of a CIF file
            if prev_id is not None and prev_id == newline_id and idx_next.item() == newline_id:
                break
            prev_id = idx_next.item()

        return idx
