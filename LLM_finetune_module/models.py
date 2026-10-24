import torch
import torch.nn as nn
from transformers import GPT2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transfer_learning_module.lattle.modeling_lattle_GPT import TransTabTransformerLayer

class DistilGPT2WithFTTHead(nn.Module):
    def __init__(self, base_model, hidden_dim=768, ftt_dim=128, num_class=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.base = base_model.transformer  # Only use the transformer part

        # Projection for CLS token
        self.proj = nn.Identity() if hidden_dim == ftt_dim else nn.Linear(hidden_dim, ftt_dim)

        # Extract Q/K/V weights from final GPT2 attention layer.
        # Registered as buffers so they live in the state dict and follow .to(device).
        gpt_attn_w = self.base.h[-1].attn.c_attn.weight.data  # shape: [hidden_dim, 3 * hidden_dim]
        d = hidden_dim
        self.register_buffer('q_vec', gpt_attn_w[:, :d].mean(dim=0).clone())  # shape: [hidden_dim]
        self.register_buffer('k_vec', gpt_attn_w[:, d:2*d].mean(dim=0).clone())
        self.register_buffer('v_vec', gpt_attn_w[:, 2*d:].mean(dim=0).clone())

        # Projection layers for q, k, v
        self.q_proj = nn.Linear(hidden_dim, ftt_dim, bias=False) if hidden_dim != ftt_dim else nn.Identity()
        self.k_proj = nn.Linear(hidden_dim, ftt_dim, bias=False) if hidden_dim != ftt_dim else nn.Identity()
        self.v_proj = nn.Linear(hidden_dim, ftt_dim, bias=False) if hidden_dim != ftt_dim else nn.Identity()

        # FTT layers
        self.ftt_layers = nn.ModuleList([
            TransTabTransformerLayer(
                d_model=ftt_dim,
                nhead=num_heads,
                dim_feedforward=ftt_dim * 4,
                dropout=dropout,
                activation='relu',
                norm_first=True,
                use_layer_norm=True
            ) for _ in range(5)
        ])

        self.classifier = nn.Linear(ftt_dim, num_class)

    def forward(self, input_ids, attention_mask, labels=None):
        # Base GPT forward
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # GPT-2 is causal: position 0 attends only to itself, so use the LAST
        # non-padded token as the sequence representation (as in
        # GPT2ForSequenceClassification), not position 0.
        last_idx = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        cls_embedding = outputs.last_hidden_state[batch_idx, last_idx]  # (batch, hidden_dim)

        # Project CLS embedding
        cls_proj = self.proj(cls_embedding).unsqueeze(1)  # (batch, 1, ftt_dim)

        q_vec = self.q_vec
        k_vec = self.k_vec
        v_vec = self.v_vec

        # Expand to match batch size
        q = self.q_proj(q_vec).unsqueeze(0).expand_as(cls_proj)
        k = self.k_proj(k_vec).unsqueeze(0).expand_as(cls_proj)
        v = self.v_proj(v_vec).unsqueeze(0).expand_as(cls_proj)

        # Pass through FTT layers (assuming they support q/k/v input)
        x = cls_proj
        for layer in self.ftt_layers:
            x = layer(x, q=q, k=k, v=v)
        x = x.squeeze(1)  # (batch, ftt_dim)

        # Classification head
        logits = self.classifier(x)

        # Loss computation
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
