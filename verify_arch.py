import torch, warnings, pickle
warnings.filterwarnings("ignore")
from transfer_learning_module import lattle

_,tr,va,te,cc,nc,bc = pickle.load(open("cached_datasets/ilpd_seed0.pkl","rb"))
m = lattle.build_classifier(cc,nc,bc,num_class=2,p=1,
                            num_layer=5, ffn_dim=2048, num_attention_head=8, hidden_dim=768)

layers = m.gftt_attention_layers()
print(f"gFTT attention layers: {len(layers)}  (paper: 5)")
print(f"FFN dim (linear1.out_features): {layers[0].linear1.out_features}  (paper: 2048)")
print(f"attn embed_dim: {layers[0].self_attn.embed_dim}, heads: {layers[0].self_attn.num_heads}")

before = [l.self_attn.in_proj_weight.detach().clone() for l in layers]
K = torch.randn(768,768); V = torch.randn(768,768)
m.apply_ca(K, V)

d = 768
after = [l.self_attn.in_proj_weight.detach() for l in layers]
for i,(b,a) in enumerate(zip(before,after)):
    k_match = torch.allclose(a[d:2*d,:].cpu(), K)
    v_match = torch.allclose(a[2*d:,:].cpu(), V)
    changed = not torch.equal(b,a)
    print(f"  l{i}: changed={changed!s:5s} K==transplant={k_match!s:5s} V==transplant={v_match!s:5s}")

print(f"frozen params registered: {len(m.frozen_kv_params)} (expect 1)")

# gradient hook: K/V rows must receive zero gradient
l0 = layers[0].self_attn.in_proj_weight
loss = l0.sum(); loss.backward()
g = l0.grad
print(f"grad norms -> Q {g[:d].norm():.4f} (should be >0), "
      f"K {g[d:2*d].norm():.4f}, V {g[2*d:].norm():.4f} (both should be 0)")
