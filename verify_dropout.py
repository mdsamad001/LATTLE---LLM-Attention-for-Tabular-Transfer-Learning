import torch.nn as nn, warnings, pickle
warnings.filterwarnings("ignore")
from transfer_learning_module import lattle
_,tr,va,te,cc,nc,bc = pickle.load(open("cached_datasets/Diabetes_seed0.pkl","rb"))
for p in (0.0, 0.3):
    m = lattle.build_classifier(cc,nc,bc,num_class=2,p=1,num_layer=5,ffn_dim=2048,
                                num_attention_head=8,hidden_dim=768,hidden_dropout_prob=p)
    drops = sorted({mod.p for mod in m.modules() if isinstance(mod, nn.Dropout)})
    attn  = sorted({mod.dropout for mod in m.modules() if isinstance(mod, nn.MultiheadAttention)})
    print(f"requested {p}: nn.Dropout p values={drops}  MHA dropout={attn}")
