import torch
from alr_transformer import ALRTransformer

x = torch.randint(0, 100000, (1, 2048))

model = ALRTransformer(
    dim=512, depth=6, num_tokens=100000, dim_head=64, heads=8, ff_mult=4
)

out = model(x)
print(out)
print(out.shape)
