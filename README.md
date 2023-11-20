[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# ALR Transformer
ALR Transformer that replaces the original transformer implementation of an joint encoder + decoder block with a feedforward/alr block with a decoder block


## Install
`pip install alr-transformer`


## Usage
```python
import torch
from alr_transformer import ALRTransformer

x = torch.randint(0, 100000, (1, 2048))

model = ALRTransformer(
    dim = 512,
    depth = 6,
    num_tokens = 100000,
    dim_head = 64,
    heads = 8,
    ff_mult = 4
)

out = model(x)
print(out)
print(out.shape)

```

## Train
- First git clone the repo then download and then run the following
```
python3 train.py
```



## Citation
```bibtex
@misc{bozic2023rethinking,
    title={Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an Alternative to Attention Layers in Transformers}, 
    author={Vukasin Bozic and Danilo Dordervic and Daniele Coppola and Joseph Thommes},
    year={2023},
    eprint={2311.10642},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

```