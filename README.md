# msa-transformer
![](https://github.com/rmrao/msa-transformer/workflows/build/badge.svg)

## Installation
Create a fresh conda environment and install requirements:

```bash
# Install miniconda if you don't have it:
wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p ${HOME}/miniconda \
    && rm miniconda.sh \
    && ${HOME}/miniconda/bin/conda init \
    && source ${HOME}/.bashrc

# Create conda environment for this project
conda create -n proteins "python>=3.8,<3.9" pytorch cudatoolkit=10.2 ipython cython -c pytorch
conda activate proteins

# Install pypi packages
pip install -r requirements.txt

# (Optional) Install apex for fp16 training
git clone https://github.com/nvidia/apex
cd apex
pip install . --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
```

## Data
If you are on `highland`, protein sequences for training are located at `/data/proteins/uniref/uniref50/train.fasta`. Protein structure data is located at `/data/proteins/trrosetta`, based on the trRosetta paper from Yang et al. 2020.

### Sequence Data
Sequence data is 95% of Uniref50 sequences. This can be read into a pytorch dataset using the `FastaDataset` option of `evo` (see [here](https://github.com/rmrao/evo/blob/325914666a8ced5d379bb9538882329558148b5a/evo/dataset.py#L231-L281)). FastaDataset caches the offsets (in bytes) for each sequence in `<dataset-path>.idx.npy`. Then, it can use file seek + read to random access any sequence in the file. Here's an example:

```python
>>> from evo.dataset import FastaDataset

>>> data = FastaDataset("/data/proteins/uniref/uniref50/train.fasta", cache_indices=True)
>>> data[3000]
('UniRef100_B5YM02',
 'MSYYNKASSKPKPKQGNQVTRITMKPAPMDAPVSFG...
 ...)
```

## Training

The standard ESM model can be trained using the script in `train_esm.py`. Configuration is done using hydra, which is a structured configuration scheme. The following command will train ESM using 12 GPUs:

```bash
python train_esm.py \
    data.fasta_path=/data/proteins/uniref/uniref50/train.fasta data.trrosetta_path=/data/proteins/trrosetta \
    train.gpus=4 train.accumulate_grad_batches=16 train.distributed_backend=ddp train.precision=16
```

When debugging, you can add the flag `fast_dev_run=True`, which will run 1 training and 1 validation step and then stop.

### Training the Performer

To use performer attention instead of standard attention, you can add the flag `model.layer.attention_type=performer`. See the model section for more details.

## Model

Model code is located in two files: `model.py` and `modules.py`.

### model.py

This file contains the top-layer of models, along with configuration objects. As a simple example, you can search for the `attention_type` variable, and see how the configuration of standard vs. performer attention is implemented.

### modules.py

This file contains all other pytorch modules (intermediate layers). In particular, Performer attention is implemented on line 628. Any attention function should implement the following interface for the `forward` function:

```python
def forward(
    self,
    x: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    need_head_weights: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Self-attention update, including computing QKV matrices and performing an output projection.
    Args:
        x (torch.Tensor): Input of shape [L x B x D]
        key_padding_mask (Optional[torch.Tensor]): Boolean mask of shape [B x L]. True if position in x is padded.
        need_weights (bool): Whether to return the attention weights or to return None.
        attn_mask (Optional[torch.Tensor]): Dummy variable, kept for compatibility with fairseq.
        need_head_weights (bool): If False, take a mean over the attention heads when returning attention weights.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention update and attention weight tensors
    """
```

When starting an implementation, don't worry *too* much about the flags other than the input `x`. Just ensure you return the attention update and attention weight tensors. Then, go back and add in the fix for padded positions. Finally, consider whether you can use the `need_weights` flags to optimize anything (if you don't _need_ to return attention weights, maybe you don't need to compute them explicitly - especially true for linear attention models.)
