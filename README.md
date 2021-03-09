# msa-transformer

## Installation
Create a fresh conda environment and run `pip install -r requirements.txt`.

Then you can download a sample dataset (138 Mb) using the `download_data.sh` script.

From here, you should also `pip install pytest` and then run `pytest tests`. If this works, you can now train ESM-1b (a single-sequence transformer masked language model)!

## Code organization
To modify any models, you're going to want to play around with `model.py` and `modules.py`. `model.py` contains the full models for ESM-1b and the MSA Transformer, as well as some additional training setup code. `modules.py` contains the individual layers for each model, such as multihead attention and axial attention.
