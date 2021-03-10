import subprocess


def test_dev_run():
    result = subprocess.run(
        [
            "python",
            "train_esm.py",
            "fast_dev_run=True",
            "model.num_layers=1",
            "train.valid_batch_size=1",
            "model.embed_dim=256",
            "model.num_attention_heads=8",
            "train.gpus=0",
        ],
        capture_output=True,
    )
    try:
        result.check_returncode()
    except subprocess.CalledProcessError:
        raise RuntimeError(result.stderr.decode())
