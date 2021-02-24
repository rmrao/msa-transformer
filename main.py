from typing import Optional, Union
import sys
from pathlib import Path
import operator
import logging
import pytorch_lightning as pl
import torch
import esm
from evo.dataset import (
    RandomCropDataset,
    SubsampleMSADataset,
    MaskedTokenWrapperDataset,
    AutoBatchingDataset,
)
from evo.tokenization import Vocab
from model import MSATransformer
from dataset import MSADataset, TRRosettaContactDataset
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)


@dataclass
class DataConfig:
    ffindex_path: str = MISSING
    trrosetta_path: str = MISSING
    trrosetta_train_split: str = "train.txt"
    trrosetta_valid_split: str = "test.txt"
    num_workers: int = 3


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    lr_scheduler: str = "warmup_linear"
    warmup_steps: int = 16000
    max_seqlen: int = 1024
    max_tokens: int = 16384
    max_seqs_validation: int = 128
    valid_batch_size: int = 1
    accumulate_grad_batches: int = 1
    distributed_backend: str = "ddp"
    gpus: int = 1
    gradient_clip_val: float = 0
    max_epochs: int = 1000
    max_steps: int = 1000000
    num_nodes: int = 1
    precision: int = 32
    patience: int = 10
    mask_prob: float = 0.15
    random_token_prob: float = 0.1
    leave_unmasked_prob: float = 0.1


@dataclass
class MSATransformerModelConfig:
    embed_dim: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    embed_positions_msa: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1


@dataclass
class LoggingConfig:
    wandb_project: Optional[str] = None
    log_every_n_steps: int = 50
    progress_bar_refresh_rate: int = 1
    track_grad_norm: bool = False


@dataclass
class Config:
    data: DataConfig
    train: TrainConfig = TrainConfig()
    model: MSATransformerModelConfig = MSATransformerModelConfig()
    logging: LoggingConfig = LoggingConfig()
    fast_dev_run: bool = False
    resume_from_checkpoint: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="data", name="default", node=DataConfig)
cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="model", name="msa-transformer", node=TrainConfig)
cs.store(group="logging", name="default", node=LoggingConfig)


@hydra.main(config_name="config")
def train(cfg: Config) -> None:
    alphabet = esm.data.Alphabet.from_architecture("MSA Transformer")
    vocab = Vocab.from_esm_alphabet(alphabet)
    train_data = MSADataset(cfg.data.ffindex_path)
    train_data = RandomCropDataset(train_data, cfg.train.max_seqlen)
    train_data = SubsampleMSADataset(train_data, cfg.train.max_tokens)
    train_data = MaskedTokenWrapperDataset(
        train_data,
        cfg.train.mask_prob,
        cfg.train.random_token_prob,
        cfg.train.random_token_prob,
    )
    train_data = AutoBatchingDataset(train_data, cfg.train.max_tokens)
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        collate_fn=operator.itemgetter(0),
    )

    with open(Path(cfg.data.trrosetta_path) / cfg.data.trrosetta_train_split) as f:
        train_pdbs = f.read().splitlines()

    with open(Path(cfg.data.trrosetta_path) / cfg.data.trrosetta_valid_split) as f:
        valid_pdbs = f.read().splitlines()

    trrosetta_train_data = TRRosettaContactDataset(
        cfg.data.trrosetta_path,
        split_files=train_pdbs,
        max_seqs_per_msa=cfg.train.max_seqs_validation,
    )

    trrosetta_valid_data = TRRosettaContactDataset(
        cfg.data.trrosetta_path,
        split_files=valid_pdbs,
        max_seqs_per_msa=cfg.train.max_seqs_validation,
    )

    valid_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        trrosetta_valid_data,
        batch_size=cfg.train.valid_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=trrosetta_valid_data.collater,
    )

    model = MSATransformer(
        vocab_size=len(vocab),
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        pad_idx=vocab.pad_idx,
        mask_idx=vocab.mask_idx,
        prepend_bos=vocab.prepend_bos,
        append_eos=vocab.append_eos,
        embed_dim=cfg.model.embed_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        num_layers=cfg.model.num_layers,
        embed_positions_msa=cfg.model.embed_positions_msa,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        activation_dropout=cfg.model.activation_dropout,
        max_seqlen=cfg.train.max_seqlen,
        optimizer=cfg.train.optimizer,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler=cfg.train.lr_scheduler,
        warmup_steps=cfg.train.warmup_steps,
        max_steps=cfg.train.max_steps,
        contact_train_data=trrosetta_train_data,
    )

    if cfg.logging.wandb_project:
        try:
            # Requires wandb to be installed
            logger: Union[
                pl.loggers.LightningLoggerBase, bool
            ] = pl.loggers.WandbLogger(project=cfg.logging.wandb_project)
            logger.log_hyperparams(cfg.train)
            logger.log_hyperparams(cfg.model)
        except ImportError:
            raise ImportError(
                "Cannot use W&B logger w/o W&b install. Run `pip install wandb` first."
            )
    else:
        logger = True

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid/Long Range P@L",
        mode="max",
        save_top_k=5,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid/Long Range P@L",
        mode="max",
        patience=cfg.train.patience,
    )

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        distributed_backend=cfg.train.distributed_backend,
        fast_dev_run=cfg.fast_dev_run,
        gpus=cfg.train.gpus,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        max_epochs=cfg.train.max_epochs,
        max_steps=cfg.train.max_steps,
        num_nodes=cfg.train.num_nodes,
        precision=cfg.train.precision,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        track_grad_norm=cfg.logging.track_grad_norm,
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    train()
