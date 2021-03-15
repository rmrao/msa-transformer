from typing import Optional, List, Any
import sys
from pathlib import Path
import logging
import pytorch_lightning as pl
import torch
import esm
from evo.dataset import (
    RandomCropDataset,
    MaskedTokenWrapperDataset,
    EncodedFastaDataset,
    BatchBySequenceLength,
)
from evo.tokenization import Vocab
from model import ESM1b, TransformerConfig, OptimizerConfig
from dataset import TRRosettaContactDataset
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

current_directory = Path(__file__).parent.absolute()


@dataclass
class DataConfig:
    fasta_path: str = str(current_directory / "data/trrosetta.fasta")
    trrosetta_path: str = str(current_directory / "data")
    trrosetta_train_split: str = "valid_train.txt"
    trrosetta_valid_split: str = "valid_test.txt"
    num_workers: int = 3


@dataclass
class TrainConfig:
    max_tokens: int = 2 ** 17
    valid_batch_size: int = 2
    accumulate_grad_batches: int = 1
    distributed_backend: Optional[str] = None
    gpus: int = 1
    gradient_clip_val: float = 1.0
    max_epochs: int = 1000
    num_nodes: int = 1
    precision: int = 32
    patience: int = 10
    mask_prob: float = 0.15
    random_token_prob: float = 0.1
    leave_unmasked_prob: float = 0.1


@dataclass
class LoggingConfig:
    wandb_project: Optional[str] = None
    log_every_n_steps: int = 50
    progress_bar_refresh_rate: int = 1
    track_grad_norm: bool = False


defaults = [{"model": "esm1b-small"}]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: TransformerConfig = TransformerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    logging: LoggingConfig = LoggingConfig()
    fast_dev_run: bool = False
    resume_from_checkpoint: Optional[str] = None
    val_check_interval: int = 5000


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="data", name="default", node=DataConfig)
cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="model", name="default", node=TransformerConfig)
cs.store(group="optimizer", name="default", node=OptimizerConfig)
cs.store(group="logging", name="default", node=LoggingConfig)


@hydra.main(config_name="config")
def train(cfg: Config) -> None:
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    vocab = Vocab.from_esm_alphabet(alphabet)
    train_data = EncodedFastaDataset(cfg.data.fasta_path, vocab)
    train_data = RandomCropDataset(train_data, cfg.model.max_seqlen)
    train_data = MaskedTokenWrapperDataset(
        train_data,
        cfg.train.mask_prob,
        cfg.train.random_token_prob,
        cfg.train.random_token_prob,
    )
    sampler = BatchBySequenceLength(train_data, cfg.train.max_tokens, shuffle=True)
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=train_data.collater,
    )

    with open(Path(cfg.data.trrosetta_path) / cfg.data.trrosetta_train_split) as f:
        train_pdbs = f.read().splitlines()

    with open(Path(cfg.data.trrosetta_path) / cfg.data.trrosetta_valid_split) as f:
        valid_pdbs = f.read().splitlines()

    trrosetta_train_data = TRRosettaContactDataset(
        cfg.data.trrosetta_path,
        vocab,
        split_files=train_pdbs,
        max_seqs_per_msa=1,
    )

    trrosetta_valid_data = TRRosettaContactDataset(
        cfg.data.trrosetta_path,
        vocab,
        split_files=valid_pdbs,
        max_seqs_per_msa=1,
    )

    valid_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        trrosetta_valid_data,
        batch_size=cfg.train.valid_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=trrosetta_valid_data.collater,
    )

    model = ESM1b(
        vocab=vocab,
        model_config=cfg.model,
        optimizer_config=cfg.optimizer,
        contact_train_data=trrosetta_train_data,
    )

    # Requires wandb to be installed
    logger = (
        pl.loggers.WandbLogger(project=cfg.logging.wandb_project)
        if cfg.logging.wandb_project is not None
        else True
    )

    if isinstance(logger, pl.loggers.LightningLoggerBase):
        logger.log_hyperparams(cfg.train)  # type: ignore
        logger.log_hyperparams(cfg.model)  # type: ignore

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
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback, lr_logger],
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        distributed_backend=cfg.train.distributed_backend,
        fast_dev_run=cfg.fast_dev_run,
        gpus=cfg.train.gpus,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        max_epochs=cfg.train.max_epochs,
        max_steps=cfg.optimizer.max_steps,
        num_nodes=cfg.train.num_nodes,
        precision=cfg.train.precision,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        track_grad_norm=cfg.logging.track_grad_norm,
        val_check_interval=cfg.val_check_interval * cfg.train.accumulate_grad_batches,
        replace_sampler_ddp=False,
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    train()
