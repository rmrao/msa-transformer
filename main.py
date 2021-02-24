import sys
import operator
import logging
import pytorch_lightning as pl
import torch
import esm
from evo.dataset import RandomCropDataset, SubsampleMSADataset, AutoBatchDataset
from evo.tokenize import Vocab
from model import MSATransformer
from dataset import MSADataset, TRRosettaContactDataset


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)


def train():
    alphabet = esm.data.Alphabet.from_architecture("MSA Transformer")
    vocab = Vocab.from_esm_alphabet(alphabet)
    model = MSATransformer(
        vocab_size=len(vocab),
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        pad_idx=vocab.pad_idx,
        mask_idx=vocab.mask_idx,
        prepend_bos=vocab.prepend_bos,
        append_eos=vocab.append_eos,
        # TODO: fill out config
    )

    train_data = MSADataset(ffindex_path)
    train_data = RandomCropDataset(train_data, max_seqlen)
    train_data = SubsampleMSADataset(train_data, max_tokens)
    train_loader = AutoBatchDataset(train_data, max_tokens)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=operator.itemgetter(0),
    )

    valid_data = TRRosettaContactDataset(
        trrosetta_path,
        split_files=split_files,
        max_seqs_per_msa=max_seqs_validation,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        collate_fn=valid_data.collater,
    )

    kwargs = {}
    if wandb_project:
        try:
            # Requires wandb to be installed
            logger = pl.loggers.WandbLogger(project=wandb_project)
            # logger.log_hyperparams(args)
            kwargs["logger"] = logger
        except ImportError:
            raise ImportError(
                "Cannot use W&B logger w/o W&b install. Run `pip install wandb` first."
            )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid/Long Range P@L",
        mode="max",
        save_top_k=5,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid/Long Range P@L",
        mode="max",
        patience=patience,
    )

    # Initialize Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        **kwargs
    )
    trainer.fit(model, train_loader=train_loader, valid_loader=valid_loader)
