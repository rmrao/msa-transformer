from typing import Optional, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import sklearn.linear_model
from tqdm import tqdm

from evo.metrics import compute_precisions
from evo.tensor import symmetrize, apc

from modules import (
    TransformerLayer,
    AxialTransformerLayer,
    ContactPredictionHead,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    RowSelfAttention,
    ColumnSelfAttention,
)

import lr_schedulers
from dataset import TRRosettaContactDataset


class Average(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, values: torch.Tensor):  # type: ignore
        self.total += values.sum().float()
        self.samples += values.numel()  # type: ignore

    def compute(self):
        return self.total / self.samples


class BaseProteinModel(pl.LightningModule, ABC):
    def __init__(
        self,
        vocab_size: int,
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        prepend_bos: bool = True,
        append_eos: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        warmup_steps: int = 0,
        max_steps: int = 10000,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
    ):
        super().__init__()
        self.alphabet_size = vocab_size
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.adam_betas = adam_betas
        self.contact_train_data = contact_train_data

        self.metrics = nn.ModuleDict(
            {
                "valid/Long Range AUC": Average(),
                "valid/Long Range P@L": Average(),
                "valid/Long Range P@L2": Average(),
                "valid/Long Range P@L5": Average(),
            }
        )

    @abstractmethod
    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        return NotImplemented

    @abstractmethod
    def get_sequence_attention(self, tokens):
        return NotImplemented

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def on_validation_epoch_start(self):
        self.train_contact_regression()

    def train_contact_regression(self, verbose=False):
        data = self.contact_train_data
        if data is None:
            raise RuntimeError(
                "Cannot train regression without trRosetta contact training set."
            )
        X = []
        y = []
        with torch.no_grad():
            iterable = data if not verbose else tqdm(data)
            for tokens, contacts in iterable:
                tokens = tokens.unsqueeze(0)
                attentions = self.get_sequence_attention(tokens)
                start_idx = int(self.prepend_bos)
                end_idx = attentions.size(-1) - int(self.append_eos)
                attentions = attentions[..., start_idx:end_idx, start_idx:end_idx]
                seqlen = attentions.size(-1)
                attentions = symmetrize(attentions)
                attentions = apc(attentions)
                attentions = attentions.view(-1, seqlen, seqlen).cpu().numpy()
                sep = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
                mask = sep >= 6
                attentions = attentions[:, mask]
                contacts = contacts[mask]
                X.append(attentions.T)
                y.append(contacts)

        X = np.concatenate(X, 0)
        y = np.concatenate(y, 0)

        clf = sklearn.linear_model.LogisticRegression(
            penalty="l1",
            C=0.15,
            solver="liblinear",
            verbose=verbose,
            random_state=0,
        )
        clf.fit(X, y)

        self.contact_head.regression.load_state_dict(
            {
                "weight": torch.from_numpy(clf.coef_),
                "bias": torch.from_numpy(clf.intercept_),
            }
        )

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src)["logits"]
        valid_mask = tgt != self.pad_idx

        logits = logits[valid_mask]
        tgt = tgt[valid_mask]
        loss = nn.CrossEntropyLoss(reduction="none")(logits, tgt)
        perplexity = loss.float().exp().mean()
        loss = loss.mean()

        self.log("train/loss", loss)
        self.log("train/perplexity", perplexity, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self.predict_contacts(batch["src_tokens"])
        metrics = compute_precisions(
            predictions,
            batch["tgt"],
            batch["tgt_lengths"],
            minsep=24,
        )

        for key, value in metrics.items():
            key = f"valid/Long Range {key}"
            logger = self.metrics[key](value)
            self.log(key, logger, prog_bar=key.endswith("P@L"))

    def configure_optimizers(self):
        no_decay = ["norm", "LayerNorm"]

        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.optimizer == "adam":
            optimizer_type = torch.optim.AdamW
        elif self.optimizer == "lamb":
            try:
                from apex.optimizers import FusedLAMB
            except ImportError:
                raise ImportError("Apex must be installed to use FusedLAMB optimizer.")
            optimizer_type = FusedLAMB
        optimizer = optimizer_type(optimizer_grouped_parameters, lr=self.learning_rate, betas=self.adam_betas)
        scheduler = lr_schedulers.get(self.lr_scheduler)(
            optimizer, self.warmup_steps, self.max_steps
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]


class ESM1b(BaseProteinModel):
    def __init__(
        self,
        vocab_size: int,
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        prepend_bos: bool = True,
        append_eos: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        warmup_steps: int = 0,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        max_steps: int = 10000,
        contact_train_data: Optional[TRRosettaContactDataset] = None,
        embed_dim: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seqlen: int = 1024,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            mask_idx=mask_idx,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            adam_betas=adam_betas,
            contact_train_data=contact_train_data,
        )
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout

        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation_dropout=self.activation_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.embed_positions = LearnedPositionalEmbedding(
            max_seqlen,
            self.embed_dim,
            self.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(self.embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.num_attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.contact_head.requires_grad_(False)

        self.init_weights()

    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        if return_contacts:
            need_head_weights = True
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.pad_idx)  # B, T

        x = self.embed_tokens(tokens)

        x = x + self.embed_positions(tokens)

        x = self.emb_layer_norm_before(x)
        x = self.dropout_layer(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(
                    2
                )
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def get_sequence_attention(self, tokens):
        return self(tokens.to(device=self.device), need_head_weights=True)["attentions"]

    @classmethod
    def from_esm(cls):
        import esm
        from evo.tokenization import Vocab

        esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        args = esm_model.args
        vocab = Vocab.from_esm_alphabet(alphabet)
        model = cls(
            vocab_size=len(vocab),  # can be off b/c of null token
            bos_idx=vocab.bos_idx,
            eos_idx=vocab.eos_idx,
            pad_idx=vocab.pad_idx,
            mask_idx=vocab.mask_idx,
            prepend_bos=vocab.prepend_bos,
            append_eos=vocab.append_eos,
            embed_dim=args.embed_dim,
            num_attention_heads=args.attention_heads,
            num_layers=args.layers,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
        )

        model.load_state_dict(esm_model.state_dict())

        return model


class MSATransformer(BaseProteinModel):
    def __init__(
        self,
        vocab_size: int,
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        prepend_bos: bool = True,
        append_eos: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        warmup_steps: int = 0,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        max_steps: int = 10000,
        contact_train_data: Optional[TRRosettaContactDataset] = None,
        embed_dim: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        embed_positions_msa: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
        max_seqlen: int = 1024,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            mask_idx=mask_idx,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            adam_betas=adam_betas,
            max_steps=max_steps,
            contact_train_data=contact_train_data,
        )
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.embed_positions_msa = embed_positions_msa
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_tokens_per_msa = max_tokens_per_msa

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, embed_dim, padding_idx=self.pad_idx
        )

        if embed_positions_msa:
            self.msa_position_embedding = nn.Parameter(
                0.01 * torch.randn(1, 1024, 1, 1),
                requires_grad=True,
            )
        else:
            self.register_parameter("msa_position_embedding", None)  # type: ignore

        self.dropout_module = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=4 * embed_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    max_tokens_per_msa=max_tokens_per_msa,
                )
                for _ in range(num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            num_layers * num_attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.contact_head.requires_grad_(False)
        self.embed_positions = LearnedPositionalEmbedding(
            max_seqlen,
            embed_dim,
            self.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

        self.init_weights()

    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.pad_idx)  # B, R, C
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens)
        x += self.embed_positions(
            tokens.view(batch_size * num_alignments, seqlen)
        ).view(x.size())
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x += self.msa_position_embedding[:, :num_alignments]

        x = self.emb_layer_norm_before(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            row_attn_weights = []
            col_attn_weights = []

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                x, col_attn, row_attn = x
                # H x C x B x R x R -> B x H x C x R x R
                col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
                # H x B x C x C -> B x H x C x C
                row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # col_attentions: B x L x H x C x R x R
            col_attentions = torch.stack(col_attn_weights, 1)
            # row_attentions: B x L x H x C x C
            row_attentions = torch.stack(row_attn_weights, 1)
            result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if return_contacts:
                contacts = self.contact_head(tokens, row_attentions)
                result["contacts"] = contacts

        return result

    def max_tokens_per_msa_(self, value: int) -> None:
        """The MSA Transformer automatically batches attention computations when
        gradients are disabled to allow you to pass in larger MSAs at test time than
        you can fit in GPU memory. By default this occurs when more than 2^14 tokens
        are passed in the input MSA. You can set this value to infinity to disable
        this behavior.
        """
        self.max_tokens_per_msa = value
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value

    def get_sequence_attention(self, tokens):
        return self(tokens.to(device=self.device), need_head_weights=True)[
            "row_attentions"
        ]

    @classmethod
    def from_esm(cls):
        import esm
        from evo.tokenization import Vocab

        esm_model, alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
        args = esm_model.args
        vocab = Vocab.from_esm_alphabet(alphabet)
        model = cls(
            vocab_size=len(vocab),  # can be off b/c of null token
            bos_idx=vocab.bos_idx,
            eos_idx=vocab.eos_idx,
            pad_idx=vocab.pad_idx,
            mask_idx=vocab.mask_idx,
            prepend_bos=vocab.prepend_bos,
            append_eos=vocab.append_eos,
            embed_dim=args.embed_dim,
            num_attention_heads=args.attention_heads,
            num_layers=args.layers,
            embed_positions_msa=args.embed_positions_msa,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_tokens_per_msa=getattr(args, "max_tokens_per_msa", args.max_tokens),
        )

        model.load_state_dict(esm_model.state_dict())

        return model
