from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import sklearn.linear_model
from tqdm import tqdm
from dataclasses import dataclass, field

from evo.tokenization import Vocab
from evo.metrics import compute_precisions
from evo.tensor import symmetrize, apc

from modules import (
    TransformerLayer,
    PKMLayer,
    AxialTransformerLayer,
    ContactPredictionHead,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    RowSelfAttention,
    ColumnSelfAttention,
)
from product_key_memory import PKM

import lr_schedulers
from dataset import TRRosettaContactDataset
import esm


@dataclass
class TransformerLayerConfig:
    embed_dim: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_type: str = "standard"
    performer_attention_features: int = 256


@dataclass
class PKMLayerConfig(TransformerLayerConfig):
    pkm_attention_heads: int = 8
    num_product_keys: int = 1024
    pkm_topk: int = 32


@dataclass
class TransformerConfig:
    layer: TransformerLayerConfig = TransformerLayerConfig()
    pkm: PKMLayerConfig = PKMLayerConfig()
    num_layers: int = 12
    max_seqlen: int = 1024
    pkm_layers: List[int] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "warmup_linear"
    warmup_steps: int = 16000
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    max_steps: int = 1000000


class BaseProteinModel(pl.LightningModule, ABC):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.optimizer_config = optimizer_config
        self.contact_train_data = contact_train_data

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
            for tokens, contacts, fam_toks in iterable:
                tokens, fam_toks = tokens.unsqueeze(0), fam_toks.unsqueeze(0)
                attentions = self.get_sequence_attention((tokens,fam_toks))
                start_idx = int(self.vocab.prepend_bos)
                end_idx = attentions.size(-1) - int(self.vocab.append_eos)
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
        print("done with forward pass in training step")
        valid_mask = tgt != self.vocab.pad_idx

        logits = logits[valid_mask]
        tgt = tgt[valid_mask]
        print("about to calculate loss")
        loss = nn.CrossEntropyLoss(reduction="none")(logits, tgt)
        print("done calculating loss")
        perplexity = loss.float().exp().mean()
        print("perplexity calculated")
        loss = loss.mean()

        self.log("train/loss", loss)
        self.log("train/perplexity", perplexity, prog_bar=True)
        print("training step func done")
        return loss

    def validation_step(self, batch, batch_idx):
        print("In validation_step")
        predictions = self.predict_contacts((batch["src_tokens"], batch["family_tokens"]))
        metrics = compute_precisions(
            predictions,
            batch["tgt"],
            batch["tgt_lengths"],
            minsep=24,
        )

        for key, value in metrics.items():
            key = f"valid/Long Range {key}"
            self.log(key, value, prog_bar=key.endswith("P@L"))
        return metrics["P@L"]

    def configure_optimizers(self):
        no_decay = ["norm", "LayerNorm"]

        pkm_params = []
        for module in self.modules():
            if isinstance(module, PKM):
                pkm_params.append(module.values.weight)
        pkm_paramset = set(pkm_params)

        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if param in pkm_paramset:
                continue

            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.optimizer_config.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
            {
                "params": pkm_params,
                "weight_decay": 0.0,
                "lr": 4 * self.optimizer_config.learning_rate,
            },
        ]

        if self.optimizer_config.name == "adam":
            optimizer_type = torch.optim.AdamW
        elif self.optimizer_config.name == "lamb":
            try:
                from apex.optimizers import FusedLAMB
            except ImportError:
                raise ImportError("Apex must be installed to use FusedLAMB optimizer.")
            optimizer_type = FusedLAMB
        optimizer = optimizer_type(
            optimizer_grouped_parameters,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.adam_betas,
        )
        scheduler = lr_schedulers.get(self.optimizer_config.lr_scheduler)(
            optimizer,
            self.optimizer_config.warmup_steps,
            self.optimizer_config.max_steps,
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]


class ESM1b(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        family_alphabet: esm.Alphabet,
        model_config: TransformerConfig = TransformerConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
    ):
        super().__init__(
            vocab=vocab,
            optimizer_config=optimizer_config,
            contact_train_data=contact_train_data,
        )
        self.model_config = model_config
        self.family_alphabet = family_alphabet

        self.embed_tokens = self.build_embedding()
        self.embed_family_tokens = self.build_family_embedding()
        self.dropout_layer = nn.Dropout(model_config.layer.dropout)

        self.layers = nn.ModuleList([])
        for i in range(self.model_config.num_layers):
            if i in self.model_config.pkm_layers:
                layer: Union[TransformerLayer, PKMLayer] = self.build_pkm_layer()
            else:
                layer = self.build_transformer_layer()
            self.layers.append(layer)

        self.embed_positions = LearnedPositionalEmbedding(
            model_config.max_seqlen,
            self.model_config.layer.embed_dim,
            vocab.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(self.model_config.layer.embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(self.model_config.layer.embed_dim)
        self.lm_head = self.build_lm_head(weight=self.embed_tokens.weight)
        self.contact_head = self.build_contact_head()

        self.init_weights()

    def build_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            len(self.vocab),
            self.model_config.layer.embed_dim,
            padding_idx=self.vocab.pad_idx,
        )

    def build_family_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            len(self.family_alphabet),
            self.model_config.layer.embed_dim,
            padding_idx=0,
        )

    def build_transformer_layer(self) -> TransformerLayer:
        config = self.model_config.layer
        return TransformerLayer(
            embed_dim=config.embed_dim,
            ffn_embed_dim=4 * config.embed_dim,
            attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            attention_type=config.attention_type,
            performer_attention_features=config.performer_attention_features,
        )

    def build_pkm_layer(self) -> PKMLayer:
        config = self.model_config.pkm
        return PKMLayer(
            embed_dim=config.embed_dim,
            ffn_embed_dim=4 * config.embed_dim,
            attention_heads=config.num_attention_heads,
            pkm_attention_heads=config.pkm_attention_heads,
            pkm_dim_head=config.embed_dim // config.num_attention_heads,
            num_product_keys=config.num_product_keys,
            pkm_topk=config.pkm_topk,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            attention_type=config.attention_type,
            performer_attention_features=config.performer_attention_features,
        )

    def build_lm_head(self, weight: torch.Tensor) -> RobertaLMHead:
        return RobertaLMHead(
            embed_dim=self.model_config.layer.embed_dim,
            output_dim=len(self.vocab),
            weight=weight,
        )

    def build_contact_head(self) -> ContactPredictionHead:
        contact_head = ContactPredictionHead(
            self.model_config.num_layers * self.model_config.layer.num_attention_heads,
            self.vocab.prepend_bos,
            self.vocab.append_eos,
            eos_idx=self.vocab.eos_idx,
        )
        contact_head.requires_grad_(False)
        return contact_head

    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        if not isinstance(tokens, tuple):
            raise RuntimeError("forward argument isn't a tuple")

        if return_contacts:
            need_head_weights = True
        
        tokens, family_tokens = tokens[0], tokens[1]
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, T
        family_embedding = self.embed_family_tokens(family_tokens)
        x = self.embed_tokens(tokens) + torch.sum(family_embedding, dim=1)

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
            attentions = []

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

                attentions.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        print("Done with emb layer norm after")
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            print("Starting need head weights process")
            # attentions: B x L x H x T x T
            attentions = torch.stack(attentions, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(
                    2
                )
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                print("in return contacts")
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts
                print("done with return contacts")
        print("done with need head weights")

        return result

    def get_sequence_attention(self, tokens):
        tokens, family_tokens = tokens
        return self((tokens.to(device=self.device), family_tokens.to(device=self.device)), need_head_weights=True)["attentions"]

    @classmethod
    def from_esm(
        cls,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[TRRosettaContactDataset] = None,
    ):
        import esm
        from evo.tokenization import Vocab

        esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        args = esm_model.args
        vocab = Vocab.from_esm_alphabet(alphabet)
        layer_config = TransformerLayerConfig(
            embed_dim=args.embed_dim,
            num_attention_heads=args.attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
        )
        model_config = TransformerConfig(
            layer=layer_config,
            num_layers=args.layers,
        )

        model = cls(
            vocab=vocab,
            model_config=model_config,
            optimizer_config=optimizer_config,
            contact_train_data=contact_train_data,
        )
        model.load_state_dict(esm_model.state_dict())
        return model


class MSATransformer(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
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
            vocab=vocab,
            optimizer_config=optimizer_config,
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
            len(vocab), embed_dim, padding_idx=vocab.pad_idx
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
            vocab.prepend_bos,
            vocab.append_eos,
            eos_idx=vocab.eos_idx,
        )
        self.contact_head.requires_grad_(False)
        self.embed_positions = LearnedPositionalEmbedding(
            max_seqlen,
            embed_dim,
            vocab.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            output_dim=len(self.vocab),
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
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, R, C
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
            vocab=vocab,
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