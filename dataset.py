from typing import List, Any, Optional, Collection, Tuple, Dict
from pathlib import Path
import torch
from evo.ffindex import MSAFFindex
from evo.tokenization import Vocab
from evo.typed import PathLike
from evo.dataset import CollatableVocabDataset, NPZDataset, A3MDataset, FastaDataset, RandomCropDataset, MaskedTokenWrapperDataset
from evo.tensor import collate_tensors
import esm
import pandas as pd
import numpy as np

class MSADataset(CollatableVocabDataset):
    def __init__(self, ffindex_path: PathLike):
        vocab = Vocab.from_esm_alphabet(
            esm.data.Alphabet.from_architecture("MSA Transformer")
        )
        super().__init__(vocab)

        ffindex_path = Path(ffindex_path)
        index_file = ffindex_path.with_suffix(".ffindex")
        data_file = ffindex_path.with_suffix(".ffdata")
        self.ffindex = MSAFFindex(index_file, data_file)

    def __len__(self):
        return len(self.ffindex)

    def __getitem__(self, idx):
        msa = self.ffindex[idx]
        return torch.from_numpy(self.vocab.encode(msa))

    def collater(self, batch: List[Any]) -> Any:
        return collate_tensors(batch)


class TRRosettaContactDataset(CollatableVocabDataset):
    def __init__(
        self,
        data_path: PathLike,
        vocab: Vocab,
        pfam_data_file: PathLike,
        pfam_alphabet_arr: PathLike,
        split_files: Optional[Collection[str]] = None,
        max_seqs_per_msa: Optional[int] = 64,
    ):
        super().__init__(vocab)

        data_path = Path(data_path)
        self.a3m_data = A3MDataset(
            data_path / "a3m",
            split_files=split_files,
            max_seqs_per_msa=max_seqs_per_msa,
        )
        self.npz_data = NPZDataset(
            data_path / "npz", split_files=split_files, lazy=True
        )

        self.pfam_data_file = pfam_data_file
        self.pfam_df = pd.read_parquet(pfam_data_file)
        self.family_alphabet = esm.Alphabet(np.load(pfam_alphabet_arr, allow_pickle=True), 
                                            prepend_toks = ("<pad>", "<eos>", "<unk>"))

        assert len(self.a3m_data) == len(self.npz_data)

    def get(self, key: str):
        msa = self.a3m_data.get(key)
        tokens = torch.from_numpy(self.vocab.encode(msa))
        distogram = self.npz_data.get(key)["dist6d"]
        contacts = (distogram > 0) & (distogram < 8)
        contacts = torch.from_numpy(contacts)
        return tokens, contacts

    def __len__(self) -> int:
        return len(self.a3m_data)

    def __getitem__(self, index):
        msa = self.a3m_data[index]
        tokens = torch.from_numpy(self.vocab.encode(msa))
        distogram = self.npz_data[index]["dist6d"]
        contacts = (distogram > 0) & (distogram < 8)
        contacts = torch.from_numpy(contacts)

        desc = self.npz_data.key(index)
        # Subtract two to account for bos and eos
        fam_toks = torch.zeros(31, tokens.shape[0] - 2, dtype=torch.int64)
        try:
            tgt = self.pfam_df.loc[desc]
            # print(desc, "WAS FOUND IN DATAFRAME")
            for seg_idx in range(len(tgt["accession"])):
                seg_fam, seg_start, seg_end = tgt["accession"][seg_idx], tgt["ali_from"][seg_idx], tgt["ali_to"][seg_idx]
                #account for inclusive 1-indexing of dataframe
                seg_start -= 1 
                fam_toks[seg_idx, seg_start:seg_end] = self.family_alphabet.get_idx(seg_fam)
        except KeyError:
            # print(desc, "NOT FOUND IN DATAFRAME")
            pass

        # self.family_vocab.add_special_tokens(fam_toks) analog
        pad_widths = [(0, 0)] * (fam_toks.ndim - 1) + [
            (1, 1)
        ]
        fam_toks = torch.from_numpy(
            np.pad(
            fam_toks,
            pad_widths,
            constant_values=[(self.family_alphabet.get_idx("<cls>"), self.family_alphabet.get_idx("<eos>"))],
        ))
        assert fam_toks.shape[1] == tokens.shape[0]

        return tokens, contacts, fam_toks

    def collater(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        tokens, contacts, fam_toks = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )
        fam_toks = collate_tensors(fam_toks, constant_value=0)

        result = {
            "src_tokens": src_tokens,
            "tgt": targets,
            "tgt_lengths": src_lengths,
            "family_tokens": fam_toks,
        }

        return result

class EncodedFamilyFastaDataset(FastaDataset):
    def __init__(self, fasta_data_file: PathLike, vocab:Vocab, pfam_data_file: PathLike, pfam_alphabet_arr: PathLike):
        super().__init__(data_file=fasta_data_file, cache_indices=True)
        self.vocab = vocab
        self.pfam_data_file = pfam_data_file
        self.pfam_df = pd.read_parquet(pfam_data_file)
        self.family_alphabet = esm.Alphabet(np.load(pfam_alphabet_arr, allow_pickle=True), 
                                            prepend_toks = ("<pad>", "<eos>", "<unk>"))

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        desc, seq = super().__getitem__(index)
        tokens = torch.from_numpy(self.vocab.encode_single_sequence(seq))
        # Subtract two to account for bos and eos
        fam_toks = torch.zeros(136, tokens.shape[0] - 2, dtype=torch.int64)
        try:
            tgt = self.pfam_df.loc[desc]
            # print(desc, "WAS FOUND IN DATAFRAME")
            for seg_idx in range(len(tgt["accession"])):
                seg_fam, seg_start, seg_end = tgt["accession"][seg_idx], tgt["ali_from"][seg_idx], tgt["ali_to"][seg_idx]
                #account for inclusive 1-indexing of dataframe
                seg_start -= 1 
                fam_toks[seg_idx, seg_start:seg_end] = self.family_alphabet.get_idx(seg_fam)
        except KeyError:
            # print(desc, "NOT FOUND IN DATAFRAME")
            pass

        # self.family_vocab.add_special_tokens(fam_toks) analog
        pad_widths = [(0, 0)] * (fam_toks.ndim - 1) + [
            (1, 1)
        ]
        fam_toks = torch.from_numpy(
            np.pad(
            fam_toks,
            pad_widths,
            constant_values=[(self.family_alphabet.get_idx("<cls>"), self.family_alphabet.get_idx("<eos>"))],
        ))
        assert fam_toks.shape[1] == tokens.shape[0]
        return tokens, fam_toks

class RandomCropFamilyDataset(RandomCropDataset):
    def __init__(self, family_dataset: EncodedFamilyFastaDataset, max_seqlen: int):
        super().__init__(family_dataset, max_seqlen)

    def __getitem__(self, idx):
        item, family_item = self.dataset[idx]
        seqlen = item.size(-1)
        if seqlen > self.max_seqlen:
            low_idx = int(self.vocab.prepend_bos)
            high_idx = seqlen - int(self.vocab.append_eos)
            start_idx = np.random.randint(low_idx, high_idx)
            end_idx = start_idx + self.max_seqlen_no_special
            item = torch.cat(
                [
                    item[..., :low_idx],
                    item[..., start_idx:end_idx],
                    item[..., high_idx:],
                ],
                -1,
            )
            family_item = torch.cat(
                [
                    family_item[..., :low_idx],
                    family_item[..., start_idx:end_idx],
                    family_item[..., high_idx:],
                ],
                -1,
            )
        return item, family_item

class MaskedTokenWrapperFamilyDataset(MaskedTokenWrapperDataset):
    def __init__(
        self,
        dataset: CollatableVocabDataset,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
    ):
        super().__init__(dataset, mask_prob, random_token_prob, leave_unmasked_prob)

    def __getitem__(self, idx):
        item, family_item = self.dataset[idx]
        random_probs = torch.rand_like(item, dtype=torch.float)
        random_probs[(item == self.vocab.bos_idx) | (item == self.vocab.eos_idx)] = 1
        do_mask = random_probs < self.mask_prob

        tgt = item.masked_fill(~do_mask, self.vocab.pad_idx)
        mask_with_token = random_probs < (
            self.mask_prob * (1 - self.leave_unmasked_prob)
        )
        src = item.masked_fill(mask_with_token, self.vocab.mask_idx)
        mask_with_random = random_probs < (self.mask_prob * self.random_token_prob)
        # TODO - maybe prevent special tokens?
        rand_tokens = torch.randint_like(src, len(self.vocab))
        src[mask_with_random] = rand_tokens[mask_with_random]
        return (src, tgt, family_item)

    def collater(self, batch: List[Any]) -> Any:
        src = collate_tensors(
            [el[0] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        tgt = collate_tensors(
            [el[1] for el in batch],
            constant_value=self.vocab.pad_idx,
        )
        fam = collate_tensors(
            [el[2] for el in batch],
            constant_value=0,
        )
        return ((src, fam), tgt)