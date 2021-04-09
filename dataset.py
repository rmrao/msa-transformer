from typing import List, Any, Optional, Collection, Tuple, Dict
from pathlib import Path
import torch
from evo.ffindex import MSAFFindex
from evo.tokenization import Vocab
from evo.typed import PathLike
from evo.dataset import CollatableVocabDataset, NPZDataset, A3MDataset
from evo.tensor import collate_tensors
import esm


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
        return tokens, contacts

    def collater(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        tokens, contacts = tuple(zip(*batch))
        src_tokens = collate_tensors(tokens, constant_value=self.vocab.pad_idx)
        targets = collate_tensors(contacts, constant_value=-1, dtype=torch.long)
        src_lengths = torch.tensor(
            [contact.size(1) for contact in contacts], dtype=torch.long
        )

        result = {
            "src_tokens": src_tokens,
            "tgt": targets,
            "tgt_lengths": src_lengths,
        }

        return result
