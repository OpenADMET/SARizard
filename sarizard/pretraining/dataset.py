import math

import numpy as np
import torch
import zarr
from chemprop.data.collate import BatchMolGraph, TrainingBatch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit.Chem import MolFromSmiles

from config import CHUNKS_PER_BATCH


class ChempropChunkwiseZarrDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: list[str], zarr_store: str, featurizer: SimpleMoleculeMolGraphFeaturizer):
        self.smiles = np.array(smiles)
        self.z = zarr.open_array(zarr_store)
        # raise rather than assert: asserts are stripped under python -O, and a silent
        # smiles/target misalignment would train every molecule against the wrong target row
        if self.z.shape[0] != len(smiles):
            raise ValueError(
                f"smiles/target row mismatch: {len(smiles)} smiles vs "
                f"{self.z.shape[0]} target rows"
            )

        self.n_rows = len(smiles)
        self.chunksize = self.z.chunks[0]

        # Calculate the effective size of a batch (multiple chunks)
        self.items_per_batch = self.chunksize * CHUNKS_PER_BATCH

        # Update length to reflect the new number of multi-chunk groups
        self.len = math.ceil(self.n_rows / self.items_per_batch)
        self.featurizer = featurizer

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        # Calculate start and stop indices based on the combined batch size
        start_idx = idx * self.items_per_batch
        stop_idx = min(start_idx + self.items_per_batch, self.n_rows)

        # Zarr handles cross-chunk slicing automatically
        targets = torch.tensor(self.z[start_idx:stop_idx, :], dtype=torch.float32)
        weights = torch.ones((stop_idx - start_idx, 1), dtype=torch.float32)

        # featurize each molecule, failing loudly on an unparseable SMILES rather than
        # passing None into the featurizer where it would raise something opaque
        graphs = []
        for s in self.smiles[start_idx:stop_idx]:
            mol = MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"unparseable SMILES in corpus: {s!r}")
            graphs.append(self.featurizer(mol))

        return TrainingBatch(BatchMolGraph(graphs), None, None, targets, weights, None, None)
