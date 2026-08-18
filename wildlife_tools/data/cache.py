import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generic, TypeVar

import lmdb
import torch
from tqdm import tqdm

from ..tools import check_dataset_output
from .dataset import FeatureDataset, ImageDataset

TBatch = tuple[torch.Tensor, torch.Tensor]
TDict = TypeVar("TDict")  # np.ndarray | dict
TFeature = TypeVar("TFeature", bound=Sequence)  # np.ndarray | list[dict]
TModel = TypeVar("TModel", bound=Sequence)  # torch.Tensor | list[dict]


class CacheMixin(ABC, Generic[TModel]):
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str | None = "cpu",
        cache_path: str | None = None,
        skip_cache_check: bool = False,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.skip_cache_check = skip_cache_check

    @abstractmethod
    def process_batch(self, batch: TBatch) -> TModel:
        pass

    def _save_entry(self, txn: lmdb.Transaction, key: bytes, entry) -> None:
        txn.put(key, pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))

    def get_key(self, dataset: ImageDataset, index: int) -> str:
        return str(dataset.metadata["image_id"][index])

    def make_loader(self, dataset: ImageDataset) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class FeatureCacheMixin(CacheMixin, Generic[TDict, TFeature, TModel]):
    @abstractmethod
    def cat_features_dictionary(self, feats: list[TDict]) -> TFeature:
        pass

    @abstractmethod
    def cat_features_model(self, feats: list[TModel]) -> TFeature:
        pass

    @abstractmethod
    def forward_batch(self, batch: TBatch) -> TModel:
        pass

    def __call__(self, dataset: ImageDataset) -> FeatureDataset:
        """
        Extract features from input dataset and return them as a new FeatureDataset.

        Args:
            dataset (ImageDataset): Extract features from this dataset.

        Returns:
            feature_dataset (FeatureDataset): A FeatureDataset containing the extracted features
        """

        check_dataset_output(dataset, check_label=False)
        self.model = self.model.to(self.device).eval()
        features = self.extract_with_cache(dataset)
        self.model = self.model.to("cpu")

        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label,
        )

    def _open_env(self) -> lmdb.Environment:
        assert self.cache_path is not None
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        return lmdb.open(
            str(self.cache_path),
            map_size=1 << 40,
            subdir=True,
            lock=True,
            readahead=False,
            meminit=False,
        )

    def extract_with_cache(self, dataset: ImageDataset) -> TFeature:

        # Handle the case when cache is not required
        if self.cache_path is None:
            loader = self.make_loader(dataset)
            feats = []
            for batch in tqdm(loader, mininterval=1, ncols=100, desc='Extracting features'):
                feats.append(self.process_batch(batch))
            return self.cat_features_model(feats)

        # Load the cache
        env = self._open_env()
        keys = [self.get_key(dataset, i) for i in range(len(dataset))]

        if not self.skip_cache_check:
            # Determine missing entries
            num_checkers = max(1, min(max(self.num_workers, 1), len(keys)))
            if num_checkers == 1:
                missing = []
                with env.begin() as txn:
                    for i, k in tqdm(enumerate(keys), desc='Checking missing cache'):
                        if txn.get(k.encode()) is None:
                            missing.append(i)
            else:
                chunk_size = (len(keys) + num_checkers - 1) // num_checkers
                chunks = [
                    range(i, min(i + chunk_size, len(keys)))
                    for i in range(0, len(keys), chunk_size)
                ]

                def find_missing(indices: range) -> list[int]:
                    with env.begin() as txn:
                        return [
                            i for i in indices
                            if txn.get(keys[i].encode()) is None
                        ]

                with ThreadPoolExecutor(max_workers=num_checkers) as executor:
                    missing_chunks = list(tqdm(
                        executor.map(find_missing, chunks),
                        total=len(chunks),
                        desc='Checking missing cache',
                    ))
                missing = [i for chunk in missing_chunks for i in chunk]

            if missing:
                # Define loader on the missing entries
                subset = torch.utils.data.Subset(dataset, missing)
                loader = self.make_loader(subset)

                # Load the missing entries
                ptr = 0
                for batch in tqdm(loader, mininterval=1, ncols=100, desc=f'Loading {len(subset)} missing entries'):
                    feats = self.forward_batch(batch)

                    # Write the batch
                    with env.begin(write=True) as txn:
                        for j in range(len(feats)):
                            key = keys[missing[ptr]].encode()
                            self._save_entry(txn, key, feats[j])
                            ptr += 1

        # Read all features back in order
        def read_entries(entries: list[str]) -> list[TDict]:
            with env.begin() as txn:
                return [pickle.loads(txn.get(k.encode())) for k in entries]

        num_readers = max(1, min(max(self.num_workers, 1), len(keys)))
        if num_readers == 1:
            outputs = []
            with env.begin() as txn:
                for k in tqdm(keys, desc='Reading cache'):
                    outputs.append(pickle.loads(txn.get(k.encode())))
        else:
            chunk_size = (len(keys) + num_readers - 1) // num_readers
            chunks = [
                keys[i:i + chunk_size]
                for i in range(0, len(keys), chunk_size)
            ]
            with ThreadPoolExecutor(max_workers=num_readers) as executor:
                chunk_outputs = list(tqdm(
                    executor.map(read_entries, chunks),
                    total=len(chunks),
                    desc='Reading cache',
                ))
            outputs = [output for chunk in chunk_outputs for output in chunk]

        # Close the cache
        env.close()

        # Merge the extracted features
        return self.cat_features_dictionary(outputs)

    def process_batch(self, batch: TBatch) -> TModel:
        return self.forward_batch(batch)
