"""Lazy Tiny-ImageNet data plumbing for Fluke.

Fluke algorithms often inspect ``FastDataLoader.tensors`` directly.  This module
preserves that public surface while keeping image files path-backed until a batch or
an explicitly requested boolean subset is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_FLUKE_IMPORT_ERROR: ImportError | None = None
try:  # Keep the package root importable without fds[fluke].
    from fluke.data import DataSplitter, DummyDataContainer, FastDataLoader
except ImportError as exc:  # pragma: no cover - exercised in minimal installations
    DataSplitter = object  # type: ignore[assignment,misc]
    DummyDataContainer = None  # type: ignore[assignment,misc]
    FastDataLoader = object  # type: ignore[assignment,misc]
    _FLUKE_IMPORT_ERROR = exc

from data_loading.dataset import BaseDataset


def _require_fluke_data() -> None:
    if _FLUKE_IMPORT_ERROR is not None:
        raise RuntimeError("Fluke support requires `pip install -e '.[fluke]'`.") from _FLUKE_IMPORT_ERROR


@dataclass(frozen=True)
class ImageSample:
    """A path-backed classification sample with a canonical integer label."""

    image: Path
    label: int


@dataclass(frozen=True)
class TinyImageNetPartition:
    """Resolved Tiny-ImageNet partition and canonical global validation samples."""

    path: Path
    sha256: str
    client_samples: tuple[tuple[ImageSample, ...], ...]
    global_validation: tuple[ImageSample, ...]
    num_classes: int = 200

    @property
    def n_clients(self) -> int:
        return len(self.client_samples)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tiny_root(dataset_root: Path) -> Path:
    root = Path(dataset_root).expanduser()
    # Callers that load a partition normalize this once up front. Avoid resolving
    # it again for every one of Tiny-ImageNet's 100,000 path-backed samples.
    if not root.is_absolute():
        root = root.resolve()
    return root if root.name == "tiny-imagenet-200" else root / "tiny-imagenet-200"


def _tiny_label_map(dataset_root: Path) -> dict[str, int]:
    train_root = _tiny_root(dataset_root) / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Tiny-ImageNet training directory does not exist: {train_root}")
    classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No Tiny-ImageNet classes found under {train_root}")
    return {label: index for index, label in enumerate(classes)}


def rebase_tiny_imagenet_path(raw_path: str | os.PathLike[str], dataset_root: Path) -> Path:
    """Rebase an existing split path through the ``tiny-imagenet-200`` suffix."""

    parts = PurePath(str(raw_path).replace("\\", "/")).parts
    try:
        suffix_index = parts.index("tiny-imagenet-200")
    except ValueError as exc:
        raise ValueError(
            f"Partition image path does not contain the Tiny-ImageNet suffix: {raw_path!r}"
        ) from exc
    relative_parts = parts[suffix_index + 1 :]
    if any(part == ".." for part in relative_parts):
        raise ValueError(f"Partition image path escapes the Tiny-ImageNet root: {raw_path!r}")
    # ``dataset_root`` is absolute for normal partition loading. Keep this lexical
    # construction path-backed; `_validate_samples` performs the one required
    # existence check without an extra network-filesystem realpath round trip.
    return _tiny_root(dataset_root).joinpath(*relative_parts)


def _parse_client_index(client_id: str) -> int:
    text = str(client_id)
    match = re.fullmatch(r"(?:client[_-]?)?(\d+)", text)
    if match is None:
        raise ValueError(
            "Tiny-ImageNet client identifiers must be contiguous numeric IDs or client_<ID>; "
            f"got {client_id!r}."
        )
    return int(match.group(1))


def _canonical_client_lists(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    clients = payload.get("clients")
    if not isinstance(clients, Mapping) or not clients:
        raise ValueError("Partition JSON must contain a non-empty 'clients' mapping.")
    ordered = sorted(
        ((_parse_client_index(key), str(key), value) for key, value in clients.items()),
        key=lambda item: item[0],
    )
    indices = [item[0] for item in ordered]
    expected = list(range(len(ordered)))
    if indices != expected:
        raise ValueError(
            f"Partition client IDs must be contiguous from 0; found {indices}, expected {expected}."
        )
    output: list[tuple[str, Mapping[str, Any]]] = []
    for _, client_id, client_payload in ordered:
        if not isinstance(client_payload, Mapping):
            raise TypeError(f"Client {client_id!r} payload must be a mapping.")
        output.append((client_id, client_payload))
    return output


def _coerce_label(value: Any, label_map: Mapping[str, int]) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean labels are invalid for Tiny-ImageNet.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    if text not in label_map:
        raise KeyError(f"Unknown Tiny-ImageNet label {value!r}.")
    return int(label_map[text])


def _coerce_sample(raw: Any, *, dataset_root: Path, label_map: Mapping[str, int]) -> ImageSample:
    if isinstance(raw, Mapping):
        image = raw.get("image", raw.get("path"))
        label = raw.get("label")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        image, label = raw
    else:
        raise TypeError("Partition samples must be {'image', 'label'} mappings or two-item sequences.")
    if image is None or label is None:
        raise ValueError(f"Partition sample requires image and label fields: {raw!r}")
    return ImageSample(
        image=rebase_tiny_imagenet_path(str(image), dataset_root),
        label=_coerce_label(label, label_map),
    )


def _validate_samples(
    samples: Iterable[ImageSample], *, num_classes: int, verify_images: bool
) -> None:
    seen: set[Path] = set()
    for sample in samples:
        if not 0 <= sample.label < num_classes:
            raise ValueError(
                f"Tiny-ImageNet label {sample.label} is outside [0, {num_classes - 1}] for "
                f"{sample.image}."
            )
        if not sample.image.is_file():
            raise FileNotFoundError(f"Partition image does not exist: {sample.image}")
        if verify_images and sample.image not in seen:
            try:
                with Image.open(sample.image) as image:
                    image.verify()
            except Exception as exc:  # noqa: BLE001 - retain path in preflight error
                raise ValueError(f"Unreadable Tiny-ImageNet image: {sample.image}") from exc
            seen.add(sample.image)


def _validate_train_sample_labels(
    samples: Iterable[ImageSample], *, label_map: Mapping[str, int]
) -> None:
    """Ensure a partition label agrees with its canonical Tiny-ImageNet class directory."""

    for sample in samples:
        try:
            dataset_index = sample.image.parts.index("tiny-imagenet-200")
            train_index = sample.image.parts.index("train", dataset_index + 1)
            class_name = sample.image.parts[train_index + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "Tiny-ImageNet train samples must be under train/<class>/images: "
                f"{sample.image}"
            ) from exc
        expected_label = label_map.get(class_name)
        if expected_label is None:
            raise ValueError(
                f"Tiny-ImageNet train sample has an unknown class directory {class_name!r}: "
                f"{sample.image}"
            )
        if sample.label != expected_label:
            raise ValueError(
                f"Partition label {sample.label} disagrees with class {class_name!r} "
                f"({expected_label}) for {sample.image}."
            )


def canonical_tiny_imagenet_validation(dataset_root: Path) -> tuple[ImageSample, ...]:
    """Load Tiny-ImageNet's canonical validation split, never a split JSON global_val."""

    label_map = _tiny_label_map(dataset_root)
    tiny_root = _tiny_root(dataset_root)
    dataset = BaseDataset("tiny_imagenet", str(tiny_root.parent), "val")
    samples = tuple(
        ImageSample(image=Path(path), label=_coerce_label(label, label_map))
        for path, label in zip(dataset.data, dataset.labels)
    )
    if not samples:
        raise ValueError("Tiny-ImageNet canonical validation split is empty.")
    return samples


def load_tiny_imagenet_partition(
    partition_path: str | os.PathLike[str],
    *,
    dataset_root: str | os.PathLike[str],
    expected_clients: int | None = None,
    verify_images: bool = True,
) -> TinyImageNetPartition:
    """Load one existing split JSON and enforce the adapter's path/label invariants."""

    path = Path(partition_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Partition JSON does not exist: {path}")
    root = Path(dataset_root).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Partition JSON root must be a mapping.")

    label_map = _tiny_label_map(root)
    num_classes = len(label_map)
    clients: list[tuple[ImageSample, ...]] = []
    assigned_train_samples: dict[Path, str] = {}
    for client_id, client_payload in _canonical_client_lists(payload):
        train = client_payload.get("train")
        if not isinstance(train, list) or not train:
            raise ValueError(f"Client {client_id!r} requires a non-empty train list.")
        client_samples = tuple(
            _coerce_sample(sample, dataset_root=root, label_map=label_map) for sample in train
        )
        _validate_samples(client_samples, num_classes=num_classes, verify_images=verify_images)
        _validate_train_sample_labels(client_samples, label_map=label_map)
        for sample in client_samples:
            other_client = assigned_train_samples.get(sample.image)
            if other_client is not None:
                raise ValueError(
                    f"Partition image is assigned to both {other_client!r} and {client_id!r}: "
                    f"{sample.image}"
                )
            assigned_train_samples[sample.image] = client_id
        clients.append(client_samples)

    if expected_clients is not None and len(clients) != expected_clients:
        raise ValueError(
            f"Partition has {len(clients)} clients but protocol requests {expected_clients}."
        )

    global_validation = canonical_tiny_imagenet_validation(root)
    _validate_samples(global_validation, num_classes=num_classes, verify_images=verify_images)
    return TinyImageNetPartition(
        path=path,
        sha256=_file_sha256(path),
        client_samples=tuple(clients),
        global_validation=global_validation,
        num_classes=num_classes,
    )


def build_tiny_imagenet_transforms(
    *, image_size: int, profile: str, train: bool
):
    """Build Tiny-ImageNet transforms without needlessly upscaling native inputs.

    Tiny-ImageNet images are already 64x64.  A shallow, randomly initialized CNN
    should train at that native resolution; padding before the random crop retains
    spatial augmentation without inventing pixels through interpolation.  Larger
    requested resolutions retain the ImageNet-style resize-and-crop path needed by
    pretrained backbones.
    """

    if image_size <= 0:
        raise ValueError("image_size must be > 0")
    if image_size == 64:
        operations: list[Any] = []
        if train:
            operations.extend(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            )
    else:
        resize_size = max(image_size + 32, int(round(image_size * 1.14)))
        operations = [transforms.Resize(resize_size)]
        if train:
            operations.extend(
                [
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            )
        else:
            operations.append(transforms.CenterCrop(image_size))
    operations.append(transforms.ToTensor())
    if profile == "imagenet":
        operations.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    elif profile != "unit_interval":
        raise ValueError(f"Unsupported transform profile: {profile}")
    return transforms.Compose(operations)


class LazyImageTensor:
    """Tensor-shaped, path-backed image collection used by ``LazyFastDataLoader``.

    Long-index selection returns another lazy view, which lets Fluke's shuffle and
    percentage logic operate without decoding the full client dataset.  Batch slices
    and boolean class masks deliberately materialize only the selected image subset.
    """

    def __init__(
        self,
        samples: Sequence[ImageSample],
        transform: Any,
        image_size: int,
        indices: torch.Tensor | None = None,
    ):
        self._samples = tuple(samples)
        self._transform = transform
        self._image_size = int(image_size)
        self._indices = (
            torch.arange(len(self._samples), dtype=torch.long)
            if indices is None
            else indices.detach().cpu().to(dtype=torch.long).clone()
        )

    @property
    def shape(self) -> torch.Size:
        return torch.Size((len(self), 3, self._image_size, self._image_size))

    @property
    def ndim(self) -> int:
        return 4

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def __len__(self) -> int:
        return int(self._indices.numel())

    @property
    def samples(self) -> tuple[ImageSample, ...]:
        return tuple(self._samples[int(index)] for index in self._indices.tolist())

    def clone(self) -> "LazyImageTensor":
        return LazyImageTensor(self._samples, self._transform, self._image_size, self._indices)

    def _decode(self, local_index: int) -> torch.Tensor:
        if local_index < 0:
            local_index += len(self)
        if not 0 <= local_index < len(self):
            raise IndexError("Lazy image index out of bounds.")
        sample = self._samples[int(self._indices[local_index])]
        with Image.open(sample.image) as image:
            image = image.convert("RGB")
            return self._transform(image) if self._transform is not None else transforms.ToTensor()(image)

    def _materialize_positions(self, positions: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().to(dtype=torch.long).tolist()
        decoded = [self._decode(int(position)) for position in positions]
        if not decoded:
            return torch.empty((0, 3, self._image_size, self._image_size), dtype=torch.float32)
        return torch.stack(decoded, dim=0)

    def _lazy_view(self, positions: torch.Tensor) -> "LazyImageTensor":
        if positions.dtype == torch.bool:
            positions = torch.nonzero(positions, as_tuple=False).reshape(-1)
        positions = positions.detach().cpu().to(dtype=torch.long)
        if positions.numel() and (positions.min() < 0 or positions.max() >= len(self)):
            raise IndexError("Lazy image index out of bounds.")
        return LazyImageTensor(
            self._samples,
            self._transform,
            self._image_size,
            self._indices[positions],
        )

    def __getitem__(self, index: Any) -> torch.Tensor | "LazyImageTensor":
        if isinstance(index, (int, np.integer)):
            return self._decode(int(index))
        if isinstance(index, torch.Tensor) and index.ndim == 0:
            return self._decode(int(index.item()))
        if isinstance(index, slice):
            return self._materialize_positions(torch.arange(len(self), dtype=torch.long)[index])
        if isinstance(index, np.ndarray):
            index = torch.from_numpy(index)
        if isinstance(index, (list, tuple)):
            index = torch.as_tensor(index)
        if not isinstance(index, torch.Tensor):
            raise TypeError(f"Unsupported lazy image index type: {type(index).__name__}")
        if index.dtype == torch.bool:
            if index.ndim != 1 or index.numel() != len(self):
                raise IndexError("Boolean image index must be one-dimensional and match the loader size.")
            return self._materialize_positions(torch.nonzero(index, as_tuple=False).reshape(-1))
        if not index.dtype.is_floating_point and index.dtype != torch.bool:
            return self._lazy_view(index.reshape(-1))
        raise TypeError("Lazy image tensor supports integer or boolean tensor indexing only.")

    def __deepcopy__(self, memo: dict[int, Any]) -> "LazyImageTensor":
        return self.clone()


class _LazyPathDataset(Dataset):
    def __init__(self, images: LazyImageTensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class LazyFastDataLoader(FastDataLoader):
    """A Fluke ``FastDataLoader`` that decodes only the current image batch."""

    def __init__(
        self,
        samples: Sequence[ImageSample],
        *,
        num_labels: int,
        batch_size: int = 32,
        shuffle: bool = False,
        transform: Any = None,
        image_size: int = 64,
        percentage: float = 1.0,
        skip_singleton: bool = True,
        single_batch: bool = False,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ):
        _require_fluke_data()
        if not samples:
            raise ValueError("LazyFastDataLoader requires at least one sample.")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self.tensors: list[Any] = [
            LazyImageTensor(samples, transform, image_size),
            torch.tensor([sample.label for sample in samples], dtype=torch.long),
        ]
        self.num_labels = int(num_labels)
        self.max_size = len(samples)
        self.percentage = float(percentage)
        self.size = 0
        self.shuffle = bool(shuffle)
        self.skip_singleton = bool(skip_singleton)
        self.single_batch = bool(single_batch)
        self.num_workers = int(num_workers)
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.transforms = None
        self._iter_index = 0
        self._batch_iterator: Any | None = None
        self.set_sample_size(percentage)
        self.batch_size = batch_size if batch_size > 0 else self.size

    @classmethod
    def _from_tensors(
        cls,
        images: LazyImageTensor,
        labels: torch.Tensor,
        *,
        num_labels: int,
        batch_size: int,
        shuffle: bool,
        percentage: float,
        skip_singleton: bool,
        single_batch: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> "LazyFastDataLoader":
        instance = object.__new__(cls)
        instance.tensors = [images, labels]
        instance.num_labels = int(num_labels)
        instance.max_size = int(images.shape[0])
        instance.percentage = 1.0
        instance.size = 0
        instance.shuffle = bool(shuffle)
        instance.skip_singleton = bool(skip_singleton)
        instance.single_batch = bool(single_batch)
        instance.num_workers = int(num_workers)
        instance.pin_memory = bool(pin_memory)
        instance.transforms = None
        instance._iter_index = 0
        instance._batch_iterator = None
        instance.set_sample_size(percentage)
        instance.batch_size = batch_size if batch_size > 0 else instance.size
        return instance

    def set_sample_size(self, percentage: float) -> int:
        if not 0 < percentage <= 1:
            raise ValueError("percentage must be in (0, 1]")
        self.percentage = float(percentage)
        self.size = max(int(self.max_size * self.percentage), 1)
        if self.size < self.max_size:
            permutation = torch.randperm(self.max_size)
            self.tensors = [tensor[permutation] for tensor in self.tensors]
        return self.size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_size must be > 0")
        self._batch_size = int(value)
        self.n_batches = (self.size + self._batch_size - 1) // self._batch_size

    def _active_tensors(self) -> tuple[LazyImageTensor, torch.Tensor]:
        """Return the current percentage view without decoding any images."""

        positions = torch.arange(self.size, dtype=torch.long)
        images = self.tensors[0][positions]
        if not isinstance(images, LazyImageTensor):  # Defensive guard for Fluke extensions.
            raise TypeError("LazyFastDataLoader image selection must remain path-backed.")
        return images, self.tensors[1][: self.size]

    def __iter__(self) -> "LazyFastDataLoader":
        self._iter_index = 0
        self._batch_iterator = None
        if self.num_workers > 0:
            images, labels = self._active_tensors()
            self._batch_iterator = iter(
                DataLoader(
                    _LazyPathDataset(images, labels),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=False,
                )
            )
            return self
        if self.shuffle:
            permutation = torch.randperm(self.size)
            self.tensors = [tensor[permutation] for tensor in self.tensors]
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._batch_iterator is not None:
            if self.single_batch and self._iter_index > 0:
                raise StopIteration
            images, labels = next(self._batch_iterator)
            self._iter_index += int(images.shape[0])
            if self.skip_singleton and images.shape[0] == 1:
                raise StopIteration
            return images, labels
        if self.single_batch and self._iter_index > 0:
            raise StopIteration
        if self._iter_index >= self.size:
            raise StopIteration
        stop = min(self._iter_index + self.batch_size, self.size)
        images = self.tensors[0][slice(self._iter_index, stop)]
        labels = self.tensors[1][self._iter_index:stop]
        self._iter_index = stop
        if self.skip_singleton and images.shape[0] == 1:
            raise StopIteration
        return images, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not -self.size <= index < self.size:
            raise IndexError("Index out of bounds.")
        return self.tensors[0][index], self.tensors[1][index]

    def as_dataloader(self, **kwargs: Any) -> DataLoader:
        kwargs.setdefault("num_workers", self.num_workers)
        kwargs.setdefault("pin_memory", self.pin_memory)
        images, labels = self._active_tensors()
        return DataLoader(
            _LazyPathDataset(images, labels),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=False,
            **kwargs,
        )

    def clone(self) -> "LazyFastDataLoader":
        return self._from_tensors(
            self.tensors[0].clone(),
            self.tensors[1].clone(),
            num_labels=self.num_labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            percentage=self.percentage,
            skip_singleton=self.skip_singleton,
            single_batch=self.single_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def __len__(self) -> int:
        return self.n_batches

    def __repr__(self) -> str:
        return (
            f"LazyFastDataLoader(num_labels={self.num_labels}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, size={self.size})"
        )


class PartitionDataSplitter(DataSplitter):
    """Fluke splitter adapter that preserves an existing partition verbatim."""

    def __init__(
        self,
        partition: TinyImageNetPartition,
        *,
        train_transform: Any,
        eval_transform: Any,
        image_size: int,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ):
        _require_fluke_data()
        self.partition = partition
        self._train_transform = train_transform
        self._eval_transform = eval_transform
        self._image_size = int(image_size)
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self._num_workers = int(num_workers)
        self._pin_memory = pin_memory
        # ``DataSplitter`` only needs a DataContainer for its default assign path;
        # this subclass returns fresh lazy loaders directly.
        super().__init__(
            dataset=DummyDataContainer([], [], None, num_classes=partition.num_classes),
            server_test=True,
        )

    def _loader(
        self,
        samples: Sequence[ImageSample],
        *,
        batch_size: int,
        shuffle: bool,
        train: bool,
    ) -> LazyFastDataLoader:
        return LazyFastDataLoader(
            samples,
            num_labels=self.partition.num_classes,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=self._train_transform if train else self._eval_transform,
            image_size=self._image_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def assign(
        self, n_clients: int, batch_size: int = 32
    ) -> tuple[tuple[list[LazyFastDataLoader], list[LazyFastDataLoader]], LazyFastDataLoader]:
        if n_clients != self.partition.n_clients:
            raise ValueError(
                f"Partition contains {self.partition.n_clients} clients, requested {n_clients}."
            )
        clients_train = [
            self._loader(samples, batch_size=batch_size, shuffle=True, train=True)
            for samples in self.partition.client_samples
        ]
        # Every personalized client deliberately receives the same canonical global
        # validation set.  The runner labels resulting metrics accordingly.
        clients_test = [
            self._loader(
                self.partition.global_validation,
                batch_size=max(batch_size, 128),
                shuffle=False,
                train=False,
            )
            for _ in range(n_clients)
        ]
        server_test = self._loader(
            self.partition.global_validation,
            batch_size=max(batch_size, 128),
            shuffle=False,
            train=False,
        )
        return (clients_train, clients_test), server_test
