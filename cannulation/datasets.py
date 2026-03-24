import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


@dataclass
class DatasetInfo:
    source: str
    dataset_type: str
    task_type: str
    input_dim: Optional[int] = None
    input_shape: Optional[tuple[int, ...]] = None
    num_classes: Optional[int] = None
    target_col: Optional[str] = None
    class_names: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetLoader:
    def load(
        self,
        source: Optional[str],
        target_col: Optional[str],
        val_split: float,
        batch_size: int,
        data_dir: str = "./data",
        image_size: int = 28,
    ) -> tuple[DataLoader, DataLoader, DatasetInfo]:
        if not 0.0 < val_split < 0.5:
            raise ValueError("val_split must be between 0 and 0.5")

        if not source or source.strip().lower() == "mnist":
            return self._load_mnist(batch_size=batch_size, data_dir=data_dir)

        source_path = Path(source).expanduser()
        if source_path.is_file() and source_path.suffix.lower() == ".csv":
            return self._load_csv(source_path, target_col, val_split, batch_size)

        if source_path.is_dir():
            return self._load_image_folder(source_path, val_split, batch_size, image_size)

        return self._load_huggingface(source, target_col, val_split, batch_size)

    def _load_mnist(self, batch_size: int, data_dir: str) -> tuple[DataLoader, DataLoader, DatasetInfo]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size),
            DatasetInfo(
                source="mnist",
                dataset_type="image",
                task_type="classification",
                input_shape=(1, 28, 28),
                num_classes=10,
                class_names=[str(idx) for idx in range(10)],
            ),
        )

    def _load_csv(
        self,
        csv_path: Path,
        target_col: Optional[str],
        val_split: float,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, DatasetInfo]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            raise ValueError(f"CSV file is empty: {csv_path}")
        if not target_col:
            raise ValueError("CSV datasets require a target column")
        if target_col not in fieldnames:
            raise ValueError(f"Target column '{target_col}' not found in {csv_path.name}")

        return self._build_tabular_loaders_from_rows(
            rows=rows,
            source=str(csv_path),
            target_col=target_col,
            batch_size=batch_size,
            val_split=val_split,
        )

    def _load_image_folder(
        self,
        root: Path,
        val_split: float,
        batch_size: int,
        image_size: int,
    ) -> tuple[DataLoader, DataLoader, DatasetInfo]:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(str(root), transform=transform)
        if len(dataset) < 2:
            raise ValueError("Image folder dataset must contain at least two images")
        if len(dataset.classes) < 2:
            raise ValueError("Image folder dataset must contain at least two class directories")

        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=42,
            stratify=dataset.targets,
        )
        sample, _ = dataset[0]
        info = DatasetInfo(
            source=str(root),
            dataset_type="image",
            task_type="classification",
            input_shape=tuple(sample.shape),
            num_classes=len(dataset.classes),
            class_names=list(dataset.classes),
        )
        return (
            DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(dataset, val_idx), batch_size=batch_size),
            info,
        )

    def _load_huggingface(
        self,
        dataset_name: str,
        target_col: Optional[str],
        val_split: float,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, DatasetInfo]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ValueError(
                "HuggingFace dataset support requires the 'datasets' package"
            ) from exc

        dataset_dict = load_dataset(dataset_name)
        if "train" in dataset_dict:
            train_split = dataset_dict["train"]
            if "validation" in dataset_dict:
                val_split_data = dataset_dict["validation"]
            elif "test" in dataset_dict:
                val_split_data = dataset_dict["test"]
            else:
                split = train_split.train_test_split(test_size=val_split, seed=42)
                train_split = split["train"]
                val_split_data = split["test"]
        else:
            first_split = next(iter(dataset_dict.keys()))
            split = dataset_dict[first_split].train_test_split(test_size=val_split, seed=42)
            train_split = split["train"]
            val_split_data = split["test"]

        rows = train_split.to_list() + val_split_data.to_list()
        if not rows:
            raise ValueError(f"HuggingFace dataset '{dataset_name}' returned no rows")

        resolved_target = target_col or self._guess_target_col(rows[0])
        if not resolved_target:
            raise ValueError("Could not infer target column for HuggingFace dataset")

        return self._build_tabular_loaders_from_rows(
            rows=rows,
            source=dataset_name,
            target_col=resolved_target,
            batch_size=batch_size,
            val_split=val_split,
        )

    def _build_tabular_loaders_from_rows(
        self,
        rows: list[dict[str, Any]],
        source: str,
        target_col: str,
        batch_size: int,
        val_split: float,
    ) -> tuple[DataLoader, DataLoader, DatasetInfo]:
        if target_col not in rows[0]:
            raise ValueError(f"Target column '{target_col}' not present in dataset")

        feature_cols = [col for col in rows[0].keys() if col != target_col]
        if not feature_cols:
            raise ValueError("Tabular datasets need at least one feature column")

        y_raw = [str(row[target_col]) for row in rows]
        task_type = self._infer_task_type(y_raw)
        if task_type != "classification":
            raise ValueError("Regression datasets are not implemented yet")

        numeric_cols = [col for col in feature_cols if self._is_numeric_column(row.get(col) for row in rows)]
        categorical_cols = [col for col in feature_cols if col not in numeric_cols]

        labels = LabelEncoder().fit_transform(y_raw)
        stratify = labels if len(np.unique(labels)) > 1 else None
        train_rows, val_rows, y_train, y_val = train_test_split(
            rows,
            labels,
            test_size=val_split,
            random_state=42,
            stratify=stratify,
        )

        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []

        if numeric_cols:
            train_num = self._extract_numeric_matrix(train_rows, numeric_cols)
            val_num = self._extract_numeric_matrix(val_rows, numeric_cols)
            medians = np.nanmedian(train_num, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            train_num = np.where(np.isnan(train_num), medians, train_num)
            val_num = np.where(np.isnan(val_num), medians, val_num)
            scaler = StandardScaler()
            train_parts.append(scaler.fit_transform(train_num).astype(np.float32))
            val_parts.append(scaler.transform(val_num).astype(np.float32))

        if categorical_cols:
            train_cat = self._extract_categorical_matrix(train_rows, categorical_cols)
            val_cat = self._extract_categorical_matrix(val_rows, categorical_cols)
            encoder_kwargs = {"handle_unknown": "ignore"}
            try:
                encoder = OneHotEncoder(sparse_output=False, **encoder_kwargs)
            except TypeError:
                encoder = OneHotEncoder(sparse=False, **encoder_kwargs)
            train_parts.append(encoder.fit_transform(train_cat).astype(np.float32))
            val_parts.append(encoder.transform(val_cat).astype(np.float32))

        if not train_parts:
            raise ValueError("Dataset has no usable feature columns")

        x_train = np.concatenate(train_parts, axis=1)
        x_val = np.concatenate(val_parts, axis=1)
        info = DatasetInfo(
            source=source,
            dataset_type="tabular",
            task_type="classification",
            input_dim=int(x_train.shape[1]),
            num_classes=int(len(np.unique(labels))),
            target_col=target_col,
            class_names=[str(label) for label in sorted({value for value in y_raw})],
        )
        return (
            DataLoader(
                TensorDataset(
                    torch.from_numpy(x_train),
                    torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
                ),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                TensorDataset(
                    torch.from_numpy(x_val),
                    torch.from_numpy(np.asarray(y_val, dtype=np.int64)),
                ),
                batch_size=batch_size,
            ),
            info,
        )

    @staticmethod
    def _guess_target_col(sample_row: dict[str, Any]) -> Optional[str]:
        for candidate in ("label", "labels", "target", "y", "class"):
            if candidate in sample_row:
                return candidate
        return None

    @staticmethod
    def _infer_task_type(values: list[str]) -> str:
        unique_count = len(set(values))
        if unique_count <= max(20, int(len(values) ** 0.5)):
            return "classification"
        try:
            numeric_values = [float(value) for value in values]
        except (TypeError, ValueError):
            return "classification"
        return "regression" if len(set(numeric_values)) > unique_count * 0.9 else "classification"

    @staticmethod
    def _is_numeric_column(values) -> bool:
        saw_value = False
        for value in values:
            if value in (None, ""):
                continue
            saw_value = True
            try:
                float(value)
            except (TypeError, ValueError):
                return False
        return saw_value

    @staticmethod
    def _extract_numeric_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
        matrix = []
        for row in rows:
            matrix.append([
                float(row[col]) if row.get(col) not in (None, "") else np.nan
                for col in columns
            ])
        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def _extract_categorical_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
        matrix = []
        for row in rows:
            matrix.append([
                str(row.get(col)).strip() if row.get(col) not in (None, "") else "__missing__"
                for col in columns
            ])
        return np.asarray(matrix, dtype=object)