"""Proyecto 2: Comparativa de reducciones de dimensionalidad y clasificadores.

Este módulo descarga la data de Fashion-MNIST, aplica diversas técnicas de
reducción de dimensionalidad, optimiza varios modelos de clasificación y
compara su rendimiento en el conjunto de prueba.
"""
from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import Isomap, SpectralEmbedding, TSNE, trustworthiness
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:  # pragma: no cover - dependencia opcional al ejecutar offline
    import gdown
except ImportError:  # pragma: no cover
    gdown = None

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - permitir ejecución parcial
    umap = None


# === Utilidades de descarga y lectura de Fashion-MNIST ===
CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def download_data(file_id: str, file_name: str | Path) -> str:
    """Descarga un archivo desde Google Drive si no existe localmente."""
    file_path = Path(file_name)
    if file_path.exists():
        return str(file_path)
    if gdown is None:
        raise RuntimeError(
            "El archivo requerido no está disponible localmente y no es posible descargarlo "
            "porque gdown no está instalado."
        )
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(file_path), quiet=False)
    return str(file_path)


def read_labels(file_path: str) -> pd.DataFrame:
    """Lee el archivo IDX con las etiquetas y devuelve un DataFrame."""
    with open(file_path, "rb") as f:
        _magic, _num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    df = pd.DataFrame(labels, columns=["label"])
    df["class_name"] = df["label"].map(CLASS_NAMES)
    return df


def extract_feature_images(file_path: str) -> np.ndarray:
    """Lee las imágenes en formato IDX y devuelve un array (n_samples, 784)."""
    with open(file_path, "rb") as f:
        _magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        X = images.reshape(num_images, rows * cols)
    return X


def load_raw_data(data_dir: Path | str = Path(".")) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Descarga los archivos necesarios y devuelve los datos crudos."""
    data_dir = Path(data_dir)

    file_train_X = download_data("1enziBIpqiv_t95KQcifsclNH2BdR8lAd", data_dir / "train_X")
    file_test_X = download_data("1Jeax6tnQ6Nmr2PTNXdQqzKnN0YqtrLe4", data_dir / "test_X")
    file_train_Y = download_data("1MZtn2iA5cgiYT1i3O0ECuR01oD0kGHh7", data_dir / "train_Y")
    file_test_Y = download_data("1K5pxwk2s3RDYsYuwv8RftJTXZ-RGR7K4", data_dir / "test_Y")

    train_X = extract_feature_images(str(file_train_X))
    test_X = extract_feature_images(str(file_test_X))
    train_Y = read_labels(str(file_train_Y))["label"]
    test_Y = read_labels(str(file_test_Y))["label"]
    return train_X, test_X, train_Y, test_Y


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def preprocess_data(random_state: int = 42) -> Dataset:
    """Carga los datos crudos y aplica el preprocesamiento requerido."""
    X_train_raw, X_test_raw, y_train_series, y_test_series = load_raw_data()

    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train_series.to_numpy()
    y_test = y_test_series.to_numpy()

    rng = np.random.default_rng(random_state)
    permutation_train = rng.permutation(len(X_train_scaled))
    permutation_test = rng.permutation(len(X_test_scaled))
    X_train_scaled = X_train_scaled[permutation_train]
    y_train = y_train[permutation_train]
    X_test_scaled = X_test_scaled[permutation_test]
    y_test = y_test[permutation_test]

    return Dataset(X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test)


# === Clases de reducción de dimensionalidad ===
class BaseReduction:
    """Clase base para las técnicas de reducción de dimensionalidad."""

    quality_metric_name: str = ""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        random_state: int = 42,
        optimization_sample_size: int = 5000,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.optimization_sample_size = optimization_sample_size

        self.best_k: Optional[int] = None
        self.reducer = None
        self.quality_metric_: Optional[float] = None
        self.metric_history_: Dict[int, float] = {}

    def _get_sample(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if max_samples is None or max_samples >= len(self.X_train):
            return self.X_train, self.y_train
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(self.X_train), size=max_samples, replace=False)
        return self.X_train[indices], self.y_train[indices]

    def visualize(self, sample: int = 500) -> Figure:
        raise NotImplementedError

    def optimize(self, k_grid: Iterable[int] = (8, 16, 32, 64, 128)) -> int:
        raise NotImplementedError

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # Utilidad común para gráficos
    def _plot_embedding(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str,
        filename: Path,
    ) -> Figure:
        os.makedirs(filename.parent, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        legend = ax.legend(
            *scatter.legend_elements(num=len(CLASS_NAMES)),
            title="Clases",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        ax.add_artist(legend)
        ax.set_title(title)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        return fig


class NoReduction(BaseReduction):
    quality_metric_name = "Sin pérdida"

    def visualize(self, sample: int = 500) -> Figure:
        sample_X, sample_y = self._get_sample(min(sample, len(self.X_train)))
        reducer = PCA(n_components=2, random_state=self.random_state)
        embedding = reducer.fit_transform(sample_X)
        return self._plot_embedding(
            embedding,
            sample_y,
            "Visualización 2D (PCA auxiliar) - Sin reducción",
            Path("figures/no_reduction_2d.png"),
        )

    def optimize(self, k_grid: Iterable[int] = (8, 16, 32, 64, 128)) -> int:
        self.best_k = self.X_train.shape[1]
        self.quality_metric_ = 0.0
        self.metric_history_[self.best_k] = self.quality_metric_
        return self.best_k

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_train, self.X_test


class PCA_Reduction(BaseReduction):
    quality_metric_name = "1 - Varianza explicada acumulada"

    def visualize(self, sample: int = 500) -> Figure:
        sample_X, sample_y = self._get_sample(min(sample, len(self.X_train)))
        reducer = PCA(n_components=2, random_state=self.random_state)
        embedding = reducer.fit_transform(sample_X)
        return self._plot_embedding(
            embedding,
            sample_y,
            "Visualización 2D - PCA",
            Path("figures/pca_2d.png"),
        )

    def optimize(self, k_grid: Iterable[int] = (8, 16, 32, 64, 128)) -> int:
        best_loss = float("inf")
        for k in k_grid:
            pca = PCA(n_components=k, random_state=self.random_state)
            pca.fit(self.X_train)
            cumulative = np.sum(pca.explained_variance_ratio_)
            loss = 1 - cumulative
            self.metric_history_[k] = loss
            if loss < best_loss:
                best_loss = loss
                self.best_k = k
        self.quality_metric_ = best_loss
        return self.best_k

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_k is None:
            raise ValueError("Se debe ejecutar optimize() antes de reduce().")
        self.reducer = PCA(n_components=self.best_k, random_state=self.random_state)
        X_train_red = self.reducer.fit_transform(self.X_train)
        X_test_red = self.reducer.transform(self.X_test)
        return X_train_red, X_test_red


class NMF_Reduction(BaseReduction):
    quality_metric_name = "Error de reconstrucción promedio"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # NMF se beneficia de un tamaño de muestra menor para la optimización
        self.optimization_sample_size = min(self.optimization_sample_size, 3000)

    def visualize(self, sample: int = 500) -> Figure:
        sample_X, sample_y = self._get_sample(min(sample, len(self.X_train)))
        model = NMF(
            n_components=2,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=300,
        )
        W = model.fit_transform(sample_X)
        return self._plot_embedding(
            W,
            sample_y,
            "Visualización 2D - NMF",
            Path("figures/nmf_2d.png"),
        )

    def optimize(self, k_grid: Iterable[int] = (8, 16, 32, 64, 128)) -> int:
        sample_X, _ = self._get_sample(self.optimization_sample_size)
        best_error = float("inf")
        for k in k_grid:
            model = NMF(
                n_components=k,
                init="nndsvda",
                random_state=self.random_state,
                max_iter=400,
            )
            W = model.fit_transform(sample_X)
            H = model.components_
            reconstruction = np.dot(W, H)
            error = np.linalg.norm(sample_X - reconstruction, ord="fro") / sample_X.shape[0]
            self.metric_history_[k] = error
            if error < best_error:
                best_error = error
                self.best_k = k
        self.quality_metric_ = best_error
        return self.best_k

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_k is None:
            raise ValueError("Se debe ejecutar optimize() antes de reduce().")
        self.reducer = NMF(
            n_components=self.best_k,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=400,
        )
        X_train_red = self.reducer.fit_transform(self.X_train)
        X_test_red = self.reducer.transform(self.X_test)
        return X_train_red, X_test_red


class TrustworthinessReduction(BaseReduction):
    """Clase auxiliar para técnicas cuya métrica de calidad es trustworthiness."""

    quality_metric_name = "Trustworthiness"
    neighbors_for_trustworthiness: int = 10

    def _fit_transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        raise NotImplementedError

    def visualize(self, sample: int = 500) -> Figure:
        sample_X, sample_y = self._get_sample(min(sample, len(self.X_train)))
        embedding = self._fit_transform(sample_X, 2)
        return self._plot_embedding(
            embedding,
            sample_y,
            f"Visualización 2D - {self.__class__.__name__.replace('_Reduction', '')}",
            Path(f"figures/{self.__class__.__name__.lower()}_2d.png"),
        )

    def optimize(self, k_grid: Iterable[int] = (8, 16, 32, 64, 128)) -> int:
        sample_X, _ = self._get_sample(self.optimization_sample_size)
        best_score = -float("inf")
        for k in k_grid:
            embedding = self._fit_transform(sample_X, n_components=k)
            score = trustworthiness(
                sample_X,
                embedding,
                n_neighbors=self.neighbors_for_trustworthiness,
            )
            self.metric_history_[k] = score
            if score > best_score:
                best_score = score
                self.best_k = k
        self.quality_metric_ = best_score
        return self.best_k


class TSNE_Reduction(TrustworthinessReduction):
    neighbors_for_trustworthiness = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_sample_size = min(self.optimization_sample_size, 3000)

    def _fit_transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        tsne = TSNE(
            n_components=n_components,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
            perplexity=30,
        )
        return tsne.fit_transform(X)

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_k is None:
            raise ValueError("Se debe ejecutar optimize() antes de reduce().")
        # t-SNE no implementa transform, por lo que se aplica fit_transform en cada split
        tsne_train = TSNE(
            n_components=self.best_k,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
            perplexity=30,
        )
        X_train_red = tsne_train.fit_transform(self.X_train)

        tsne_test = TSNE(
            n_components=self.best_k,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
            perplexity=30,
        )
        X_test_red = tsne_test.fit_transform(self.X_test)
        self.reducer = tsne_train
        return X_train_red, X_test_red


class UMAP_Reduction(TrustworthinessReduction):
    neighbors_for_trustworthiness = 15

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_sample_size = min(self.optimization_sample_size, 5000)
        self.n_neighbors = 15
        self.min_dist = 0.1

    def _fit_transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        if umap is None:
            raise RuntimeError("UMAP no está disponible. Instala 'umap-learn' para utilizar esta reducción.")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
        )
        return reducer.fit_transform(X)

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_k is None:
            raise ValueError("Se debe ejecutar optimize() antes de reduce().")
        if umap is None:
            raise RuntimeError("UMAP no está disponible. Instala 'umap-learn' para utilizar esta reducción.")
        self.reducer = umap.UMAP(
            n_components=self.best_k,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
        )
        X_train_red = self.reducer.fit_transform(self.X_train)
        X_test_red = self.reducer.transform(self.X_test)
        return X_train_red, X_test_red


class SpectralEmbedding_Reduction(TrustworthinessReduction):
    neighbors_for_trustworthiness = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_sample_size = min(self.optimization_sample_size, 4000)
        self.n_neighbors = 15

    def _fit_transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        reducer = SpectralEmbedding(
            n_components=n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
        )
        return reducer.fit_transform(X)

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_k is None:
            raise ValueError("Se debe ejecutar optimize() antes de reduce().")
        self.reducer = SpectralEmbedding(
            n_components=self.best_k,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
        )
        X_train_red = self.reducer.fit_transform(self.X_train)
        X_test_red = self.reducer.fit_transform(self.X_test)
        return X_train_red, X_test_red


class Isomap_Reduction(TrustworthinessReduction):
    neighbors_for_trustworthiness = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimization_sample_size = min(self.optimization_sample_size, 4000)
        self.n_neighbors = 10

    def _fit_transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        reducer = Isomap(
            n_neighbors=self.n_neighbors,
            n_components=n_components,
        )
        return reducer.fit_transform(X)

    def reduce(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.best_k is None:
            raise ValueError("Se debe ejecutar optimize() antes de reduce().")
        self.reducer = Isomap(
            n_neighbors=self.n_neighbors,
            n_components=self.best_k,
        )
        X_train_red = self.reducer.fit_transform(self.X_train)
        X_test_red = self.reducer.transform(self.X_test)
        return X_train_red, X_test_red


# === Modelos de clasificación ===
class BaseModel:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.best_params_: Optional[Dict[str, object]] = None
        self.estimator_ = None

    def get_estimator(self):
        raise NotImplementedError

    def get_param_grid(self) -> Dict[str, Iterable]:
        raise NotImplementedError

    def optimize(self, scoring: str = "f1_macro", cv: int = 5) -> Dict[str, object]:
        estimator = self.get_estimator()
        param_grid = self.get_param_grid()
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=splitter,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(self.X_train, self.y_train)
        self.best_params_ = grid.best_params_
        self.estimator_ = grid.best_estimator_
        return self.best_params_

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.estimator_ is None:
            raise ValueError("Se debe ejecutar optimize() antes de predict().")
        return self.estimator_.predict(X_test)


class LogisticRegressionModel(BaseModel):
    def get_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=200,
            random_state=self.random_state,
        )

    def get_param_grid(self) -> Dict[str, Iterable]:
        return {"C": [0.01, 0.1, 1, 10]}


class KNNModel(BaseModel):
    def get_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier()

    def get_param_grid(self) -> Dict[str, Iterable]:
        return {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
        }


class SVMModel(BaseModel):
    def get_estimator(self) -> SVC:
        return SVC(kernel="rbf")

    def get_param_grid(self) -> Dict[str, Iterable]:
        return {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        }


class RandomForestModel(BaseModel):
    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

    def get_param_grid(self) -> Dict[str, Iterable]:
        return {
            "n_estimators": [200, 400],
            "max_depth": [None, 20, 40],
            "max_features": ["sqrt", "log2"],
        }


# === Ejecución del experimento ===
def evaluate_model(
    model_cls: type[BaseModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Tuple[Dict[str, object], Dict[str, float], float, float]:
    model = model_cls(X_train, y_train, random_state=random_state)
    start_fit = time.time()
    best_params = model.optimize(scoring="f1_macro", cv=5)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    return best_params, metrics, fit_time, pred_time


class ExperimentRunner:
    def __init__(
        self,
        dataset: Dataset,
        random_state: int = 42,
        reduction_k_grid: Iterable[int] = (8, 16, 32, 64, 128),
        figures_dir: Path = Path("figures"),
        results_dir: Path = Path("results"),
        sample_train_size: Optional[int] = None,
        sample_test_size: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.reduction_k_grid = tuple(reduction_k_grid)
        self.figures_dir = figures_dir
        self.results_dir = results_dir
        self.dataset = self._maybe_sample_dataset(dataset, sample_train_size, sample_test_size)

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.reduction_classes: Dict[str, Tuple[type[BaseReduction], Dict[str, object]]] = {
            "Sin reducción": (NoReduction, {}),
            "PCA": (PCA_Reduction, {}),
            "NMF": (NMF_Reduction, {}),
            "t-SNE": (TSNE_Reduction, {}),
            "UMAP": (UMAP_Reduction, {}),
            "Spectral": (SpectralEmbedding_Reduction, {}),
            "Isomap": (Isomap_Reduction, {}),
        }
        if umap is None:
            self.reduction_classes.pop("UMAP")

        self.model_classes: Dict[str, type[BaseModel]] = {
            "Logistic Regression": LogisticRegressionModel,
            "K-Nearest Neighbors": KNNModel,
            "Support Vector Machine": SVMModel,
            "Random Forest": RandomForestModel,
        }

    def _maybe_sample_dataset(
        self,
        dataset: Dataset,
        sample_train_size: Optional[int],
        sample_test_size: Optional[int],
    ) -> Dataset:
        rng = np.random.default_rng(self.random_state)

        X_train, y_train = dataset.X_train, dataset.y_train
        if sample_train_size is not None and sample_train_size < len(X_train):
            idx = rng.choice(len(X_train), size=sample_train_size, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        X_test, y_test = dataset.X_test, dataset.y_test
        if sample_test_size is not None and sample_test_size < len(X_test):
            idx = rng.choice(len(X_test), size=sample_test_size, replace=False)
            X_test = X_test[idx]
            y_test = y_test[idx]

        return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def run(self) -> pd.DataFrame:
        results: List[Dict[str, object]] = []

        for reduction_name, (reduction_cls, kwargs) in self.reduction_classes.items():
            reduction = reduction_cls(
                self.dataset.X_train,
                self.dataset.y_train,
                self.dataset.X_test,
                self.dataset.y_test,
                random_state=self.random_state,
                **kwargs,
            )
            reduction.visualize(sample=500)
            best_k = reduction.optimize(self.reduction_k_grid)
            X_train_red, X_test_red = reduction.reduce()

            for model_name, model_cls in self.model_classes.items():
                best_params, metrics, fit_time, pred_time = evaluate_model(
                    model_cls,
                    X_train_red,
                    self.dataset.y_train,
                    X_test_red,
                    self.dataset.y_test,
                    random_state=self.random_state,
                )
                results.append(
                    {
                        "reduction": reduction_name,
                        "n_components_opt": int(best_k),
                        "quality_metric": float(reduction.quality_metric_ or 0.0),
                        "model": model_name,
                        "best_params": json.dumps(best_params),
                        "accuracy": metrics["accuracy"],
                        "precision_macro": metrics["precision_macro"],
                        "recall_macro": metrics["recall_macro"],
                        "f1_macro": metrics["f1_macro"],
                        "fit_time_s": fit_time,
                        "pred_time_s": pred_time,
                    }
                )

        df_results = pd.DataFrame(results)
        df_results.sort_values(by="f1_macro", ascending=False, inplace=True)

        csv_path = self.results_dir / "dimensionality_reduction_classification_results.csv"
        df_results.to_csv(csv_path, index=False)

        self._save_f1_bar_chart(df_results)
        self._save_summary_markdown(df_results, csv_path)

        top5 = df_results.head(5)
        print("Top-5 combinaciones (ordenadas por F1 macro):")
        print(top5[["reduction", "model", "n_components_opt", "f1_macro"]])

        return df_results

    def _save_f1_bar_chart(self, df_results: pd.DataFrame) -> None:
        labels = [
            f"{row['reduction']}+{row['model']}(k={row['n_components_opt']})"
            for _, row in df_results.iterrows()
        ]
        scores = df_results["f1_macro"].to_numpy()

        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.3)))
        bars = ax.barh(range(len(labels)), scores, color="steelblue")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("F1 macro")
        ax.set_title("Comparativa de combinaciones Reducción + Modelo")
        ax.invert_yaxis()
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.3f}")
        fig.tight_layout()
        fig.savefig(self.figures_dir / "f1_macro_comparison.png")
        plt.close(fig)

    def _save_summary_markdown(self, df_results: pd.DataFrame, csv_path: Path) -> None:
        best_row = df_results.iloc[0]
        summary_lines = [
            "# Resumen del experimento",
            "",
            "## Datos y preprocesamiento",
            "- Dataset: Fashion-MNIST (imágenes 28x28 aplanadas a 784 pixeles).",
            "- Escalado: valores normalizados en [0, 1] y ajuste MinMaxScaler en el train.",
            "- Semilla global: 42 para divisiones y algoritmos estocásticos.",
        ]
        if "UMAP" not in self.reduction_classes:
            summary_lines.append("- Nota: UMAP no se ejecutó porque la dependencia 'umap-learn' no está disponible en el entorno.")
        summary_lines.extend(["", "## Selección de n_components"])

        for reduction_name, (reduction_cls, _) in self.reduction_classes.items():
            metric_name = reduction_cls.quality_metric_name
            rows = df_results[df_results["reduction"] == reduction_name]
            if rows.empty:
                continue
            best_row_local = rows.iloc[0]
            summary_lines.append(
                f"- **{reduction_name}** → k óptimo = {int(best_row_local['n_components_opt'])}, "
                f"métrica ({metric_name}) = {best_row_local['quality_metric']:.4f}."
            )
        summary_lines.extend(
            [
                "",
                "## Mejor combinación",
                f"- Reducción: **{best_row['reduction']}** (k={int(best_row['n_components_opt'])}).",
                f"- Modelo: **{best_row['model']}**.",
                f"- Métricas en test: Accuracy={best_row['accuracy']:.4f}, "
                f"Precision macro={best_row['precision_macro']:.4f}, "
                f"Recall macro={best_row['recall_macro']:.4f}, F1 macro={best_row['f1_macro']:.4f}.",
                "",
                "## Consideraciones",
                "- t-SNE no ofrece `transform`, por lo que train/test se reducen por separado; los espacios no son directamente comparables.",
                "- UMAP permite `transform` y balancea costo y desempeño, siendo una alternativa práctica frente a t-SNE." if "UMAP" in self.reduction_classes else "- Para utilizar UMAP es necesario instalar 'umap-learn'.",
                f"- Resultados completos disponibles en `{csv_path}`.",
            ]
        )

        report_path = self.results_dir / "summary.md"
        report_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    dataset = preprocess_data(random_state=42)
    runner = ExperimentRunner(
        dataset,
        random_state=42,
        reduction_k_grid=(8, 16, 32, 64, 128),
        figures_dir=Path("figures"),
        results_dir=Path("results"),
        sample_train_size=None,
        sample_test_size=None,
    )
    runner.run()


if __name__ == "__main__":
    main()
