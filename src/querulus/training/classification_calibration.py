"""ClassificationCalibration: isotonic ``cv='prefit'`` на DSM train (OutBoxML).

Порт логики из старого ``calibration/mldw_pipelines.ClassificationCalibration``
без mldataworker / ResultPickle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    unwrap_estimator,
)


@dataclass
class ClassificationCalibration:
    """Isotonic-калибровка вероятностей на **train** DSM (``cv='prefit'``).

    По умолчанию калибрует одну модель ``model_name``. Если ``model_name`` не задан,
    проходит по всем результатам ``manager.get_result()``.
    """

    manager: Any
    model_name: str | None = None
    method: str = "isotonic"
    _calibrated_model: Any = None
    _feature_columns: list[str] = field(default_factory=list)
    _features_numerical: list[str] = field(default_factory=list)
    _features_categorical: list[str] = field(default_factory=list)
    _calibrators: dict[str, Any] = field(default_factory=dict)

    def fit_transform(self) -> Any:
        results = self.manager.get_result()
        names = [self.model_name] if self.model_name else list(results.keys())
        last = None
        for name in names:
            if name is None:
                continue
            last = self._fit_one(name, results[name])
            self._calibrators[name] = last
        if self.model_name and last is not None:
            self._calibrated_model = last
        elif names and last is not None:
            # Последняя модель — активная для transform / predict_proba_positive.
            self._calibrated_model = last
            self.model_name = names[-1]
        return self._calibrated_model

    def _fit_one(self, name: str, result: Any) -> Any:
        subset = result.data_subset
        num = [str(f) for f in (subset.features_numerical or [])]
        cat = [str(f) for f in (subset.features_categorical or [])]
        cols = [*num, *cat]
        self._features_numerical = num
        self._features_categorical = cat
        self._feature_columns = cols

        x_train = subset.X_train.loc[:, cols]
        y_train = subset.y_train
        estimator = unwrap_estimator(ensure_predictable_model(result.model))
        calibrator = CalibratedClassifierCV(
            estimator, method=self.method, cv="prefit"
        )
        sample_weight = getattr(subset, "exposure_train", None)
        try:
            calibrator.fit(x_train, y_train.astype(int), sample_weight=sample_weight)
        except TypeError:
            calibrator.fit(x_train, y_train.astype(int))

        x_test = subset.X_test.loc[:, cols]
        pred_train = pd.Series(
            calibrator.predict_proba(x_train)[:, 1], index=x_train.index, dtype=float
        )
        pred_test = pd.Series(
            calibrator.predict_proba(x_test)[:, 1], index=x_test.index, dtype=float
        )
        load = getattr(result, "load_predictions", None)
        if callable(load):
            load(pred_train, "train")
            load(pred_test, "test")
        return calibrator

    def predict_proba_positive(self, x: pd.DataFrame) -> pd.Series:
        if self._calibrated_model is None:
            raise RuntimeError("Сначала вызовите fit_transform()")
        cols = self._feature_columns or list(x.columns)
        matrix = x.loc[:, cols]
        return pd.Series(
            self._calibrated_model.predict_proba(matrix)[:, 1],
            index=matrix.index,
            dtype=float,
        )

    def transform(self, result: Any = None, data: pd.DataFrame | None = None) -> pd.Series:
        """Как в mldw: калиброванные proba[:, 1] по индексу полного X или по ``data``."""
        if self._calibrated_model is None:
            raise RuntimeError("Сначала вызовите fit_transform()")
        if result is not None:
            subset = result.data_subset
            x = pd.concat([subset.X_train, subset.X_test]).loc[subset.X.index]
        elif data is not None:
            x = data
        else:
            raise ValueError("Нужен result или data")
        return self.predict_proba_positive(x)

    def save_results(self, path: str | Path | None = None) -> Path:
        """Сохранить калибратор и списки признаков (без ResultPickle)."""
        import pickle
        from datetime import datetime

        if self._calibrated_model is None:
            raise RuntimeError("Сначала вызовите fit_transform()")
        if path is None:
            group = getattr(self.manager, "group_name", "cf")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = Path(
                getattr(
                    getattr(self.manager, "_external_config", None),
                    "results_path",
                    ".",
                )
            )
            path = base / f"{group}_cc_{stamp}.pickle"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._calibrated_model,
            "features_categorical": self._features_categorical,
            "features_numerical": self._features_numerical,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        return path

    def load_result(self, pickle_name: str, pickle_path: str | Path = ".") -> None:
        import pickle

        name = pickle_name if pickle_name.endswith(".pickle") else f"{pickle_name}.pickle"
        with (Path(pickle_path) / name).open("rb") as f:
            results = pickle.load(f)
        self._calibrated_model = results["model"]
        self._features_categorical = list(results.get("features_categorical") or [])
        self._features_numerical = list(results.get("features_numerical") or [])
        self._feature_columns = [
            *self._features_numerical,
            *self._features_categorical,
        ]
