import os
import threading
from dataclasses import dataclass
from typing import Optional

import pandas as pd

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]


@dataclass
class LoadedModel:
    model: object
    name: str
    version: Optional[str]
    alias: Optional[str]
    run_id: Optional[str]


class MlflowModelLoader:
    """Carga el modelo `MODEL_NAME@MODEL_ALIAS` desde el Model Registry de MLflow.

    Misma idea que ModelRegistry de proyecto1 (cache + lock + recarga cuando cambia
    la version), pero la fuente es MLflow en vez de MinIO directo.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current: Optional[LoadedModel] = None

    def _load(self) -> LoadedModel:
        # TODO(taller004): implementar la carga desde MLflow. Ver NOTAS_PENDIENTES.md, punto 3.
        raise NotImplementedError("Carga desde MLflow pendiente (ver NOTAS_PENDIENTES.md)")

    def _latest_version(self) -> Optional[str]:
        # TODO(taller004): devolver la version que apunta hoy el alias, para saber si
        # hay que recargar. Ver NOTAS_PENDIENTES.md, punto 3.
        raise NotImplementedError("Consulta del alias en MLflow pendiente (ver NOTAS_PENDIENTES.md)")

    def refresh_if_needed(self) -> LoadedModel:
        version = self._latest_version()
        with self._lock:
            if self._current is None or self._current.version != version:
                self._current = self._load()
            return self._current

    def force_reload(self) -> LoadedModel:
        with self._lock:
            self._current = None
        return self.refresh_if_needed()

    def predict(self, X: pd.DataFrame) -> str:
        loaded = self.refresh_if_needed()
        # TODO(taller004): convertir la salida del modelo a nombre de especie si el
        # modelo devuelve indices en vez de strings. Ver NOTAS_PENDIENTES.md, punto 3.
        return str(loaded.model.predict(X)[0])

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._current


loader = MlflowModelLoader()
