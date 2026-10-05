"""Shared contract implemented by every selectable classifier."""

from abc import ABC, abstractmethod
from PIL import Image


class Classifier(ABC):
    model_id: str
    display_name: str

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model internals onto device. Called once per process."""

    @abstractmethod
    def predict(self, image: Image.Image) -> dict:
        """Return top, confidence, probs and optional model-specific fields."""

    def metadata(self) -> dict:
        """Metadata used to populate the UI model picker."""
        return {"id": self.model_id, "name": self.display_name}
