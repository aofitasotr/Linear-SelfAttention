from abc import ABC, abstractmethod

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class TransformerLoaderInterface(ABC):
    """Интерфейс для загрузчиков трансформеров."""

    @property
    @abstractmethod
    def tokenizer(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def config(self):
        pass


class TransformerLoader(TransformerLoaderInterface):
    """Ленивая обёртка для загрузки токенизатора, модели и конфига."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._config = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
                num_labels=5,
            )
        return self._model

    @property
    def config(self):
        if self._config is None:
            self._config = AutoConfig.from_pretrained(self._model_name)
        return self._config

