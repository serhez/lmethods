from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from ldata import Dataset  # TODO: Drop this dependency with own Dataset interface
from mloggers import Logger


class Method(ABC):
    """
    The abstract class for a method to solve problems.
    A method wraps a Model and provides higher-level features to enhance the model's capabilities.
    """

    @dataclass(kw_only=True)
    class Config:
        name: str = MISSING
        """The name of the method."""

        debug: bool = False
        """Whether to enable debug mode with extended logging."""

    class _Model(ABC):
        """An interface for a model object used within a method."""

        @classmethod
        def __subclasshook__(cls, subclass):
            return (
                hasattr(subclass, "generate")
                and callable(subclass.generate)
                and hasattr(subclass, "fine_tune")
                and callable(subclass.fine_tune)
            )

        @property
        @abstractmethod
        def generate(
            self,
            context: Union[
                str,
                List[str],
                Iterator[str],
                Dataset[str, str],
            ],
            max_tokens: Optional[int],
        ) -> Union[str, List[str]]:
            """Generates the next tokens given some language context sequence."""
            raise NotImplementedError

        @abstractmethod
        def fine_tune(self, dataset: Union[Dataset, List[Tuple[str, str]]]):
            """Fine-tunes the model given some dataset."""
            raise NotImplementedError

    def __init__(self, model: _Model, config: Config, logger: Logger):
        """
        Initialize the method.

        ### Parameters
        ----------
        `model`: the language model to be used.
        `config`: the configuration of the method.
        `logger`: the logger to be used.
        """

        self._model = model
        self._config = config
        self._logger = logger

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._config.name})"

    @property
    def name(self) -> str:
        """The name of the method."""
        return self._config.name

    def generate(
        self,
        context: Union[
            str,
            List[str],
            Iterator[str],
            Dataset,
        ],
        max_tokens: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Generates the next given number of tokens in the sequence.
        It has similar functionality to HuggingFace's `pipeline` method.

        ### Parameters
        ----------
        `context`: the context/s to generate from.
        - If `context` is a dataset, the method will generate tokens for each context string in the test set.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If None, the method will generate tokens until the EOS token is produced.

        ### Returns
        -------
        The generated tokens.
        - If `context` is a string, the return value is a string.
        - If `context` is a list or iterator of strings, the return value is a list of strings.
        """

        if isinstance(context, str):
            return self._generate_impl(context, max_tokens)

        if isinstance(context, list):
            return [self._generate_impl(c, max_tokens) for c in context]

        if isinstance(context, Iterator):
            return [self._generate_impl(c, max_tokens) for c in context]

        if isinstance(context, Dataset):
            return [self._generate_impl(c, max_tokens) for c in context.test_set.inputs]

        raise ValueError(
            f"Invalid type for `context`: {type(context)}. Must be a string, list of strings, iterator returning strings or `Dataset`."
        )

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _generate_impl(self, context: str, max_tokens: Optional[int] = None) -> str:
        """
        The method's internal implementation of `generate` acting on a single context string.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If None, the method will generate tokens until the EOS token is produced.

        ### Returns
        -------
        The generated tokens.
        """

        pass

    @abstractmethod
    def train(self, dataset: Union[Dataset, List[Tuple[str, str]]]):
        """
        Trains the method (if needed).

        ### Parameters
        ----------
        `dataset`: the dataset to fine-tune the method on.
        """

        pass
