from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Union

from ldata import Dataset
from lmodels import Model
from mloggers import Logger


class Method(ABC):
    """
    The abstract class for a method to solve problems.
    A method wraps a Model and provides higher-level features to enhance the model's capabilities.
    """

    @dataclass(kw_only=True)
    class Config:
        debug: bool = False
        """Whether to enable debug mode with extended logging."""

    def __init__(self, model: Model, config: Config, logger: Logger):
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

        # TODO: implement this once the Dataset class is implemented
        if isinstance(context, Dataset):
            raise NotImplementedError(
                "The `Dataset` class is not yet implemented. Please use a list of strings instead."
            )

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
    def fine_tune(self, dataset: Dataset):
        """
        Fine-tunes the method.

        ### Parameters
        ----------
        `dataset`: the dataset to fine-tune the method on.
        """

        pass
