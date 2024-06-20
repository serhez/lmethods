from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, overload

import numpy as np
import numpy.typing as npt

from lmethods.protocols import Dataset, Logger, Model
from lmethods.utils import BaseShotsCollection, NullLogger, Usage, classproperty


# TODO: Support multiple models for performing different parts of a method (e.g., a dict of models)
class Method(ABC):
    """
    The abstract class for a method to solve problems.
    A method wraps a Model and provides higher-level features to enhance the model's capabilities.
    """

    @dataclass(kw_only=True)
    class Config:
        """The configuration of a method."""

        name: str
        """The name of the method."""

        max_threads: int = 4
        """The maximum number of threads to use for parallel processing."""

        answer_regex: str | None = None
        """
        The regular expression used to extract the answer from the output.
        - If `None`, the method will not attempt to extract the answers and the whole output will be considered as the answer.
        - If the model's output does not match the regular expression, a warning will be issued and the whole output will be considered as the answer.
        """

    @dataclass(kw_only=True)
    class GenerationInfo:
        """Extra information about the generation process of the method."""

        usage: Usage = field(default_factory=Usage)
        """The usage statistics of the method."""

        def __add__(self, other: Method.GenerationInfo | None) -> Method.GenerationInfo:
            """
            Combines the generation information of two methods.

            ### Parameters
            ----------
            `other`: the other generation information to combine with.

            ### Returns
            ----------
            The combined generation information.
            """

            return Method.GenerationInfo(
                usage=self.usage + other.usage if other is not None else self.usage
            )

        def __radd__(
            self, other: Method.GenerationInfo | None
        ) -> Method.GenerationInfo:
            """
            Combines the generation information of two methods in reverse.

            ### Parameters
            ----------
            `other`: the other generation information to combine with.

            ### Returns
            ----------
            The combined generation information.
            """

            return self + other

        def __iadd__(
            self, other: Method.GenerationInfo | None
        ) -> Method.GenerationInfo:
            """
            Combines the generation information of two methods in-place.

            ### Parameters
            ----------
            `other`: the other generation information to combine with.

            ### Returns
            ----------
            The combined generation information.
            """

            if other is not None:
                self.usage += other.usage

            return self

        def to_json(self) -> dict[str, Any]:
            """
            Converts the generation information to a JSON-serializable dictionary.

            ### Returns
            ----------
            The JSON-serializable dictionary.
            """

            return {
                "usage": self.usage.to_json(),
            }

    @classproperty
    @abstractmethod
    def config_cls(cls) -> type[Config]:
        """The configuration class of the method."""

        ...

    @classproperty
    @abstractmethod
    def generation_info_cls(cls) -> type[GenerationInfo]:
        """The generation information class of the method."""

        ...

    @classproperty
    @abstractmethod
    def shots_collection_cls(self) -> type[BaseShotsCollection]:
        """The class of the collection of shots to use for in-context learning."""

        ...

    def __init__(self, model: Model, config: Config, logger: Logger | None = None):
        """
        Initialize the method.

        ### Parameters
        ----------
        `model`: the language model to be used, complying with the `Model` protocol specified in this library.
        `config`: the configuration of the method.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        self._model = model
        self._config = config
        if logger is None:
            self._logger = NullLogger()
        else:
            self._logger = logger
        self._usage = Usage()

        self._answer_regex = None
        if self._config.answer_regex is not None:
            self._answer_regex = re.compile(
                self._config.answer_regex, flags=re.MULTILINE | re.DOTALL
            )

        self._logger.debug({"[Method.config]": asdict(self._config)})

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._config.name})"

    @property
    def name(self) -> str:
        """The name of the method."""

        return self._config.name

    @property
    def usage(self) -> Usage:
        """The aggregated usage statistics of the method instance, accounting for all generations."""

        return self._usage

    @usage.setter
    def usage(self, value: Usage):
        """Sets the usage statistics of the model instance."""

        self._usage = value

    @abstractmethod
    def _reset_state(self):
        """Resets the method's state."""

        ...

    def _extract_answer(self, output: str) -> str:
        """
        Extracts the answer from the output using the regular expression specified in `Config.answer_regex`.

        ### Parameters
        ----------
        `output`: the output from the model.

        ### Returns
        -------
        The extracted answer.

        ### Notes
        -------
        - If the regular expression is `None`, the whole output will be considered as the answer.
        - If the regular expression does not match the output, a warning will be issued and the whole output will be considered as the answer.
        """

        if self._answer_regex is None:
            return output

        # Use try-except to not break the `generate` pipeline for any reason
        try:
            match = self._answer_regex.findall(output)
            if len(match) > 1:
                self._logger.warn(
                    {
                        f"Multiple matches found for the regex '{self._config.answer_regex}'.": None,
                        "Output": output,
                        "Corrective action": "The last match will be considered as the answer.",
                    }
                )
            answer = match[-1]
        except Exception:
            self._logger.warn(
                {
                    f"Could not find the regex '{self._config.answer_regex}' in the output.": None,
                    "Output": output,
                    "Corrective action": "The whole output will be considered as the answer.",
                }
            )
            answer = output

        return answer

    @overload
    def generate(
        self,
        context: str,
        shots: BaseShotsCollection,
        max_tokens: int = 500,
    ) -> tuple[str, GenerationInfo]: ...

    @overload
    def generate(
        self,
        context: list[str] | npt.NDArray[np.str_] | Dataset,
        shots: BaseShotsCollection,
        max_tokens: int = 500,
    ) -> tuple[list[str], GenerationInfo]: ...

    # TODO: include n_samples
    # TODO: return a `numpy.NDArray`
    def generate(
        self,
        context,
        shots,
        max_tokens=500,
    ):
        """
        Generates the next given number of tokens in the sequence.
        It has similar functionality to HuggingFace's `pipeline` method.
        This method can be overriden by the child class to take advantage of GPU parallelization for multi-context inputs.

        ### Parameters
        ----------
        `context`: the contexts to generate from.
        - If `context` is a dataset, the method will generate tokens for each context string in the test set.
        `shots`: a collection of shots to use for in-context learning in different parts of the method's generation process.
        - If any unit of the collection is empty, the method will not use in-context learning for the relevant task (i.e., zero-shot).
        `max_tokens`: the maximum number of tokens to generate per context string.

        ### Returns
        -------
        A tuple containing:
        - The generated text. This will be a list if multiple contexts are provided as input; otherwise, it will be a single string.
        - Extra information about the generation process of the method.
        """

        if isinstance(context, str):
            output, info = self._generate_impl(context, shots, max_tokens)
            self._reset_state()
            return output, info
        elif isinstance(context, Dataset):
            context = list(context.test_set.inputs)
        elif isinstance(context, np.ndarray):
            context = list(context)
        elif not isinstance(context, list):
            raise ValueError(
                f"Invalid type for `context`: {type(context)}. Must be a string, list of strings, iterator returning strings or `Dataset`."
            )

        outputs = []
        agg_info = self.generation_info_cls()
        for i, c in enumerate(context):
            self._logger.info(
                f"[{self.__class__.__name__}] Generating {i+1}/{len(context)}"
            )

            output, info = self._generate_impl(c, shots, max_tokens)
            self._reset_state()

            outputs.append(output)
            agg_info += info

        return outputs, info

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _generate_impl(
        self,
        context: str,
        shots: BaseShotsCollection,
        max_tokens: int = 500,
    ) -> tuple[str, GenerationInfo]:
        """
        The method's internal implementation of `generate` acting on a single context string.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `shots`: a collection of shots to use for in-context learning in different parts of the method's generation process.
        - If any unit of the collection is empty, the method will not use in-context learning for the relevant task (i.e., zero-shot).
        `max_tokens`: the maximum number of tokens to generate per context string.

        ### Returns
        -------
        A tuple containing:
        - The generated text.
        - Extra information about the generation process of the model.
        """

        ...

    @abstractmethod
    def train(self, dataset: Dataset | list[tuple[str, str]]):
        """
        Trains the method (if needed).

        ### Parameters
        ----------
        `dataset`: the dataset to fine-tune the method on.
        """

        ...
