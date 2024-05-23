from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from lmethods.method import Method
from lmethods.protocols import Dataset, Logger, Model
from lmethods.utils import (
    BaseShotsCollection,
    Usage,
    choose_response_via_sc,
    classproperty,
    construct_shots_str,
    read_prompt,
)


class MetaPrompting(Method):
    """
    A method that uses meta-prompting to solve multi-step reasoning problems.
    Depending on the meta-prompt, the method can be turned into CoT (zero-shot or few-shot), least-to-most, skeleton-of-thought, etc.
    Self-Consistency (X. Wang et al., 2022) can be used to improve the quality of the responses by tuning the `Config.self_consistency_n` parameter.
    """

    @dataclass(kw_only=True)
    class Config(Method.Config):
        """The configuration of the meta-prompting method."""

        name: str = "MetaPrompting"
        """The name of the method."""

        prompt_path: str
        """The path to the meta-prompt used to solve multi-step reasoning problems."""

        self_consistency_n: int = 1
        """
        The number of samples that will be taken from the model to find the response via Self-Consistency (X. Wang et al., 2022).
        - If `== 1`, the Self-Consistency mechanism will not be used.
        - If `>= 2`, the model will vote on the most common reponses or, if there are no common responses, to choose the best response.
        """

        self_consistency_max_n_per_call: int = 10
        """
        The number of samples that will be provided in a single Self-Consistency (X. Wang et al., 2022) call.
        If multiple calls are needed, binary search will be used to find the best response.
        The value must be greater than or equal to 2.
        """

        def __post_init__(self):
            if self.self_consistency_n < 1:
                raise ValueError(
                    "`self_consistency_n` must be greater than or equal to 1."
                )
            if self.self_consistency_max_n_per_call < 2:
                raise ValueError(
                    "`self_consistency_max_n_per_call` must be greater than or equal to 2."
                )

    @dataclass(kw_only=True)
    class GenerationInfo(Method.GenerationInfo):
        all_responses: list[list[str | None]] = field(default_factory=list)
        """
        All the responses sampled from the model before choosing one via Self-Consistency.
        If the the responses were not generated successfully (e.g., an exception was thrown), the value is `None`.
        - In such case, only a single `None` is appended to the inner list, hence the length of the inner list will not be equal to `Config.self_consistency_n`.
        """

        def __add__(
            self, other: MetaPrompting.GenerationInfo | None
        ) -> MetaPrompting.GenerationInfo:
            return MetaPrompting.GenerationInfo(
                usage=self.usage + other.usage if other is not None else self.usage,
                all_responses=self.all_responses + other.all_responses
                if other is not None
                else self.all_responses + [[None]],
            )

        def __radd__(
            self, other: MetaPrompting.GenerationInfo | None
        ) -> MetaPrompting.GenerationInfo:
            return self + other

        def __iadd__(
            self, other: MetaPrompting.GenerationInfo | None
        ) -> MetaPrompting.GenerationInfo:
            if other is not None:
                self.usage += other.usage
                self.all_responses += other.all_responses
            else:
                self.all_responses += [[None]]  # type: ignore[reportAttributeAccessIssue]
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
                "all_responses": self.all_responses,
            }

    class ShotsCollection(BaseShotsCollection):
        """A collection of (input, target) pairs to use for in-context learning."""

        def __init__(
            self,
            solve: list[tuple[str, str]] = [],
        ):
            """
            Initialize the shots collection.

            ### Parameters
            ----------
            `solve`: the shots for solving problems.
            """

            super().__init__(
                {
                    "solve": solve,
                }
            )

        @property
        def solve(self) -> list[tuple[str, str]]:
            """The shots for solving problems."""

            return self._shots["solve"]

    @classproperty
    def config_cls(cls) -> type[Config]:
        return cls.Config

    @classproperty
    def generation_info_cls(cls) -> type[GenerationInfo]:
        return cls.GenerationInfo

    @classproperty
    def shots_collection_cls(cls) -> type[ShotsCollection]:
        return cls.ShotsCollection

    def __init__(self, model: Model, config: Config, logger: Logger | None = None):
        """
        Initialize the meta-prompting method.

        ### Parameters
        ----------
        `model`: the language model to be used, complying with the `Model` protocol specified in this library.
        `config`: the configuration of the method.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        super().__init__(model, config, logger)
        self._config: (
            MetaPrompting.Config
        )  # pyright is not smart enough to infer the type

        self._prompt = read_prompt(config.prompt_path)

    def _reset_state(self):
        return

    def generate(
        self,
        context: str | list[str] | npt.NDArray[np.str_] | Dataset,
        shots: ShotsCollection = ShotsCollection(),
        max_tokens: int = 500,
    ) -> tuple[str | list[str], GenerationInfo]:
        if isinstance(context, str):
            return self._generate_impl(context, shots, max_tokens)
        elif isinstance(context, Dataset):
            context = list(context.test_set.inputs)
        elif isinstance(context, np.ndarray):
            context = list(context)
        elif not isinstance(context, list):
            raise ValueError(
                f"Invalid type for `context`: {type(context)}. Must be a string, list of strings, numpy array of strings or `Dataset`."
            )

        local_usage = Usage()
        shots_str = construct_shots_str(shots.solve)

        inputs = [[self._prompt.format(problem=c, shots=shots_str)] for c in context]

        try:
            # One sample is chosen with a low temperature (0.0)
            outputs, info = self._model.generate(
                inputs, max_tokens=max_tokens, temperature=0.0
            )

            # All other samples are chosen with a high temperature (the model's default temp.)
            if self._config.self_consistency_n > 1:
                extra_outputs, extra_info = self._model.generate(
                    inputs,
                    n_samples=self._config.self_consistency_n - 1,
                    max_tokens=max_tokens,
                )
                outputs = np.concatenate((outputs, extra_outputs), axis=1)
                info += extra_info

            outputs = [[s for s in o] for o in outputs]
            responses = [[self._extract_answer(s) for s in o] for o in outputs]
            local_usage += info.usage
            self.usage += info.usage
        except Exception as e:
            self._logger.error(
                {
                    "[MetaPrompting.generate] The model failed to generate responses": None,
                    "Error": str(e),
                }
            )
            outputs = [[""] * self._config.self_consistency_n] * len(context)
            responses = outputs

        # Self-Consistency
        chosen_idxs = []
        for i in range(len(responses)):
            chosen_idx, sc_usage = choose_response_via_sc(
                self._model,
                context[i],
                outputs[i],
                self._config.self_consistency_max_n_per_call,
                self._logger,
            )
            chosen_idxs.append(chosen_idx)
            local_usage += sc_usage
            self.usage += sc_usage
        chosen_responses = [responses[i][chosen_idxs[i]] for i in range(len(context))]

        self._logger.debug(
            {
                "[MetaPrompting.generate]": None,
                "Batch context": context,
                "Batch input": inputs,
                "Batch output": outputs,
                "Batch all answers": responses,
                "Batch chosen indices": chosen_idxs,
                "Batch chosen answers": chosen_responses,
                "Usage": local_usage,
            }
        )

        return chosen_responses, MetaPrompting.GenerationInfo(
            usage=local_usage,
            all_responses=responses,  # type: ignore[reportArgumentType]
        )

    def _generate_impl(
        self,
        context: str,
        shots: ShotsCollection = ShotsCollection(),
        max_tokens: int = 500,
    ) -> tuple[str, GenerationInfo]:
        local_usage = Usage()
        shots_str = construct_shots_str(shots.solve)

        input = self._prompt.format(problem=context, shots=shots_str)

        try:
            output, info = self._model.generate(
                input, n_samples=self._config.self_consistency_n, max_tokens=max_tokens
            )
            local_usage += info.usage
            self.usage += info.usage
        except Exception as e:
            self._logger.error(
                {
                    "[MetaPrompting.generate] The model failed to generate a response": None,
                    "Error": str(e),
                }
            )
            output = [[""] * self._config.self_consistency_n]

        # Self-Consistency
        responses = [self._extract_answer(s) for s in output[0]]
        chosen_idx, sc_usage = choose_response_via_sc(
            self._model,
            context,
            responses,
            self._config.self_consistency_max_n_per_call,
            self._logger,
        )
        local_usage += sc_usage
        self.usage += sc_usage

        self._logger.debug(
            {
                "[MetaPrompting.generate]": None,
                "Context": context,
                "Input": input,
                "Output": [s for s in output[0]],
                "All parsed answers": responses,
                "Chosen index": chosen_idx,
                "Chosen answer": responses[chosen_idx],
                "Usage": local_usage,
            }
        )

        return responses[chosen_idx], MetaPrompting.GenerationInfo(
            usage=local_usage,
            all_responses=[responses],  # type: ignore[reportArgumentType]
        )

    def train(self, dataset: Dataset | list[tuple[str, str]]):
        self._model.fine_tune(dataset)
