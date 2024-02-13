from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from lmethods.method import Method
from lmethods.protocols import Logger, Model
from lmethods.utils import BaseShotsCollection, read_prompt


# TODO: Design prompts for particular benchmarks + make sure the `format()` calls on the prompts are correct below
# TODO: Implement the `unwrap()` methods referenced below
#       - If they can be generic, then implement them within this class
#       - If they need to be task-specific, then create a `ToTEnabledBenchmark` interface which defines the unwrap methods needed for a benchmark to be compatible
class ToT(Method):
    """
    The Tree-of-Thoughts (ToT) method.
    This is an adaptation of the [the official implementation](https://github.com/princeton-nlp/tree-of-thought-llm) to our interfaces.
    Paper title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models".
    Authors: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao and Karthik Narasimhan.
    Conference: 37th Conference on Neural Information Processing Systems (NeurIPS 2023).
    """

    class GenerationMethod(str, Enum):
        """The generation method."""

        SAMPLE = "sample"
        """The sample method."""

        PROPOSE = "propose"
        """The propose method."""

    class SelectionMethod(str, Enum):
        """The selection method."""

        SAMPLE = "sample"
        """The sample method."""

        GREEDY = "greedy"
        """The greedy method."""

    class EvaluationMethod(str, Enum):
        """The evaluation method."""

        VOTE = "vote"
        """The vote method."""

        VALUE = "value"
        """The value method."""

    @dataclass(kw_only=True)
    class Config(Method.Config):
        """The configuration of the Tree-of-Thoughts (ToT) method."""

        name: str = "TreeOfThoughts"
        """The name of the method."""

        standard_prompt_path: str
        """The path to the standard prompt file."""

        propose_prompt_path: str
        """The path to the propose prompt file."""

        value_prompt_path: str
        """The path to the value prompt file."""

        vote_prompt_path: str
        """The path to the vote prompt file."""

        depth: int = 4
        """The depth of the tree; in the paper, they refer to it as "thought steps"."""

        n_generations: int = 3
        """The number of thoughts to initially sample in a step."""

        n_selections: int = 2
        """The number of thoughts to select in a step."""

        n_evaluations: int = 1
        """The number of scores/votes to sample during evaluation."""

        generation_method: ToT.GenerationMethod = "sample"
        """The generation method."""

        selection_method: ToT.SelectionMethod = "sample"
        """The selection method."""

        evaluation_method: ToT.EvaluationMethod = "vote"
        """The evaluation method."""

        internal_max_tokens: int = 200
        """The maximum number of tokens when calling `model.generate` with internal prompts (e.g., evaluation, proposals, voting, etc.)."""

        use_cache: bool = True
        """Whether to cache the values of the value function."""

    @property
    def config_cls(self) -> type[Method.Config]:
        return ToT.Config

    def __init__(self, model: Model, config: Config, logger: Logger | None = None):
        """
        Initialize the Tree-of-Thoughts (ToT) method.

        ### Parameters
        ----------
        `model`: the language model to be used, complying with the `Model` protocol specified in this library.
        `config`: the configuration of the method.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        super().__init__(model, config, logger)
        self._config: ToT.Config  # pyright is too dumb to understand this

        self._standard_prompt = read_prompt(config.standard_prompt_path)
        self._propose_prompt = read_prompt(config.propose_prompt_path)
        self._value_prompt = read_prompt(config.value_prompt_path)
        self._vote_prompt = read_prompt(config.vote_prompt_path)

        self._cache = {}

    def _reset_state(self):
        self._cache = {}

    # TODO: use max_tokens and shots
    # TODO: check the return type to be a single string
    def _generate_impl(
        self,
        context: str,
        shots: BaseShotsCollection,
        max_tokens: int = 500,
    ) -> tuple[str, dict[str, Any]]:
        outputs = [""]  # current output candidates

        for depth in range(self._config.depth):
            # Step 1: generation
            if self._config.generation_method == ToT.GenerationMethod.SAMPLE:
                candidates = [self._get_samples(context, o) for o in outputs]
            elif self._config.generation_method == ToT.GenerationMethod.PROPOSE:
                candidates = [self._get_proposals(context, o) for o in outputs]
            candidates = list(itertools.chain(*candidates))
            ids = list(range(len(candidates)))

            # Step 2: evaluation
            if self._config.evaluation_method == ToT.EvaluationMethod.VOTE:
                values = self._get_votes(context, candidates)
            elif self._config.evaluation_method == ToT.EvaluationMethod.VALUE:
                values = self._get_values(context, candidates)

            # Step 3: selection
            if self._config.selection_method == ToT.SelectionMethod.SAMPLE:
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(
                    ids, size=self._config.n_selections, p=ps
                ).tolist()
            elif self._config.selection_method == ToT.SelectionMethod.GREEDY:
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[
                    : self._config.n_selections
                ]
            selected = [candidates[id] for id in select_ids]

            self._logger.debug(
                {
                    "Depth": depth,
                    "Context": context,
                    "Outputs": outputs,
                    "Candidates": candidates,
                    "Values": values,
                    "Selected": selected,
                }
            )

            outputs = selected

        return outputs, {"usage": {}}

    def _get_value(self, context, output):
        value_prompt = self._value_prompt.format(
            problem=context, solution=output
        ).strip()

        if self._config.use_cache and value_prompt in self._cache:
            return self._cache[value_prompt]

        try:
            outputs, info = self._model.generate(
                value_prompt,
                n_samples=self._config.n_evaluations,
                max_tokens=self._config.internal_max_tokens,
            )
            value_outputs = list(outputs[0])
            self._record_global_usage(info)
        except Exception as e:
            self._logger.error(
                {
                    "[TreeOfThoughts.generate] The model failed to generate the values": None,
                    "Corrective action": "The method will use 0 as the values",
                    "Error": str(e),
                }
            )
            value_outputs = [0] * self._config.n_evaluations

        value = value_outputs_unwrap(context, output, value_outputs)  # TODO: implement

        if self._config.use_cache:
            self._cache[value_prompt] = value

        return value

    def _get_values(self, context, outputs):
        values = []
        seen_outputs = []

        for output in outputs:  # each partial output
            if output in seen_outputs:  # avoid duplicate candidates
                value = 0
            else:
                value = self._get_value(context, output)
                seen_outputs.append(output)

            values.append(value)

        return values

    def _get_votes(self, context, outputs):
        vote_prompt = self._vote_prompt.format(
            problem=context, solutions=outputs
        ).strip()

        try:
            outputs, info = self._model.generate(
                vote_prompt,
                n_samples=self._config.n_evaluations,
                max_tokens=self._config.internal_max_tokens,
            )
            vote_outputs = list(outputs[0])
            self._record_global_usage(info)
        except Exception as e:
            self._logger.error(
                {
                    "[TreeOfThoughts.generate] The model failed to generate the votes": None,
                    "Corrective action": "The method will use zeros as the votes",
                    "Error": str(e),
                }
            )
            vote_outputs = [0] * self._config.n_evaluations

        values = vote_outputs_unwrap(vote_outputs, len(outputs))  # TODO: implement

        return values

    def _get_proposals(self, context, output):
        propose_prompt = self._propose_prompt.format(
            problem=context, solution=output
        ).strip()

        try:
            outputs, info = self._model.generate(
                propose_prompt,
                max_tokens=self._config.internal_max_tokens,
            )
            proposals = outputs[0][0].split("\n")
            self._record_global_usage(info)
        except Exception as e:
            self._logger.error(
                {
                    "[TreeOfThoughts.generate] The model failed to generate the proposals": None,
                    "Corrective action": "The method will use empty strings as the proposals",
                    "Error": str(e),
                }
            )
            proposals = [""] * self._config.n_generations

        return [output + _ + "\n" for _ in proposals]

    def _get_samples(self, context, output):
        prompt = self._standard_prompt.format(problem=context, solution=output).strip()

        try:
            outputs, info = self._model.generate(
                prompt,
                n_samples=self._config.n_generations,
                max_tokens=self._config.internal_max_tokens,
            )
            samples = list(outputs[0])
            self._record_global_usage(info)
        except Exception as e:
            self._logger.error(
                {
                    "[TreeOfThoughts.generate] The model failed to generate the samples": None,
                    "Corrective action": "The method will use empty strings as the samples",
                    "Error": str(e),
                }
            )
            samples = [""] * self._config.n_generations

        return [output + _ for _ in samples]

    def _value_outputs_unwrap(self, context, output, outputs):
        raise NotImplementedError

    def _vote_outputs_unwrap(self, outputs, n):
        raise NotImplementedError

    def train(self, _):
        raise NotImplementedError
