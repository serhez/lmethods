from dataclasses import dataclass
from typing import Optional

from lmodels import Model
from mloggers import Logger

from lmethods.method import Method


class MetaPrompting(Method):
    """
    A method that uses meta-prompting to solve multi-step reasoning problems.
    Depending on the meta-prompt, the method can be turned into CoT (zero-shot or few-shot), least-to-most, etc.
    """

    @dataclass(kw_only=True)
    class Config(Method.Config):
        meta_prompt_path: str
        """The path to the meta-prompt used to solve multi-step reasoning problems."""

    def __init__(self, model: Model, config: Config, logger: Logger):
        """
        Initialize the recursive prompting method.

        ### Parameters
        ----------
        `model`: the language model to be used.
        `config`: the configuration of the method.
        `logger`: the logger to be used.
        """

        super().__init__(model, config, logger)

        with open(config.meta_prompt_path, "r") as file:
            self._meta_prompt = file.read()

    def _generate_impl(self, context: str, max_tokens: Optional[int] = None) -> str:
        input = self._meta_prompt.format(problem=context)

        if self._config.debug:
            self._logger.debug(f"Input (including meta-prompt):\n\n{input}")

        return self._model.generate(input, max_tokens)

    def fine_tune(self, dataset):
        self._model.fine_tune(dataset)
