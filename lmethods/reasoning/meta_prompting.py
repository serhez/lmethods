from dataclasses import MISSING, dataclass
from typing import List, Optional, Tuple, Union

from ldata import Dataset
from mloggers import Logger

from lmethods.method import Method


class MetaPrompting(Method):
    """
    A method that uses meta-prompting to solve multi-step reasoning problems.
    Depending on the meta-prompt, the method can be turned into CoT (zero-shot or few-shot), least-to-most, etc.
    """

    @dataclass(kw_only=True)
    class Config(Method.Config):
        """The configuration of the recursive prompting method."""

        name: str = "MetaPrompting"
        """The name of the method."""

        meta_prompt_path: str = MISSING
        """The path to the meta-prompt used to solve multi-step reasoning problems."""

    def __init__(self, model: Method._Model, config: Config, logger: Logger):
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

    def train(self, dataset: Union[Dataset, List[Tuple[str, str]]]):
        self._model.fine_tune(dataset)


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_meta_prompting", node=MetaPrompting.Config)
except ModuleNotFoundError:
    pass
