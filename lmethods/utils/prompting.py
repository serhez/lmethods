from abc import ABC
from typing import overload

from lmethods.utils.decomposition import HierarchySyntax

SEP_CHARS = [".", "!", "?", ":", ";", ",", "|"]
"""The characters that can be used as separators."""

END_CHARS = [".", "!", "?"]
"""The characters that can end a sentence."""

BULLET_POINTS_CHARS = ["-", "*", "•", "+"] + [str(i) for i in range(1, 50)]
"""
The characters that can be used to start a bullet point.
Only the first 50 numbers are considered as bullet points (of a numbered list); consider extending this list if you need more.
"""


class BaseShotsCollection(ABC):
    """
    A base class for a collection of shots.

    This class encapsulates a dictionary of shots for different parts of the method's generation process.
    For example, in a decomposition method, different sets of shots can be use to demonstrate the decomposition and solution-merging steps.

    Child classes should re-implement the `__init__` method with parameters corresponding to each key of the dictionary and call this class's `__init__` method with the custom shots dictionary.
    Child classes can also add properties to access the shots in a more convenient way.
    """

    def __init__(self, shots: dict[str, list[tuple[str, str]]] | None = None):
        """
        Initialize the base shots collection.

        ### Parameters
        ----------
        `shots`: a dictionary of shots for different parts of the method's generation process.
        """

        if shots is None:
            shots = {}

        self._shots: dict[str, list[tuple[str, str]]] = shots

    @overload
    def add(self, input: str, target: str, key: str): ...

    @overload
    def add(self, pair: tuple[str, str], key: str): ...

    def add(self, *args, **_):
        """Add a (input, target) pair to a part of the collection."""

        if len(args) == 2:
            self._shots[args[1]].append(args[0])
        elif len(args) == 3:
            self._shots[args[2]].append((args[0], args[1]))

    def __getitem__(self, key: str) -> list[tuple[str, str]]:
        return self._shots[key]

    def __setitem__(self, key: str, value: list[tuple[str, str]]):
        self._shots[key] = value

    def __delitem__(self, key: str):
        del self._shots[key]

    def __iter__(self):
        return iter(self._shots)

    def __len__(self) -> int:
        return len(self._shots)


def read_prompt(path: str) -> str:
    """
    Read a prompt from a file.

    ### Parameters
    ----------
    `path`: the path to the file containing the prompt.

    ### Returns
    -------
    The content of the file as a string.

    ### Notes
    -------
    - The extra newline added at the end of the file by the native `read` function is removed; newlines added by the user are preserved.
    """

    with open(path, "r") as f:
        prompt = f.read()
        if len(prompt) > 0 and prompt[-1] == "\n":
            prompt = prompt[:-1]

    return prompt


def construct_shots_str(
    shots: list[tuple[str, str]],
    syntax: HierarchySyntax = HierarchySyntax.BULLET_POINTS,
    header_level: int = 2,
) -> str:
    """
    Construct a string from a list of shots.

    ### Parameters
    ----------
    `shots`: a list of (input, target) pairs to use for in-context learning.
    `syntax`: the syntax of the subproblems in the shots.
    `header_level`: the level to use for the headers (e.g., "Answer" and "Problem description").
    - This parameter is only used when `syntax` is `HierarchySyntax.MARKDOWN_HEADERS`.

    ### Returns
    -------
    A string representation of the shots.

    ### Notes
    -------
    - The returned string contains two newlines at its end.
    """

    if len(shots) == 0:
        return ""

    shots_str = ""
    for shot in shots:
        if syntax == HierarchySyntax.BULLET_POINTS:
            # Problem
            shots_str += f"Problem: {shot[0]}{'' if any(shot[0].endswith(c) for c in END_CHARS) else '.'}\n"

            # Answer
            if any(shot[1].strip().startswith(c) for c in BULLET_POINTS_CHARS):
                sep = "\n"
            else:
                sep = " "
            shots_str += f"Answer:{sep}{shot[1]}{'' if any(shot[1].endswith(c) for c in END_CHARS) else '.'}\n\n"
        elif syntax == HierarchySyntax.MARKDOWN_HEADERS:
            # Problem
            shots_str += f"{'#' * header_level} Problem description\n\n{shot[0]}{'' if any(shot[0].endswith(c) for c in END_CHARS) else '.'}\n\n"

            # Answer
            shots_str += f"{'#' * header_level} Answer\n\n{shot[1]}{'' if any(shot[1].endswith(c) for c in END_CHARS) else '.'}\n\n"

    return shots_str


def add_roles_to_context(
    context: str, system_chars: int = 0, assistant_chars: int = 0
) -> list[dict[str, str]]:
    """
    Add role messages to the context, as specified by `Config.last_chars_as_assistant` and `Config.first_chars_as_system`.

    ### Parameters
    ----------
    `context`: the context to add the role messages to.

    ### Returns
    ----------
    The context with the role messages added.
    """

    result = []

    if system_chars > 0:
        result.append(
            {
                "role": "system",
                "content": context[:system_chars],
            }
        )

    if assistant_chars > 0:
        result.append(
            {
                "role": "user",
                "content": context[system_chars:-assistant_chars],
            }
        )
        result.append(
            {
                "role": "assistant",
                "content": context[-assistant_chars:],
            }
        )
    else:
        result.append(
            {
                "role": "user",
                "content": context[system_chars:],
            }
        )

    return result
