from enum import Enum


class DependencySyntax(str, Enum):
    """The syntax used to represent dependencies between sub-problems."""

    NONE = "none"
    """No dependencies between sub-problems are expected."""

    BRACKETS_PARENS = "brackets_parens"
    """
    The id of the sub-problem within brackets, followed by the comma-separated ids of the sub-problems it depends on within parenthesis.
    E.g., "[P3](P0, P2)".
    """
