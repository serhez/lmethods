from enum import Enum


class DependencySyntax(str, Enum):
    """The syntax used to represent dependencies between sub-problems."""

    NONE = "none"
    """No dependencies between sub-problems are expected."""

    BRACKETS_PARENS = "brackets_parens"
    """
    The ID of the sub-problem within brackets, followed by the comma-separated ids of the sub-problems it depends on within parenthesis.
    E.g., "[P3](P0, P2)".
    """

    BRACKETS_ANGLE = "brackets_angle"
    """
    The ID of the sub-problem within brackets, followed by the description which can contain dependencies whose IDs are specified between angle brackets.
    E.g., "[P3] Some description that depends on <P1> and <P2>.".
    """

    BRACKETS_CURLY = "brackets_curly"
    """
    The ID of the sub-problem within brackets, followed by the description which can contain dependencies whose IDs are specified between curly braces.
    E.g., "[P3] Some description that depends on {P1} and {P2}.".
    """


class SearchStrategy(str, Enum):
    """The strategy used to traverse a decomposition graph."""

    DFS = "depth_first"
    """The sub-problems are searched for in a depth-first order."""

    BFS = "breadth_first"
    """The sub-problems are searched for in a breadth-first order."""
