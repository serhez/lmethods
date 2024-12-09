from enum import Enum


class DependencySyntax(str, Enum):
    """The syntax used to represent dependencies between sub-problems."""

    NONE = "none"
    """No dependencies between sub-problems are expected."""

    BRACKETS_PARENS = "brackets_parens"
    """
    The ID of the sub-problem within brackets, followed by the comma-separated ids of the sub-problems it depends on within parenthesis.
    Only compatible with `HierarchySyntax.BULLET_POINTS`.
    E.g., "[P3](P0, P2)".
    """

    BRACKETS_ANGLE = "brackets_angle"
    """
    The ID of the sub-problem within brackets, followed by the description which can contain dependencies whose IDs are specified between angle brackets.
    Only compatible with `HierarchySyntax.BULLET_POINTS`.
    E.g., "[P3] Some description that depends on <P1> and <P2>.".
    """

    BRACKETS_CURLY = "brackets_curly"
    """
    The ID of the sub-problem within brackets, followed by the description which can contain dependencies whose IDs are specified between curly braces.
    Only compatible with `HierarchySyntax.BULLET_POINTS`.
    E.g., "[P3] Some description that depends on {P1} and {P2}.".
    """

    HEADER_ANGLE = "header_angle"
    """
    The ID of the sub-problem is specified in a markdown header, followed by the description which can contain dependencies whose IDs are specified between angle brackets.
    Only compatible with `HierarchySyntax.MARKDOWN_HEADERS`.
    E.g.:

    ### P3

    Some description that depends on <P1> and <P2>.
    """

    HEADER_CURLY = "header_curly"
    """
    The ID of the sub-problem is specified in a markdown header, followed by the description which can contain dependencies whose IDs are specified between curly braces.
    Only compatible with `HierarchySyntax.MARKDOWN_HEADERS`.
    E.g.:

    ### P3

    Some description that depends on {P1} and {P2}.
    """


class HierarchySyntax(str, Enum):
    """The syntax used to represent sub-problems when splitting."""

    BULLET_POINTS = "bullet_points"
    """
    Each sub-problem is represented as a bullet point.
    Only compatible with `DependencySyntax.BRACKETS_PARENS`, `DependencySyntax.BRACKETS_ANGLE`, and `DependencySyntax.BRACKETS_CURLY`.
    """

    MARKDOWN_HEADERS = "markdown_headers"
    """
    Each sub-problem is presented below a markdown header, which specifies the subproblem ID and its dependencies (if any).
    The sub-problem description is given below the header.
    The header level can be any. Only the lowest level (numerically) will be considered as the starting point of sub-problems;
    higher levels will be considered part of the sub-problem's description.
    Only compatible with `DependencySyntax.HEADER_ANGLE` and `DependencySyntax.HEADER_CURLY`.
    """


class SearchStrategy(str, Enum):
    """The strategy used to traverse a decomposition graph."""

    DFS = "depth_first"
    """The sub-problems are searched for in a depth-first order."""

    BFS = "breadth_first"
    """The sub-problems are searched for in a breadth-first order."""
