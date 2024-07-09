from __future__ import annotations

import json
from dataclasses import dataclass
from queue import Queue
from typing import Any

import n2w

from lmethods.method import Method
from lmethods.protocols import Logger, Model
from lmethods.utils import (
    BULLET_POINTS_CHARS,
    END_CHARS,
    BaseShotsCollection,
    DependencySyntax,
    IDGenerator,
    SearchStrategy,
    Usage,
    add_roles_to_context,
    classproperty,
    construct_shots_str,
    read_prompt,
)


# TODO: Provide prompts in a similar way as with the ShotsCollection
# TODO: Parallelize `generate` (look at the WIP file)
# TODO: Multi-process sub-problem solving
class RecursivePrompting(Method):
    """A method that uses recursive prompting to solve multi-step reasoning problems."""

    @dataclass(kw_only=True)
    class Config(Method.Config):
        """The configuration of the recursive prompting method."""

        name: str = "RecursivePrompting"
        """The name of the method."""

        max_nodes: int = 100
        """
        The maximum number of graph nodes that can be generated in the problem-solving process.
        When the number of nodes reaches this value, the method will stop decomposing the problem and will attempt to solve each existing node directly.
        """

        max_depth: int = 3
        """The maximum depth of the recursion."""

        max_width: int = 4
        """The maximum number of sub-problems that can be provided at any given decomposition step."""

        enforce_max_width: bool = False
        """
        Whether to enforce the maximum amount of sub-problems at each decomposition step.
        If `True`, the method will truncate the list of sub-problems to the maximum width.
        If `False`, the method will not enforce the maximum width, but will still warn if the number of sub-problems exceeds the maximum width.
        - The max. width will still be passed to the split prompt to inform the model of the maximum width.
        - It is recommended to set `max_nodes` to a reasonable value if this is set to `False`.
        """

        max_internal_tokens: int = 500
        """The maximum number of tokens that can be generated in internal calls to the model (e.g., decomposing, generating instructions, merging sub-solutions, etc.)."""

        elicit_instructions: bool = False
        """
        Whether to elicit instructions to merge sub-solutions to solve the original problem.
        If `False`, the method will not generate instructions and will directly attempt to merge sub-solutions.
        """

        # TODO: implement
        elicit_revision: bool = False
        """Whether to revise the solution using the model."""

        dependency_syntax: DependencySyntax = DependencySyntax.BRACKETS_PARENS
        """
        The syntax used to represent dependencies between sub-problems.
        This is irrelevant if dependencies are not elicited via the split prompt.
        """

        search_strategy: SearchStrategy = SearchStrategy.BFS
        """The strategy used to traverse the decomposition graph."""

        unit_prompt_path: str
        """The path to the meta-prompt used to solve unit problems."""

        split_prompt_path: str
        """The path to the meta-prompt used to split problems into sub-problems."""

        instructions_prompt_path: str | None = None
        """The path to the meta-prompt used to generate instructions to merge sub-solutions."""

        merge_prompt_path: str
        """The path to the meta-prompt used to combine solutions to sub-problems to solve the original problem."""

        unit_first_chars_as_system: int = 0
        """The amount of characters from the beginning of the unit context that will be passed to the model with the role of 'system'."""

        unit_last_chars_as_assistant: int = 0
        """The amount of characters from the end of the unit context that will be passed to the model with the role of 'assistant'."""

        split_first_chars_as_system: int = 0
        """The amount of characters from the beginning of the split context that will be passed to the model with the role of 'system'."""

        split_last_chars_as_assistant: int = 0
        """The amount of characters from the end of the split context that will be passed to the model with the role of 'assistant'."""

        instructions_first_chars_as_system: int = 0
        """The amount of characters from the beginning of the instructions context that will be passed to the model with the role of 'system'."""

        instructions_last_chars_as_assistant: int = 0
        """The amount of characters from the end of the instructions context that will be passed to the model with the role of 'assistant'."""

        merge_first_chars_as_system: int = 0
        """The amount of characters from the beginning of the merge context that will be passed to the model with the role of 'system'."""

        merge_last_chars_as_assistant: int = 0
        """The amount of characters from the end of the merge context that will be passed to the model with the role of 'assistant'."""

        graph_file_path: str | None = None
        """
        The path to the file where the graph of the problem-solving process will be saved.
        This is a JSON file that can be used to visualize the problem-solving process.
        The graph will be saved only if the path is provided (i.e., the value is not `None`).
        The JSON file will be structured as an array of graphs (one per problem solved via calls to `generate`).
        Each graph will be a dictionary with the following keys:
        - `root_id`: the ID of the root problem.
        - `nodes`: a dictionary where the keys are the node IDs and the value is a dictionary with the node information. Each node dictionary will have the following keys:
            - `description`: the description of the problem.
            - `dependencies`: a list of the IDs of the problems that this problem depends on.
            - `solution`: the solution to the problem.
        """

    class ShotsCollection(BaseShotsCollection):
        """A collection of (input, target) pairs to use for in-context learning."""

        def __init__(
            self,
            unit: list[tuple[str, str]] | None = None,
            split: list[tuple[str, str]] | None = None,
            instructions: list[tuple[str, str]] | None = None,
            merge: list[tuple[str, str]] | None = None,
        ):
            """
            Initialize the shots collection.

            ### Parameters
            ----------
            `unit`: the shots for solving unit problems.
            `split`: the shots for splitting problems into sub-problems.
            `instructions`: the shots for generating instructions to merge sub-solutions.
            `merge`: the shots for merging sub-solutions to solve the original problem.
            """

            super().__init__(
                {
                    "unit": unit if unit is not None else [],
                    "split": split if split is not None else [],
                    "instructions": instructions if instructions is not None else [],
                    "merge": merge if merge is not None else [],
                }
            )

        @property
        def unit(self) -> list[tuple[str, str]]:
            return self._shots["unit"]

        @property
        def split(self) -> list[tuple[str, str]]:
            return self._shots["split"]

        @property
        def instructions(self) -> list[tuple[str, str]]:
            return self._shots["split"]

        @property
        def merge(self) -> list[tuple[str, str]]:
            return self._shots["merge"]

    class _Problem:
        """A problem to be solved."""

        def __init__(
            self,
            uid: str,
            lid: str,
            description: str,
            parent: str | None = None,
            depth: int = 1,
            dependencies: list[str] | None = None,
            instructions: str = "",
            solution: str | None = None,
        ):
            """
            Initialize the problem.

            ### Parameters
            ----------
            `uid`: the unique global ID of the problem.
            `lid`: the local ID of the problem in the context of its parents and children in the decomposition graph.
            - This ID may not be unique within the whole graph.
            `description`: the description of the problem.
            `parent`: the UID of the parent problem in the decomposition graph.
            - If `None`, the problem is the root of the graph.
            `depth`: the minimum distance from the root of the graph to this node.
            `dependencies`: the IDs of the problems that this problem depends on.
            `instructions`: the instructions to merge the sub-solutions.
            `solution`: the solution to the problem.

            ### Notes
            ----------
            - The description and solution will be stripped of surrounding whitespace.
            - The description will be punctuated if it is not already.
            - The instructions will be punctuated if they are not already.
            - The solution will be punctuated if it is not already.
            """

            self._uid = uid
            self._lid = lid
            self._dependencies = dependencies if dependencies is not None else []
            self._parent = parent
            self._depth = depth
            self.description = description
            self.instructions = instructions
            self.solution = solution

        @property
        def uid(self) -> str:
            """The unique global ID of the problem."""

            return self._uid

        @uid.setter
        def uid(self, value: str):
            """Set the global ID of the problem."""

            self._uid = value

        @property
        def lid(self) -> str:
            """
            The local ID of the problem in the context of its parents and children in the decomposition graph.
            This ID may not be unique within the whole graph.
            """

            return self._lid

        @lid.setter
        def lid(self, value: str):
            """Set the local ID of the problem."""

            self._lid = value

        @property
        def description(self) -> str:
            """The description of the problem."""

            return self._description

        @description.setter
        def description(self, value: str):
            """Set the description of the problem."""

            self._description = value.strip()
            if self._description != "":
                self._description = f"{self._description}{'' if any(self._description[-1] == c for c in END_CHARS) else '.'}"

        @property
        def parent(self) -> str | None:
            """The UID of the parent problem in the decomposition graph."""

            return self._parent

        @property
        def depth(self) -> int:
            """The minimum distance from the root of the graph to this node."""

            return self._depth

        @property
        def dependencies(self) -> list[str]:
            """The IDs of the problems that this problem depends on."""

            return self._dependencies

        @dependencies.setter
        def dependencies(self, value: list[str]):
            """Set the IDs of the problems that this problem depends on."""

            self._dependencies = value

        @property
        def instructions(self) -> str:
            """The instructions to merge the sub-solutions."""

            return self._instructions

        @instructions.setter
        def instructions(self, value: str):
            """Set the instructions to merge the sub-solutions."""

            self._instructions = value
            if self._instructions != "":
                self._instructions = value.strip()
                self._instructions = f"{self._instructions}{'' if any(self._instructions[-1] == c for c in END_CHARS) else '.'}"

        @property
        def solution(self) -> str | None:
            """The solution to the problem."""

            return self._solution

        @solution.setter
        def solution(self, value: str | None):
            """Set the solution to the problem."""

            self._solution = value
            if self._solution is not None and self._solution != "":
                self._solution = self._solution.strip()
                self._solution = f"{self._solution}{'' if any(self._solution[-1] == c for c in END_CHARS) else '.'}"

        @property
        def is_solved(self) -> bool:
            return self.solution is not None

        def to_json(self) -> dict[str, Any]:
            """
            Converts the problem to a JSON-serializable dictionary.

            ### Returns
            ----------
            The JSON-serializable dictionary.
            """

            return {
                "uid": self.uid,
                "lid": self.lid,
                "description": self.description,
                "parent": self.parent,
                "depth": self.depth,
                "dependencies": self.dependencies,
                "instructions": self.instructions,
                "solution": self.solution,
            }

    @classproperty
    def config_cls(cls) -> type[Config]:
        return cls.Config

    @classproperty
    def generation_info_cls(cls) -> type[Method.GenerationInfo]:
        return cls.GenerationInfo

    @classproperty
    def shots_collection_cls(cls) -> type[ShotsCollection]:
        return cls.ShotsCollection

    def __init__(self, model: Model, config: Config, logger: Logger | None = None):
        """
        Initialize the recursive prompting method.

        ### Parameters
        ----------
        `model`: the language model to be used, complying with the `Model` protocol specified in this library.
        `config`: the configuration of the method.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        super().__init__(model, config, logger)
        self._config: (
            RecursivePrompting.Config
        )  # pyright is too dumb to understand this type

        self._problems_cache: dict[str, RecursivePrompting._Problem] = {}
        self._id_gen = IDGenerator()
        self._current_root_id: str | None = None

        self._unit_prompt = read_prompt(config.unit_prompt_path)
        self._split_prompt = read_prompt(config.split_prompt_path)
        self._merge_prompt = read_prompt(config.merge_prompt_path)
        if config.elicit_instructions:
            assert (
                config.instructions_prompt_path is not None
            ), "`config.instructions_prompt_path` is required when 'elicit_instructions' is `True`."
            try:
                self._instructions_prompt = read_prompt(config.instructions_prompt_path)
            except FileNotFoundError as e:
                self._logger.warn(
                    f"The file '{config.instructions_prompt_path}' was not found. An instructions prompt is required when 'elicit_instructions' is `True`."
                )
                raise e
        else:
            self._instructions_prompt = None

        self._local_usage = Usage()

        self._subproblem_prefixes = [
            "sub-problem:",
            "subproblem:",
            "sub problem:",
            "problem:",
        ]
        self._subproblem_prefixes.extend(BULLET_POINTS_CHARS)

    def _add_to_cache(self, problems: _Problem | list[_Problem]):
        if isinstance(problems, RecursivePrompting._Problem):
            problems = [problems]

        for problem in problems:
            if problem.uid in self._problems_cache:
                raise ValueError(
                    f"[RecursivePrompting.generate] The problem with UID '{problem.uid}' is already in the cache."
                )
            self._problems_cache[problem.uid] = problem

    def _reset_state(self):
        self._logger.debug("[RecursivePrompting.reset_state] Resetting the state.")
        self._problems_cache = {}
        self._id_gen.reset()
        self._local_usage = Usage()
        self._current_root_id = None

    def _generate_impl(
        self,
        context: str,
        shots: ShotsCollection = ShotsCollection(),
        max_tokens: int = 500,
    ) -> tuple[str, Method.GenerationInfo]:
        self._current_root_id = self._id_gen.next()
        problem = RecursivePrompting._Problem(
            self._current_root_id, self._current_root_id, context
        )

        self._logger.debug(
            {
                "[RecursivePrompting.generate]": None,
                "Context": context,
                "Max. tokens": max_tokens,
                "N. unit shots": len(shots.unit),
                "N. split shots": len(shots.split),
                "N. instructions shots": len(shots.instructions),
                "N. merge shots": len(shots.merge),
                "Root ID": self._current_root_id,
                "Root obj.": problem,
                "N. of nodes": len(self._problems_cache),
                "Local usage": self._local_usage,
                "Global usage": self.usage,
            }
        )

        self._add_to_cache(problem)

        if self._config.search_strategy == SearchStrategy.BFS:
            self._solve_bfs(problem, shots)
        elif self._config.search_strategy == SearchStrategy.DFS:
            self._solve_dfs(problem, shots)
        else:
            raise ValueError(
                f"[RecursivePrompting.generate] The search strategy '{self._config.search_strategy}' is not supported."
            )
        assert problem.is_solved, "The problem was not solved."

        output = problem.solution[:max_tokens]  # type: ignore[reportOptionalSubscript]

        # TODO: revise the solution
        if self._config.elicit_revision:
            raise NotImplementedError(
                "The revision of the solution is not implemented."
            )

        # Save the graph
        self._save_graph(self._current_root_id)

        self._logger.debug(
            {
                "[RecursivePrompting.generate]": None,
                "Root ID": self._current_root_id,
                "Root obj.": problem,
                "Context": context,
                "Max. tokens": max_tokens,
                "Output": output,
                "N. of nodes": len(self._problems_cache),
                "Usage stats": self._local_usage,
            }
        )

        return output, Method.GenerationInfo(usage=self._local_usage)

    def _solve_dfs(
        self,
        problem: _Problem,
        shots: ShotsCollection = ShotsCollection(),
        only_merge: bool = False,
        _visited: set[str] | None = None,
    ):
        """
        Solves a decomposition graph from the root using recursive prompting via the DFS strategy and stores the solutions in the problem objects.
        The method will split the problem into sub-problems while traversing the graph if `only_merge` is `False`; otherwise, it will only unit-solve and merge the graph.
        The method tracks the visited nodes to detect cycles in the graph.

        ### Parameters
        ----------
        `problem`: the problem to be solved.
        `shots`: a shots collection to use for in-context learning.
        - If empty at any stage, the method will not use in-context learning (i.e., zero-shot).
        `only_merge`: whether to only merge the sub-solutions to solve the original problem.
        - If `True`, the method will not attempt to split the problem into sub-problems.
        [DO NOT USE] `_visited`: the set of IDs of the problems that have already been visited via DFS.
        - Used internally to detect cycles in the graph; should not be set by the user.
        """

        # Reset the default `_visited` value if problem is the root.
        # This is necessary as default arguments are evaluated only once in Python,
        # which is the trick we use here to share the visited nodes across recursive calls.
        if _visited is None:
            self._logger.debug(
                f"[RecursivePrompting.generate:dfs] Resetting the visited nodes from {_visited}"
            )
            _visited = set()

        mode = "bfs" if only_merge else "dfs"

        if problem.uid in _visited:
            self._logger.error(
                {
                    f"[RecursivePrompting.generate:{mode}] A cycle has been detected.": None,
                    "Depth": problem.depth,
                    "Problem UID": problem.uid,
                    "Visited nodes": _visited,
                }
            )
            if self._current_root_id:
                self._save_graph(self._current_root_id)
            raise RuntimeError(
                f"[RecursivePrompting.generate:{mode}] The dependencies of the sub-problems contain a cycle."
            )
        _visited.add(problem.uid)

        if not only_merge:
            self._logger.debug(f"Splitting problem {problem.uid} via DFS")
            self._split(problem, shots)

        # Solve each sub-problem recursively (if any)
        # TODO: we can solve sub-problems in parallel here
        for dep_id in [
            id for id in problem.dependencies if not self._problems_cache[id].is_solved
        ]:
            self._solve_dfs(self._problems_cache[dep_id], shots, only_merge, _visited)

        # Obtain merging instructions
        if self._config.elicit_instructions:
            self._set_instructions(problem, shots.instructions)

        # Merge the sub-solutions
        self._logger.debug(
            f"Solving problem {problem.uid} with dependencies {problem.dependencies}"
        )
        self._solve_directly(problem, shots)

        self._logger.debug(
            {
                f"[RecursivePrompting.generate:{mode}]": None,
                "Depth": problem.depth,
                "Problem UID": problem.uid,
                "Problem desc.": problem.description,
                "Problem sol.": problem.solution,
                "Sub-problems IDs": problem.dependencies,
                "Sub-problems desc.": [
                    self._problems_cache[id].description for id in problem.dependencies
                ],
                "Sub-problems sol.": [
                    self._problems_cache[id].solution for id in problem.dependencies
                ],
                "Instructions": problem.instructions,
            }
        )

    def _solve_bfs(
        self,
        problem: _Problem,
        shots: ShotsCollection = ShotsCollection(),
    ):
        """
        Solves a problem using recursive prompting via the BFS strategy and stores the solution in the problem object.

        ### Parameters
        ----------
        `problem`: the problem to be solved.
        `subproblems_ids`: the IDs of the sub-problems to be solved.
        `instructions`: the instructions to merge the sub-solutions.
        `shots`: a shots collection to use for in-context learning.
        - If empty at any stage, the method will not use in-context learning (i.e., zero-shot).
        """

        self._logger.debug(
            f"Splitting root problem {problem.uid} via BFS: {problem} with {len(problem.dependencies)} pre-existing dependencies"
        )
        self._split(problem, shots)

        unsolved = Queue()
        self._logger.debug(f"Initializing the BFS queue to {unsolved.queue}")

        for dep_id in problem.dependencies:
            unsolved.put(dep_id)
            self._logger.debug(
                f"Added root problem dep. {dep_id} to the BFS queue: {unsolved.queue}"
            )

        # Split (or solve directly if necessary) the sub-problems using BFS
        # TODO: parallelize this
        while not unsolved.empty():
            dep_id = unsolved.get()
            dep = self._problems_cache[dep_id]

            if not dep.is_solved:
                self._logger.debug(f"Splitting problem {dep.uid} via BFS: {dep}")
                self._split(dep, shots)
                for subdep_id in dep.dependencies:
                    if not self._problems_cache[subdep_id].is_solved:
                        unsolved.put(subdep_id)
                        self._logger.debug(
                            f"Added dep. {subdep_id} to the BFS queue: {unsolved.queue}"
                        )

        # Merge all problems in the graph reusing the DFS recursive logic
        self._solve_dfs(problem, shots, only_merge=True)

        self._logger.debug(
            {
                "[RecursivePrompting.generate:bfs:root]": None,
                "Problem UID": problem.uid,
                "Problem desc.": problem.description,
                "Problem sol.": problem.solution,
                "Sub-problems IDs": problem.dependencies,
                "Sub-problems desc.": [
                    self._problems_cache[id].description for id in problem.dependencies
                ],
                "Sub-problems sol.": [
                    self._problems_cache[id].solution for id in problem.dependencies
                ],
                "Instructions": problem.instructions,
            }
        )

    def _split(
        self,
        problem: _Problem,
        shots: ShotsCollection = ShotsCollection(),
    ):
        """
        Split a problem into sub-problems.

        ### Parameters
        ----------
        `problem`: the problem to be split or solved.
        `shots`: a shots collection to use for in-context learning.
        - If empty at any stage, the method will not use in-context learning (i.e., zero-shot).
        """

        # If the max. depth or the max. num. of nodes are reached, solve the problem directly
        if (
            problem.depth >= self._config.max_depth
            or len(self._problems_cache) >= self._config.max_nodes
        ):
            self._solve_directly(problem, shots)
            return

        # Split the problem into sub-problems
        shots_str = construct_shots_str(shots.split)

        split_prompt = self._split_prompt.format(
            problem=problem.description,
            width=n2w.convert(self._config.max_width),
            shots=shots_str,
        ).strip()
        split_prompt = add_roles_to_context(
            split_prompt,
            system_chars=self._config.split_first_chars_as_system,
            assistant_chars=self._config.split_last_chars_as_assistant,
        )

        try:
            output, info = self._model.generate(
                split_prompt, max_tokens=self._config.max_internal_tokens
            )
            split = self._extract_answer(output[0][0])
            self._local_usage += info.usage
            self.usage += info.usage
        except Exception as e:
            self._logger.error(
                {
                    f"[RecursivePrompting.generate:split] The problem with UID '{problem.uid}' could not be split": None,
                    "Error source": "model",
                    "Error message": str(e),
                }
            )
            split = ""

        # Insanity check...
        if split is None:
            self._logger.error(
                f"[RecursivePrompting.generate:split] The problem with UID '{problem.uid}' was not split."
            )
            split = ""

        subproblems_ids = self._parse_subproblems(split, problem.uid)
        self._logger.debug(
            f"Extending the dependencies of problem {problem.uid}: {problem.dependencies} + {subproblems_ids}"
        )
        problem.dependencies.extend(subproblems_ids)

    def _set_instructions(
        self, problem: _Problem, shots: list[tuple[str, str]] | None = None
    ):
        """
        Generate instructions to merge sub-solutions.

        ### Parameters
        ----------
        `problem`: the original problem.
        `subproblems`: the sub-problems to be solved.

        ### Raises
        -------
        `RuntimeError`: if the method is configured to elicit instructions, but no instructions prompt is provided.

        ### Notes
        ----------
        - The method will not generate instructions if `Config.elicit_instructions` is `False`.
        """

        if shots is None:
            shots = []

        shots_str = construct_shots_str(shots)
        deps_str = self._construct_dependencies_str(problem.dependencies)

        if self._instructions_prompt is None:
            raise RuntimeError(
                "[RecursivePrompting.generate:instructions] The method is configured to elicit instructions, but no instructions prompt is provided."
            )

        instructions_prompt = self._instructions_prompt.format(
            problem=problem.description, subproblems=deps_str, shots=shots_str
        ).strip()
        instructions_prompt = add_roles_to_context(
            instructions_prompt,
            system_chars=self._config.instructions_first_chars_as_system,
            assistant_chars=self._config.instructions_last_chars_as_assistant,
        )

        try:
            output, info = self._model.generate(
                instructions_prompt, max_tokens=self._config.max_internal_tokens
            )
            instructions = self._extract_answer(output[0][0])
            self._local_usage += info.usage
            self.usage += info.usage
        except Exception as e:
            self._logger.error(
                {
                    f"[RecursivePrompting.generate:instructions] The problem with UID '{problem.uid}' could not generate instructions": None,
                    "Error source": "model",
                    "Error message": str(e),
                }
            )
            instructions = ""
        if instructions is None:
            self._logger.error(
                f"[RecursivePrompting.generate] Instructions for the problem with UID '{problem.uid}' were not generated."
            )
            instructions = ""

        problem.instructions = instructions

    def _solve_directly(self, problem: _Problem, shots: ShotsCollection):
        """
        Solve a problem directly, either as a unit problem (no dependencies) or by merging (dependencies).

        ### Parameters
        ----------
        `problem`: the problem to be solved.
        `shots`: a list of (input, target) pairs to use for in-context learning.
        """

        if problem.depth >= self._config.max_depth:
            case = "max_depth"
        elif len(self._problems_cache) >= self._config.max_nodes:
            case = "max_budget"
        elif len(problem.dependencies) == 0:
            case = "unit"
        else:
            case = "merge"

        try:
            output, info = self._model.generate(
                self._construct_unit_context(problem, shots),
                max_tokens=self._config.max_internal_tokens,
            )
            if output[0][0] is None:
                self._logger.error(
                    f"[RecursivePrompting.generate:{case}] The problem with UID '{problem.uid}' was not solved."
                )
                problem.solution = ""
            else:
                problem.solution = self._extract_answer(output[0][0])
            self._local_usage += info.usage
            self.usage += info.usage
        except Exception as e:
            self._logger.error(
                {
                    f"[RecursivePrompting.generate:{case}] The problem with UID '{problem.uid}' could not be solved": None,
                    "Error source": "model",
                    "Error message": str(e),
                }
            )
            output = [[""]]
            problem.solution = ""

        # Insanity check to avoid infinite loops
        if not problem.is_solved:
            self._logger.error(
                f"[RecursivePrompting.generate:{case}] The problem with UID '{problem.uid}' was not solved."
            )
            problem.solution = ""

        self._logger.debug(
            {
                f"[RecursivePrompting.generate:{case}]": None,
                "Depth": problem.depth,
                "N. of nodes": len(self._problems_cache),
                "Model output": output[0][0],
                "Problem UID": problem.uid,
                "Problem desc.": problem.description,
                "Problem sol.": problem.solution,
            }
        )

    def _construct_unit_context(
        self, problem: _Problem, shots_collection: ShotsCollection
    ) -> list[dict[str, str]]:
        """
        Constructs the context to solve a unit problem.

        ### Parameters
        ----------
        `problem`: the problem to be solved.
        `shots`: a list of (input, target) pairs to use for in-context learning.

        ### Returns
        -------
        The context to solve the unit problem.

        ### Notes
        ----------
        - If the problem has dependencies and they all have been solved, the merge prompt with no instructions will be used.
        - Otherwise, the unit prompt will be used.
        """

        if len(problem.dependencies) > 0 and all(
            self._problems_cache[id].is_solved for id in problem.dependencies
        ):
            context = self._merge_prompt.format(
                problem=problem.description,
                subsolutions=self._construct_dependencies_str(problem.dependencies),
                instructions=problem.instructions,
                shots=construct_shots_str(shots_collection.merge),
            ).strip()
            context = add_roles_to_context(
                context,
                system_chars=self._config.merge_first_chars_as_system,
                assistant_chars=self._config.merge_last_chars_as_assistant,
            )
        else:
            context = self._unit_prompt.format(
                problem=problem.description,
                shots=construct_shots_str(shots_collection.unit),
            ).strip()
            context = add_roles_to_context(
                context,
                system_chars=self._config.unit_first_chars_as_system,
                assistant_chars=self._config.unit_last_chars_as_assistant,
            )

        return context

    def _parse_subproblems(self, output: str, parent_uid: str) -> list[str]:
        """
        Parse the output of the model to find sub-problems.

        ### Parameters
        ----------
        `output`: the output of the model.
        `parent_uid`: the UID of the parent problem in the decomposition graph.

        ### Returns
        -------
        A list of the IDs of the sub-problems to be solved.
        - If the list is empty, the problem is a unit problem.
        - The list may be empty if the maximum width or the maximum number of nodes is exceeded, in which case the problem must be solved directly.
        """

        # The dictionary with the local IDs as keys, in order to correctly work out the dependencies
        subproblems_dict: dict[str, RecursivePrompting._Problem] = {}

        detected_prefix = None
        local_i = 1

        for line in output.split("\n"):
            if any(line.startswith(char) for char in ["  ", "\t"]):
                self._logger.warn(
                    "[RecursivePrompting.parse_subproblems] The output contains a nested list. Only considering the first level of the list."
                )
                continue

            line = line.strip()

            for prefix in self._subproblem_prefixes:
                if line.lower().startswith(prefix):
                    if detected_prefix is not None and detected_prefix != prefix:
                        self._logger.warn(
                            f"[RecursivePrompting.parse_subproblems] A different sub-problem prefix was used in the output: '{prefix}'. "
                            f"The first detected prefix was '{detected_prefix}'. Ignoring the line."
                        )
                        continue
                    elif detected_prefix is None:
                        detected_prefix = prefix

                    # Get rid of the prefix
                    line = line[len(prefix) :].strip()

                    # Construct the problem
                    p = self._parse_raw_subproblem(line, local_i, parent_uid)
                    local_i += 1

                    # Create global IDs; this is necessary because the IDs generated
                    # by the model are likely to repeat across independent splitting steps.
                    # The keys of the dictionary are still the local IDs.
                    if p.lid in subproblems_dict:
                        self._logger.warn(
                            f"[RecursivePrompting.parse_subproblems] The local ID '{p.lid}' is repeated in the output. Incoming dependencies will be ignored."
                        )
                        p.lid = p.uid
                    subproblems_dict[p.lid] = p

                    break

        # Exceeding the maximum width
        if len(subproblems_dict) > self._config.max_width:
            log_msg = f"[RecursivePrompting.parse_subproblems] The number of sub-problems ({len(subproblems_dict)}) exceeds the max. width ({self._config.max_width})."
            if self._config.enforce_max_width:
                log_msg += " The problem will be solved directly."
                self._logger.warn(log_msg)
                return []
            self._logger.warn(log_msg)

        # Exceeding the maximum number of nodes
        elif len(subproblems_dict) + len(self._problems_cache) > self._config.max_nodes:
            self._logger.warn(
                f"[RecursivePrompting.[parse_subproblems]] Adding the proposed sub-problems ({len(subproblems_dict)}) "
                f"to the existing amount of problems ({len(self._problems_cache)}) would exceed the max. "
                f"num. of nodes ({self._config.max_nodes}). The problem will be solved directly."
            )
            return []

        # Substitute the local IDs with the global IDs in the dependencies
        for p_local_id, p in subproblems_dict.items():
            global_deps = []
            for dep_local_id in p.dependencies:
                if dep_local_id in subproblems_dict:
                    global_deps.append(subproblems_dict[dep_local_id].uid)
                else:
                    self._logger.warn(
                        f"[RecursivePrompting.parse_subproblems] The dependency with local ID '{dep_local_id}' "
                        "of the problem with local ID '{p_local_id}' does not exist. "
                        "It may have been removed if the maximum num. of nodes was exceeded. Ignoring the dependency."
                    )
            self._logger.debug(
                f"Extending same-depth dependencies of problem {p.uid}: {p.dependencies} + {global_deps}"
            )
            p.dependencies.extend(global_deps)

        # Add the sub-problems to the cache
        for p in subproblems_dict.values():
            self._add_to_cache(p)

        return [p.uid for p in list(subproblems_dict.values())]

    def _parse_raw_subproblem(
        self, raw_problem: str, problem_idx: int, parent_uid: str
    ) -> _Problem:
        """
        Parse a raw sub-problem string (given in a bullet point) to find the ID, description and dependencies between sub-problems.

        ### Parameters
        ----------
        `raw_problem`: the raw problem string.
        `problem_idx`: the local index of the raw problem in the list of raw problems, which is used to generate the local ID if it is not provided by the model.
        `parent_uid`: the UID of the parent problem in the decomposition graph.

        ### Returns
        -------
        The parsed problem's ID.

        ### Notes
        ----------
        - The raw sub-problem is expected to NOT contain the initial bullet point character.
        - The raw sub-problem will be stripped of surrounding whitespace.
        - The sub-problem is NOT stored in the cache.
        - The parsing will be done according to `Config.dependency_syntax`.
        - The dependencies will be referred to by their local IDs (`lid`); you likely want to then replace these with their global IDs (`uid`).
        """

        # Get rid of surrounding whitespace
        raw_problem = raw_problem.strip()

        if self._config.dependency_syntax == DependencySyntax.NONE:
            uid = self._id_gen.next()
            lid = str(problem_idx)
            desc = raw_problem
            deps_ids = []

        elif self._config.dependency_syntax == DependencySyntax.BRACKETS_PARENS:
            desc = raw_problem

            # ID
            uid = self._id_gen.next()
            try:
                if not desc.startswith("["):
                    raise ValueError
                right_bracket_i = desc.index("]")
                lid = desc[1:right_bracket_i]
                desc = desc[right_bracket_i + 1 :]
            except ValueError:
                self._logger.error(
                    "[RecursivePrompting.generate] The ID of the problem could not be found for dependency syntax 'BRACKETS_PARENS'."
                )
                lid = str(problem_idx)

            # Dependencies
            try:
                if not desc.startswith("("):
                    raise ValueError
                right_paren_i = desc.index(")")
                deps_ids = [id.strip() for id in desc[1:right_paren_i].split(",")]
                if len(deps_ids) == 0 or (len(deps_ids) == 1 and deps_ids[0] == ""):
                    deps_ids = []
                desc = desc[right_paren_i + 1 :]
            except ValueError:
                self._logger.warn(
                    "[RecursivePrompting.generate] The dependencies of the problem could not be found for dependency syntax 'BRACKETS_PARENS'."
                )
                deps_ids = []

            # Description
            desc = desc.strip()

        else:
            raise NotImplementedError(
                "The specified dependency syntax is not implemented."
            )

        return RecursivePrompting._Problem(
            uid,
            lid,
            desc,
            parent_uid,
            self._problems_cache[parent_uid].depth + 1,
            deps_ids,
        )

    def _construct_dependencies_str(self, dependencies: list[str]) -> str:
        if len(dependencies) == 0:
            return ""

        return "".join(
            [
                f"- Sub-problem {dep.lid}: {dep.description}"
                + ("" if not dep.is_solved else f" Sub-solution: {dep.solution}\n")
                for dep in [self._problems_cache[id] for id in dependencies]
            ]
        )[:-1]

    def _save_graph(self, root_id: str) -> bool:
        """
        Generate and save a graph of the problem-solving process.

        ### Parameters
        ----------
        `root`: the ID of the root problem of the graph.

        ### Returns
        -------
        Whether the graph was saved successfully.

        ### Notes
        ----------
        - The graph will only be saved if `Config.graph_file_path` is not `None`.
        """

        if self._config.graph_file_path is None:
            return False

        nodes = {}
        for id, problem in self._problems_cache.items():
            nodes[id] = {
                "description": problem.description,
                "dependencies": problem.dependencies,
                "solution": problem.solution,
            }

        graph_dict = {"root_id": root_id, "nodes": nodes}

        try:
            with open(self._config.graph_file_path, "r") as file:
                graphs = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self._logger.warn(
                f"[RecursivePrompting.save_graph] The graph file '{self._config.graph_file_path}' could not be read. The contents will be overwritten."
            )
            graphs = []
        graphs.append(graph_dict)

        try:
            with open(self._config.graph_file_path, "w") as file:
                json.dump(graphs, file, indent=4)
        except Exception as e:
            self._logger.error(
                {
                    "[RecursivePrompting.save_graph] The graph could not be written to the file": None,
                    "Error source": "file",
                    "Error message": str(e),
                }
            )
            return False

        return True

    def train(self, _):
        raise NotImplementedError("RecursivePrompting does not support training.")
