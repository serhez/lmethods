from __future__ import annotations

import json
from dataclasses import dataclass

import n2w

from lmethods.method import Method
from lmethods.protocols import Logger, Model
from lmethods.utils import (
    BULLET_POINTS_CHARS,
    END_CHARS,
    BaseShotsCollection,
    DependencySyntax,
    IDGenerator,
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
        The graph will be saved only if the path is provided.
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
            unit: list[tuple[str, str]] = [],
            split: list[tuple[str, str]] = [],
            instructions: list[tuple[str, str]] = [],
            merge: list[tuple[str, str]] = [],
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
                    "unit": unit,
                    "split": split,
                    "instructions": instructions,
                    "merge": merge,
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
            id: str,
            description: str,
            dependencies: list[str] | None = None,
            solution: str | None = None,
        ):
            """
            Initialize the problem.

            ### Parameters
            ----------
            `id`: the ID of the problem.
            `description`: the description of the problem.
            `dependencies`: the IDs of the problems that this problem depends on.
            `solution`: the solution to the problem.

            ### Notes
            ----------
            - The description and solution will be stripped of surrounding whitespace.
            - The description will be punctuated if it is not already.
            - The solution will be punctuated if it is not already.
            """

            self._id = id

            if dependencies is None:
                self._dependencies = []
            else:
                self._dependencies = dependencies

            self._description = description.strip()
            self._description = f"{self._description}{'' if any(self._description.endswith(c) for c in END_CHARS) else '.'}"

            self._solution = solution
            if self._solution is not None:
                self._solution = self._solution.strip()
                self._solution = f"{self._solution}{'' if any(self._solution.endswith(c) for c in END_CHARS) else '.'}"

        @property
        def id(self) -> str:
            """The ID of the problem."""

            return self._id

        @id.setter
        def id(self, value: str):
            """Set the ID of the problem."""

            self._id = value

        @property
        def description(self) -> str:
            """The description of the problem."""

            return self._description

        @description.setter
        def description(self, value: str):
            """Set the description of the problem."""

            self._description = value.strip()
            self._description = f"{self._description}{'' if any(self._description.endswith(c) for c in END_CHARS) else '.'}"

        @property
        def dependencies(self) -> list[str]:
            """The IDs of the problems that this problem depends on."""

            return self._dependencies

        @dependencies.setter
        def dependencies(self, value: list[str]):
            """Set the IDs of the problems that this problem depends on."""

            self._dependencies = value

        @property
        def solution(self) -> str | None:
            """The solution to the problem."""

            return self._solution

        @solution.setter
        def solution(self, value: str | None):
            """Set the solution to the problem."""

            self._solution = value
            if self._solution is not None:
                self._solution = self._solution.strip()
                self._solution = f"{self._solution}{'' if any(self._solution.endswith(c) for c in END_CHARS) else '.'}"

        @property
        def is_solved(self) -> bool:
            return self.solution is not None

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
            if problem.id in self._problems_cache:
                raise ValueError(
                    f"[RecursivePrompting.generate] The problem with ID '{problem.id}' is already in the cache."
                )
            self._problems_cache[problem.id] = problem

    def _reset_state(self):
        self._problems_cache = {}
        self._id_gen.reset()
        self._local_usage = Usage()

    def _generate_impl(
        self,
        context: str,
        shots: ShotsCollection = ShotsCollection(),
        max_tokens: int = 500,
    ) -> tuple[str, Method.GenerationInfo]:
        problem = RecursivePrompting._Problem(self._id_gen.next(), context)
        self._add_to_cache(problem)

        self._solve(problem, 1, shots)
        assert problem.is_solved, "The problem was not solved."

        output = problem.solution[:max_tokens]  # type: ignore[reportOptionalSubscript]

        # TODO: revise the solution
        if self._config.elicit_revision:
            raise NotImplementedError(
                "The revision of the solution is not implemented."
            )

        # Save the graph
        if self._config.graph_file_path is not None:
            graph_dict = self._generate_graph(problem.id)

            try:
                with open(self._config.graph_file_path, "r") as file:
                    graphs = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                graphs = []
            graphs.append(graph_dict)

            with open(self._config.graph_file_path, "w") as file:
                json.dump(graphs, file, indent=4)

        self._logger.debug(
            {
                "[RecursivePrompting.generate]": None,
                "Root ID": problem.id,
                "Context": context,
                "Max. tokens": max_tokens,
                "Output": output,
                "N. unit shots": len(shots.unit),
                "N. split shots": len(shots.split),
                "N. instructions shots": len(shots.instructions),
                "N. merge shots": len(shots.merge),
                "Usage stats": self._local_usage,
            }
        )

        return output, Method.GenerationInfo(usage=self._local_usage)

    def _solve(
        self, problem: _Problem, depth: int, shots: ShotsCollection = ShotsCollection()
    ):
        """
        Solves a problem using recursive prompting and stores the solution in the problem object.

        ### Parameters
        ----------
        `problem`: the problem to be solved.
        `depth`: the current depth of the recursion.
        `shots`: a shots collection to use for in-context learning.
        - If empty at any stage, the method will not use in-context learning (i.e., zero-shot).
        """

        # If the max. depth or the max. num. of nodes are reached, solve the problem directly
        if (
            depth >= self._config.max_depth
            or len(self._problems_cache) >= self._config.max_nodes
        ):
            try:
                output, info = self._model.generate(
                    self._construct_unit_context(problem, shots),
                    max_tokens=self._config.max_internal_tokens,
                )
                if output[0][0] is None:
                    self._logger.error(
                        f"[RecursivePrompting.generate:max_depth] The problem with ID '{problem.id}' was not solved."
                    )
                    problem.solution = ""
                else:
                    problem.solution = self._extract_answer(output[0][0])
                self._local_usage += info.usage
                self.usage += info.usage
            except Exception as e:
                self._logger.error(
                    {
                        f"[RecursivePrompting.generate:max_depth] The problem with ID '{problem.id}' could not be solved": None,
                        "Error source": "model",
                        "Error message": str(e),
                    }
                )
                output = [[""]]
                problem.solution = ""

            if not problem.is_solved:
                self._logger.error(
                    f"[RecursivePrompting.generate:max_budget] The problem with ID '{problem.id}' was not solved."
                )
                problem.solution = ""

            self._logger.debug(
                {
                    "[RecursivePrompting.generate:max_budget]": None,
                    "Depth": depth,
                    "N. of nodes": len(self._problems_cache),
                    "Model output": output[0][0],
                    "Problem ID": problem.id,
                    "Problem desc.": problem.description,
                    "Problem sol.": problem.solution,
                }
            )

            return

        # Split the problem into sub-problems
        subproblems_ids = self._split(problem, shots.split)

        # If it is a unit problem, solve it
        if len(subproblems_ids) == 0:
            try:
                output, info = self._model.generate(
                    self._construct_unit_context(problem, shots),
                    max_tokens=self._config.max_internal_tokens,
                )
                if output[0][0] is None:
                    self._logger.error(
                        f"[RecursivePrompting.generate:unit] The problem with ID '{problem.id}' was not solved."
                    )
                    problem.solution = ""
                else:
                    problem.solution = self._extract_answer(output[0][0])
                self._local_usage += info.usage
                self.usage += info.usage
            except Exception as e:
                self._logger.error(
                    {
                        f"[RecursivePrompting.generate:unit] The problem with ID '{problem.id}' could not be solved": None,
                        "Error source": "model",
                        "Error message": str(e),
                    }
                )
                output = [[""]]
                problem.solution = ""

            if not problem.is_solved:
                self._logger.error(
                    f"[RecursivePrompting.generate:unit] The problem with ID '{problem.id}' was not solved."
                )
                problem.solution = ""

            self._logger.debug(
                {
                    "[RecursivePrompting.generate:unit]": None,
                    "Depth": depth,
                    "Model output": output[0][0],
                    "Problem ID": problem.id,
                    "Problem desc.": problem.description,
                    "Problem sol.": problem.solution,
                }
            )

            return

        # Obtain merging instructions
        if self._config.elicit_instructions:
            instructions = self._get_instructions(problem, shots.instructions)
        else:
            instructions = ""

        # Solve each sub-problem
        while any(
            [
                not self._problems_cache[dep_id].is_solved
                for dep_id in problem.dependencies
            ]
        ):
            to_solve = []

            for dep_id in problem.dependencies:
                if not self._problems_cache[dep_id].is_solved and all(
                    self._problems_cache[subdep_id].is_solved
                    for subdep_id in self._problems_cache[dep_id].dependencies
                ):
                    to_solve.append(dep_id)

            if len(to_solve) == 0:
                raise RuntimeError(
                    "[RecursivePrompting.generate] The dependencies of the sub-problems contain a cycle."
                )

            for dep_id in to_solve:
                # TODO: we can solve sub-problems in parallel here
                dep = self._problems_cache[dep_id]
                self._solve(dep, depth + 1, shots)

        # Combine solutions to sub-problems to solve the original problem
        problem.solution = self._merge(problem, instructions, shots.merge)

        self._logger.debug(
            {
                "[RecursivePrompting.generate:complex]": None,
                "Depth": depth,
                "Problem ID": problem.id,
                "Problem desc.": problem.description,
                "Problem sol.": problem.solution,
                "Sub-problems desc.": [
                    self._problems_cache[id].description for id in subproblems_ids
                ],
                "Sub-problems sol.": [
                    self._problems_cache[id].solution for id in subproblems_ids
                ],
                "Instructions": instructions,
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
                instructions="",
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

    def _split(self, problem: _Problem, shots: list[tuple[str, str]] = []) -> list[str]:
        """
        Split a problem into sub-problems.

        ### Parameters
        ----------
        `problem`: the problem to be split.

        ### Returns
        -------
        A list of the IDs of the sub-problems to be solved.
        - If the list is empty, the problem is a unit problem.
        """

        shots_str = construct_shots_str(shots)

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
                    f"[RecursivePrompting.generate:split] The problem with ID '{problem.id}' could not be split": None,
                    "Error source": "model",
                    "Error message": str(e),
                }
            )
            split = ""
        if split is None:
            self._logger.error(
                f"[RecursivePrompting.generate] The problem with ID '{problem.id}' was not split."
            )
            split = ""

        subproblems_ids = self._parse_subproblems(split)

        problem.dependencies.extend(subproblems_ids)

        return subproblems_ids

    def _get_instructions(
        self, problem: _Problem, shots: list[tuple[str, str]] = []
    ) -> str:
        """
        Generate instructions to merge sub-solutions.

        ### Parameters
        ----------
        `problem`: the original problem.
        `subproblems`: the sub-problems to be solved.

        ### Returns
        -------
        Instructions to merge sub-solutions to solve the original problem.

        ### Raises
        -------
        `RuntimeError`: if the method is configured to elicit instructions, but no instructions prompt is provided.

        ### Notes
        ----------
        - The method will not generate instructions if `Config.elicit_instructions` is `False`.
        """

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
                    f"[RecursivePrompting.generate:instructions] The problem with ID '{problem.id}' could not generate instructions": None,
                    "Error source": "model",
                    "Error message": str(e),
                }
            )
            instructions = ""
        if instructions is None:
            self._logger.error(
                f"[RecursivePrompting.generate] Instructions for the problem with ID '{problem.id}' were not generated."
            )
            instructions = ""

        return instructions

    def _merge(
        self,
        problem: _Problem,
        instructions: str,
        shots: list[tuple[str, str]] = [],
    ) -> str:
        """
        Merge sub-solutions to solve the original problem.

        ### Parameters
        ----------
        `problem`: the original problem.
        `instructions`: the instructions to merge the sub-solutions.
        `shots`: a list of (input, target) pairs to use for in-context learning.

        ### Returns
        -------
        The solution to the original problem.
        """

        shots_str = construct_shots_str(shots)

        merge_prompt = self._merge_prompt.format(
            problem=problem.description,
            subsolutions=self._construct_dependencies_str(problem.dependencies),
            instructions=instructions,
            shots=shots_str,
        ).strip()
        merge_prompt = add_roles_to_context(
            merge_prompt,
            system_chars=self._config.merge_first_chars_as_system,
            assistant_chars=self._config.merge_last_chars_as_assistant,
        )

        try:
            output, info = self._model.generate(
                merge_prompt, max_tokens=self._config.max_internal_tokens
            )
            merge = self._extract_answer(output[0][0])
            self._local_usage += info.usage
            self.usage += info.usage
        except Exception as e:
            self._logger.error(
                {
                    f"[RecursivePrompting.generate:merge] The problem with ID '{problem.id}' could not be merged": None,
                    "Error source": "model",
                    "Error message": str(e),
                }
            )
            merge = ""

        # Insanity check to avoid infinite loops
        if merge is None:
            self._logger.error(
                f"[RecursivePrompting.generate] The problem with ID '{problem.id}' was not merged."
            )
            merge = ""

        return merge

    def _parse_subproblems(self, output: str) -> list[str]:
        """
        Parse the output of the model to find sub-problems.

        ### Parameters
        ----------
        `output`: the output of the model.

        ### Returns
        -------
        A list of the IDs of the sub-problems to be solved.
        - If the list is empty, the problem is a unit problem.
        """

        subproblems_dict: dict[str, RecursivePrompting._Problem] = {}
        detected_prefix = None

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
                    p = self._parse_raw_subproblem(line)

                    # Create global IDs; this is necessary because the IDs generated
                    # by the model are likely to repeat across independent splitting steps.
                    # The keys of the dictionary are still the local IDs.
                    if p.id in subproblems_dict:
                        self._logger.warn(
                            f"[RecursivePrompting.parse_subproblems] The ID '{p.id}' is repeated in the output. Incoming dependencies will be ignored."
                        )
                        local_id = self._id_gen.next()
                        global_id = local_id
                    else:
                        local_id = p.id
                        global_id = self._id_gen.next()

                    p.id = global_id
                    subproblems_dict[local_id] = p

                    break

        # Exceeding the maximum width
        if len(subproblems_dict) > self._config.max_width:
            log_msg = f"[RecursivePrompting.parse_subproblems] The number of sub-problems ({len(subproblems_dict)}) exceeds the max. width ({self._config.max_width})."
            if self._config.enforce_max_width:
                log_msg += " Truncating the list of sub-problems to the max. width."
                self._logger.error(log_msg)
                subproblems_dict = dict(
                    list(subproblems_dict.items())[: self._config.max_width]
                )
            else:
                self._logger.warn(log_msg)

        # Exceeding the maximum number of nodes
        elif len(subproblems_dict) + len(self._problems_cache) > self._config.max_nodes:
            n_accepted_subproblems = max(
                self._config.max_nodes - len(self._problems_cache), 0
            )
            self._logger.error(
                f"[RecursivePrompting.[parse_subproblems]] Adding the proposed sub-problems ({len(subproblems_dict)}) "
                f"to the existing amount of problems ({len(self._problems_cache)}) would exceed the max. "
                f"num. of nodes ({self._config.max_nodes}). Only the first {n_accepted_subproblems} "
                "sub-problems will be considered."
            )
            if n_accepted_subproblems < 1:
                subproblems_dict = {}
            else:
                subproblems_dict = dict(
                    list(subproblems_dict.items())[:n_accepted_subproblems]
                )

        # Substitute the local IDs with the global IDs in the dependencies
        for p_local_id, p in subproblems_dict.items():
            global_deps = []
            for dep_local_id in p.dependencies:
                if dep_local_id in subproblems_dict:
                    global_deps.append(subproblems_dict[dep_local_id].id)
                else:
                    self._logger.warn(
                        f"[RecursivePrompting.parse_subproblems] The dependency '{dep_local_id}' of the problem '{p_local_id}' does not exist. "
                        "It may have been removed if the maximum num. of nodes was exceeded. Ignoring the dependency."
                    )
            p.dependencies = global_deps

        # Add the sub-problems to the cache
        for p in subproblems_dict.values():
            self._add_to_cache(p)

        return [p.id for p in list(subproblems_dict.values())]

    def _parse_raw_subproblem(self, raw_problem: str) -> _Problem:
        """
        Parse a raw sub-problem string (given in a bullet point) to find the ID, description and dependencies between sub-problems.

        ### Parameters
        ----------
        `raw_problem`: the raw problem string.

        ### Returns
        -------
        The parsed problem's ID.

        ### Notes
        ----------
        - The raw sub-problem is expected to NOT contain the initial bullet point character.
        - The raw sub-problem will be stripped of surrounding whitespace.
        - The sub-problem is also stored in the cache.
        - The parsing will be done according to `Config.dependency_syntax`.
        """

        # Get rid of surrounding whitespace
        raw_problem = raw_problem.strip()

        if self._config.dependency_syntax == DependencySyntax.NONE:
            id = self._id_gen.random()
            desc = raw_problem
            deps_ids = []

        elif self._config.dependency_syntax == DependencySyntax.BRACKETS_PARENS:
            desc = raw_problem

            # ID
            try:
                if not desc.startswith("["):
                    raise ValueError
                right_bracket_i = desc.index("]")
                id = desc[1:right_bracket_i]
                desc = desc[right_bracket_i + 1 :]
            except ValueError:
                self._logger.error(
                    "[RecursivePrompting.generate] The ID of the problem could not be found for dependency syntax 'BRACKETS_PARENS'."
                )
                id = self._id_gen.random()

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

        return RecursivePrompting._Problem(id, desc, deps_ids)

    def _construct_dependencies_str(self, dependencies: list[str]) -> str:
        if len(dependencies) == 0:
            return ""

        return "".join(
            [
                f"- Sub-problem {i+1}: {dep.description}"
                + ("" if not dep.is_solved else f" Sub-solution: {dep.solution}\n")
                for i, dep in enumerate(
                    [self._problems_cache[id] for id in dependencies]
                )
            ]
        )[:-1]

    def _generate_graph(
        self, root_id: str
    ) -> dict[str, str | dict[str, dict[str, str]]]:
        """
        Generate a graph of the problem-solving process.

        ### Parameters
        ----------
        `root`: the ID of the root problem of the graph.

        ### Returns
        -------
        A dictionary representing the graph.
        """

        nodes = {}
        for id, problem in self._problems_cache.items():
            nodes[id] = {
                "description": problem.description,
                "dependencies": problem.dependencies,
                "solution": problem.solution,
            }

        return {"root_id": root_id, "nodes": nodes}

    def train(self, _):
        raise NotImplementedError("RecursivePrompting does not support training.")
