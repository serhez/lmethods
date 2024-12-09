import numpy as np

from lmethods.protocols import Logger, Model
from lmethods.utils.decomposition import HierarchySyntax
from lmethods.utils.usage import Usage


def construct_self_consistency_context(
    query: str, responses: list[str], syntax: HierarchySyntax
) -> str:
    """
    Returns a self-consistency context to prompt the model to choose the most common response to a query or, in case all responses are different, the best reponse as chosen by the model.

    ### Parameters
    --------------
    `query`: the query that was answered by the model.
    `responses`: a list of responses to the query.

    ### Returns
    --------------
    A string with the context for the self-consistency task.

    ### Raises
    --------------
    `AssertionError`: if the list of responses is empty.
    """

    assert len(responses) != 0, "The list of responses must have at least one element."

    if syntax == HierarchySyntax.BULLET_POINTS:
        context = f"Choose the response that better represents the most common response. If they are all different, choose the response that best solves the query.\n\nQuery: {query}\n"
        for i, response in enumerate(responses):
            context += f"- Response {i}: {response}\n"
        context += "\nThe response I choose is the one with number "

    elif syntax == HierarchySyntax.MARKDOWN_HEADERS:
        context = f"Choose the response that better represents the most common response. If they are all different, choose the response that best solves the query."
        context += f"\n\n# Query: {query}\n\n"
        for i, response in enumerate(responses):
            context += f"# Response {i}\n\n{response}\n\n"
        context += "# Chosen response\n\nThe response I choose is the one with number "

    return context


def parse_self_consistency_output(output: str) -> int:
    """
    Parses the output of a self-consistency task to return the chosen response's index.
    If the output is not valid, returns 0.

    ### Parameters
    --------------
    `output`: the output of the self-consistency task.

    ### Returns
    --------------
    The index of the chosen response.
    """

    try:
        parsed_output = ""
        for c in output:
            if c.isnumeric():
                parsed_output += c
            elif parsed_output != "":
                break

        return int(parsed_output)

    except ValueError:
        return 0


def choose_response_via_sc(
    model: Model,
    context: str,
    responses: list[str],
    syntax: HierarchySyntax,
    max_n_per_call: int = 10,
    logger: Logger | None = None,
) -> tuple[int, Usage]:
    """
    Choose the best response via Self-Consistency (X. Wang et al., 2022) with recursive binary search.

    ### Parameters
    ----------
    `model`: the model to use for the task.
    `context`: the context of the problem.
    `responses`: the responses to choose from.
    `syntax`: the syntax of the responses.
    `max_n_per_call`: the maximum number of responses to include in a single call to the model.
    `logger`: a logger to record the task's progress and any errors.

    ### Returns
    ----------
    A tuple containing:
    - The index of the chosen response.
    - The usage statistics of the model during Self-Consistency.

    ### Notes
    ----------
    Binary search will be performed when `max_n_per_call > len(responses)` by partitioning the set of responses recursively.
    - This mechanism can be used for large number of responses which do not fit in the context size of the underlying model.
    """

    usage = Usage()
    choices = responses

    if len(responses) == 1:
        return 0, usage

    if len(responses) > max_n_per_call:
        # Divide into two partitions of equal size
        partitions = [p.tolist() for p in np.array_split(responses, 2)]
        chosen_idxs = []
        for p in partitions:
            i, u = choose_response_via_sc(
                model, context, p, syntax, max_n_per_call, logger
            )
            chosen_idxs.append(i)
            usage += u

        choices = [partitions[i][idx] for i, idx in enumerate(chosen_idxs)]

    input = construct_self_consistency_context(context, choices, syntax)
    try:
        output, info = model.generate(
            input,
            max_tokens=len(str(len(responses))),
            temperature=0.0,
            use_context_template=False,  # we want the old-school inference to get a single number as the next token
        )
        usage += info.usage
        chosen_idx = parse_self_consistency_output(output[0][0])
        if chosen_idx < 0 or chosen_idx >= len(choices):
            if logger is not None:
                logger.error(
                    {
                        "[MetaPrompting.self-consistency] Invalid index when choosing the response via Self-Consistency": None,
                        "N. responses to choose from": len(responses),
                        "Self-Consistency output": output[0][0],
                        "Parsed chosen index": {chosen_idx},
                        "Corrective action": "The first response will be chosen instead",
                    }
                )
            chosen_idx = 0
    except Exception as e:
        if logger is not None:
            logger.error(
                {
                    "[MetaPrompting.self-consistency] The model failed to generate a response": None,
                    "Error": str(e),
                }
            )
        output = None
        chosen_idx = 0

    if logger is not None:
        logger.debug(
            {
                "[MetaPrompting.Self-Consistency]": None,
                "Input": input,
                "Output": output,
                "Chosen index": chosen_idx,
                "Total usage": usage,
            }
        )

    return chosen_idx, usage
