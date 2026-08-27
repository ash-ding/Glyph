"""What the agent is told.

The system prompt states the protocol and the containers this arm may spend
on, and stops there.  It does not hint at what the right answer is: A7's whole
result is whether the agent finds the boundary on its own, and a prompt that
nudges it toward weights or toward code would be measuring our prior instead
of its judgement.  The single-path arms get the same text with one container
named, so the only difference between arms is which tools exist.
"""

from __future__ import annotations

from .schema import Container

SYSTEM = """You are given a task you cannot answer from what you already know: \
a small language whose syntax is public and whose semantics are private.

You have a preparation phase with a fixed compute budget. During it you may \
buy information by querying a hidden interpreter. When you are done you seal \
an artifact, and that artifact -- not you -- answers {n_test} expressions you \
will never see. After sealing you are gone: no oracle, no further training, no \
second look.

{container_text}

Facts worth planning around:
- The value space holds {n_values} values. Unary operators are maps over it; \
binary operators are maps over pairs, so their table is roughly {n_pairs} \
entries. You cannot buy either table outright.
- Your budget buys roughly {q_hint} queries if you spend it all on queries, \
and you need some of it for whatever you build. The test set is much larger \
than anything you can query, so the artifact has to answer far more than you \
asked.
- Values are written with their digits spelled out. Whether that structure \
means anything is for you to find out.

Spend the budget as you see fit. Call `seal` before it runs out -- if it runs \
out first you seal with whatever exists, which is usually worse."""

CONTAINERS = {
    frozenset({Container.CONTEXT}): (
        "This run may only put capability into **context**: the artifact is a "
        "prompt prefix that a small model carries in front of every test "
        "query. Its token count is recorded and re-paid on each of those "
        "queries."),
    frozenset({Container.CODE}): (
        "This run may only put capability into **code**: the artifact is a "
        "Python program that answers each expression on its own. It runs "
        "sandboxed, with no network and no model."),
    frozenset({Container.WEIGHTS}): (
        "This run may only put capability into **weights**: the artifact is a "
        "small model you fine-tune on data you build from what you queried. "
        "At test time it answers with no prompt and no program."),
}

FREE_CHOICE = (
    "This run may put capability into **context, code, weights, or any "
    "combination**. Writing code and training draw on the same budget, so "
    "they compete. What to spend on is your decision and it is the thing "
    "being measured.")


def system_prompt(allowed: set, *, n_test: int, n_values: int,
                  q_hint: int) -> str:
    key = frozenset(allowed)
    text = CONTAINERS.get(key, FREE_CHOICE)
    return SYSTEM.format(container_text=text, n_test=n_test, n_values=n_values,
                         n_pairs=f"{n_values ** 2:,}", q_hint=q_hint)


def opening_message(spec: str, demos: str, n_demo: int) -> str:
    return (f"{spec}\n\n"
            f"Here are {n_demo} worked examples, free of charge:\n\n{demos}\n\n"
            "Begin. Query when you want to learn something, and seal when you "
            "are ready.")


SATURATED = """You are answering expressions in a small language whose syntax \
is public and whose semantics are private.

{spec}

Below is every query anyone has made against the hidden interpreter, with its \
answer. This is the complete purchased record -- there is no more to be had.

{evidence}

Using only that, evaluate the following {n} expressions.

{items}

Answer with exactly {n} lines of the form

  <number>: <result>

written the way results are written above. Give your best guess for every \
one; do not skip any and do not explain."""
