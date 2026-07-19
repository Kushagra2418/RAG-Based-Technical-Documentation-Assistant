# LangGraph Reference Guide

LangGraph is a library for building stateful, multi-step applications with large language
models. It models an application as a graph: nodes perform work, edges define control
flow, and a shared state object is threaded through every step. It is well suited to
agentic and self-corrective workflows where the path taken depends on intermediate
results.

## Core Concepts

- **State**: A shared data structure passed between nodes. It is typically a `TypedDict`
  and represents the current snapshot of the application.
- **Node**: A Python function that receives the current state and returns a partial update
  to it.
- **Edge**: A connection that determines which node runs next. Edges can be fixed or
  conditional.

## Defining State

State is usually a `TypedDict`. Each node returns a dictionary containing only the keys it
wants to update; LangGraph merges those updates into the running state:

```python
from typing import TypedDict, List

class State(TypedDict):
    question: str
    documents: List[str]
    answer: str
    retries: int
```

By default, when a node returns a key, the new value replaces the old one. To accumulate
values instead of replacing them, annotate the field with a reducer using `Annotated`,
for example `Annotated[list, operator.add]` to append to a list.

## Building the Graph

Create a `StateGraph`, add nodes, wire edges, and compile:

```python
from langgraph.graph import StateGraph, START, END

def retrieve(state: State) -> dict:
    docs = search(state["question"])
    return {"documents": docs}

def generate(state: State) -> dict:
    answer = llm_answer(state["question"], state["documents"])
    return {"answer": answer}

builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
```

`START` and `END` are special sentinel nodes marking the entry and exit points of the
graph.

## Running the Graph

Invoke the compiled graph with an initial state. The result is the final state after all
nodes have run:

```python
result = graph.invoke({"question": "What is LangGraph?", "retries": 0})
print(result["answer"])
```

Use `stream` instead of `invoke` to receive the state after each node, which is useful for
showing progress or debugging.

## Conditional Edges

A conditional edge routes to different nodes based on a function that inspects the state.
The function returns a key that is mapped to a destination node:

```python
def decide(state: State) -> str:
    if state["documents"]:
        return "generate"
    return "rewrite"

builder.add_conditional_edges(
    "grade",
    decide,
    {"generate": "generate", "rewrite": "rewrite_query"},
)
```

Conditional edges are the mechanism for self-correction: a grading node evaluates
intermediate results, and the routing function decides whether to proceed, retry, or fall
back.

## Cycles and Loops

Unlike a simple pipeline, LangGraph supports cycles. An edge can point back to an earlier
node, creating a loop — for example, rewriting a query and retrieving again. To prevent
infinite loops, keep a counter in the state and check it in the routing function:

```python
def decide(state: State) -> str:
    if state["documents"]:
        return "generate"
    if state["retries"] < 2:
        return "rewrite"
    return "give_up"
```

A node in the loop increments the counter (`return {"retries": state["retries"] + 1}`) so
the routing function can enforce a maximum number of attempts.

## Persistence and Memory

A compiled graph can be given a checkpointer to persist state between invocations. With a
checkpointer, you pass a `thread_id` in the configuration, and LangGraph reloads the
matching state, which enables multi-turn conversations and human-in-the-loop workflows.

## Why Use a Graph

Expressing a workflow as an explicit graph makes the control flow visible and testable.
Each node is an isolated, unit-testable function, the state schema documents exactly what
data flows through the system, and conditional edges make branching logic explicit rather
than hidden inside imperative code.
