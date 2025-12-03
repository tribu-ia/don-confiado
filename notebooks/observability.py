# %% [markdown]
"""
# AI Agents Observability with LangSmith

## 1. The "Why": Why do we need Observability?

Building AI agents is fundamentally different from traditional software engineering. In traditional software:
- **Deterministic:** Input A always leads to Output B.
- **Traceable:** You can step through code line-by-line and know exactly what happened.

In AI Engineering (and specifically with LLMs):
- **Stochastic:** The same input might produce different outputs.
- **Opaque:** The "reasoning" happens inside a black box (the model).
- **Complex Chains:** Agents involve multiple steps, tools, and loops. If an agent fails, did it fail because:
    1. The retrieval step missed the document?
    2. The prompt was ambiguous?
    3. The model hallucinated?
    4. The tool output was malformed?

**Observability** gives us X-ray vision into these complex interactions. It allows us to:
- **Debug:** Pinpoint exactly which step in a chain failed.
- **Evaluate:** Measure performance over time (latency, cost, quality).
- **Iterate:** confidentially improve prompts and architectures based on real data.
"""

# %% [markdown]
"""
## 2. The Basics: How LangSmith Works

LangSmith is a platform by LangChain specifically designed for LLM application development. Its core mechanism is **Tracing**.

### Key Concepts:
- **Trace:** A record of a single execution of your application (e.g., one user query).
- **Run:** An individual step within a trace (e.g., a call to OpenAI, a tool execution, a retrieval step).
- **Project:** A collection of traces (usually corresponding to a specific application or environment).

### Setup Requirements:
To use LangSmith, you typically only need to set environment variables. No massive code changes are required if you are using LangChain or LangGraph.

1. Get an API key from [smith.langchain.com](https://smith.langchain.com).
2. Set `LANGCHAIN_TRACING_V2=true`.
3. Set `LANGCHAIN_API_KEY=<your-key>`.
"""

# %% [markdown]
"""
## 3. Implementation Example: A Simple LangGraph Agent

Let's build a simple agent using **LangGraph** and observe it with **LangSmith**.

This agent will be a simple "Math Assistant" that can perform calculations. We will see how LangSmith captures the agent's "thought process".
"""

# %%
# Step 1: Install necessary libraries (if running in Colab/local)
#!pip install -qU langgraph langchain-openai langsmith python-dotenv

# %%
import os
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Step 2: Configure Environment Variables for LangSmith
# This is the CRITICAL step for observability.

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Ensure OpenAI Key is set
# _set_env("OPENAI_API_KEY") # Not using OpenAI for this example

# Ensure Gemini Key is set
_set_env("GOOGLE_API_KEY")

# Ensure LangSmith config is set
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Don Confiado Class - Observability" # Optional: Name your project
_set_env("LANGCHAIN_API_KEY")

print("Observability configured!")

# %% [markdown]
"""
### Defining Tools and Model

We'll create a simple tool for multiplication to demonstrate tool usage in the trace.
"""

# %%
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define a tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

tools = [multiply, add]

# 2. Initialize the Model (LLM) with tools bound to it
# We use Gemini 2.5 Flash as it is fast and cost-effective
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# %% [markdown]
"""
### Building the LangGraph

We will build a simple "ReAct" style graph:
1. **Agent Node:** Calls the LLM to decide what to do.
2. **Tools Node:** Executes the tools if the LLM asks for them.
3. **Edge:** Loops back to the Agent after tool execution.
"""

# %%
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# 1. Define the Graph State
# MessagesState is a built-in TypedDict with a "messages" key
class AgentState(MessagesState):
    pass

# 2. Define the Nodes

def agent_node(state: AgentState):
    messages = state["messages"]
    # Invoke the model
    response = llm_with_tools.invoke(messages)
    # Return the new message to append to history
    return {"messages": [response]}

# ToolNode is a prebuilt node in LangGraph that handles tool execution
tool_node = ToolNode(tools)

# 3. Define the Conditional Edge Logic
def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM returned tool_calls, go to "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end
    return END

# 4. Construct the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # Loop back to agent after tools

app = workflow.compile()

# Visualize the graph (ASCII art version)
print(app.get_graph().draw_ascii())

# %% [markdown]
"""
## 4. Running the Agent and Generating Traces

Now we run the agent. Because we set `LANGCHAIN_TRACING_V2="true"`, this execution will automatically be logged to LangSmith.
"""

# %%
# Let's ask a question that requires multiple steps (reasoning + tools)
query = "What is 15 times 13, and what do you get if you add 20 to that result?"

print(f"User Query: {query}\n")

inputs = {"messages": [HumanMessage(content=query)]}

# Stream the output to see the steps
for chunk in app.stream(inputs, stream_mode="values"):
    message = chunk["messages"][-1]
    if hasattr(message, "tool_calls") and message.tool_calls:
        print(f"🤖 Agent decides to call tool: {message.tool_calls[0]['name']} with args {message.tool_calls[0]['args']}")
    elif hasattr(message, "content") and message.content:
         print(f"🗣️ Output: {message.content}")

# %% [markdown]
"""
## 5. Analyzing the Trace

Go to [smith.langchain.com](https://smith.langchain.com) and look at the "Don Confiado Class - Observability" project.

### What to look for:

1. **The Graph Structure:** You will see the visual representation of `agent` -> `tools` -> `agent`.
2. **Token Usage:** Click on the `ChatGoogleGenerativeAI` run to see exactly how many input/output tokens were used (and the cost).
3. **Latency:** How long did the tool call take vs the LLM generation?
4. **Inputs/Outputs:**
    - See the raw input prompt sent to the LLM (including system messages).
    - See the exact output from the `multiply` tool.

### Debugging Example
If the answer was wrong, you could check:
- Did the tool return the wrong number? (Tool error)
- Did the LLM call the tool with wrong arguments? (Model error)
- Did the final synthesis ignore the tool output? (Context window/Model error)

This visibility is what makes LangSmith essential for AI Engineering.
"""
