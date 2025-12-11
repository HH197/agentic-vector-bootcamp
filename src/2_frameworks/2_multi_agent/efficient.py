"""Example code for planner-worker agent collaboration.

With reference to:

github.com/ComplexData-MILA/misinfo-datasets
/blob/3304e6e/misinfo_data_eval/tasks/web_search.py
"""

import asyncio
import contextlib
import signal
import sys

import os
import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.prompts import REACT_INSTRUCTIONS, RETRIEVER_INSTRUCTIONS, PLANNER_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

set_up_logging()

AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}

configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="cibc_2",
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

#----------------------------agents------------------

# Worker Agent: handles long context efficiently
worker_agent = agents.Agent(
    name="WorkerAgent",
    instructions=(
        "You perform a single knowledge search query using the knowledgebase tool "
        "and return raw search results. Do NOT summarize; the retriever will handle that."
    ),
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)

# Retriever Agent: orchestrates multiple queries and cleaning
retriever_agent = agents.Agent(
    name="RetrieverAgent",
    instructions=RETRIEVER_INSTRUCTIONS,
    tools=[
        worker_agent.as_tool(
            tool_name="kb_search",
            tool_description="Run a knowledge base query and return raw results."
        )
    ],
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"],
        openai_client=async_openai_client
    ),
)

# Main Agent: more expensive and slower, but better at complex planning
main_agent = agents.Agent(
    name="MainAgent",
    instructions=PLANNER_INSTRUCTIONS,
    # Allow the planner agent to invoke the worker agent.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        retriever_agent.as_tool(
            tool_name="retrieve",
            tool_description="Retrieve multiple relevant snippets from the KB."
        )
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
    ),
)

#---------------------- end of agents -------------

async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    # Use the main agent as the entry point- not the worker agent.
    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace2") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)

        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

            # # avoid duplicates
            # if new_msgs:
            #     gr_messages.extend(new_msgs)
            #     yield gr_messages

        span.update(output=result_stream.final_output)


demo = gr.ChatInterface(
    _main,
    title="CIBC: Smart Server",
    type="messages",
    examples=[
        "Introduce me of CIBC credit cards",
        "Which CIBC credit cards are best for students?",
    ],
)

if __name__ == "__main__":

    # key_value = os.getenv("WEAVIATE_API_KEY")
    # print("Value of YOUR_KEY:", key_value)
    
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
