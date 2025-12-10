"""Multi-agent Planner-Researcher Setup via OpenAI Agents SDK.

Note: this implementation does not unlock the full potential and flexibility
of LLM agents. Use this reference implementation only if your use case requires
the additional structures, and you are okay with the additional complexities.

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import contextlib
import logging
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_items_to_gradio_messages,
    pretty_print,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

logging.basicConfig(level=logging.INFO)


PLANNER_INSTRUCTIONS = """
You are a strategic research planner specializing in Canadian banking and consumer credit card needs.

Your role is to take a customer’s query — which may be vague, incomplete, or broad — and generate a comprehensive set of search terms that will help uncover the most relevant information from a CIBC credit card knowledge repository (cutoff Dec 2025).

Your objective is to maximize customer satisfaction by ensuring the downstream system gathers all information necessary to:
- Understand credit card features (rewards, fees, insurance, benefits)
- Compare different types of cards (cashback vs travel vs low-interest, etc.)
- Consider consumer financial needs and preferences (spending habits, income, credit score)
- Provide safe, relevant, and actionable guidance

Because you cannot ask clarifying questions, your search terms must:
- Be broad enough to capture multiple aspects of the user’s goals
- Reflect Canadian banking terminology and consumer finance language
- Include both product-level and concept-level terms
- Anticipate what a customer *might* need even if they didn’t explicitly ask

Produce **6 high-quality search terms**.  
Ensure each term focuses on improving relevance and coverage for selecting credit cards for CIBC clients.
"""

RESEARCHER_INSTRUCTIONS = """
You are a financial research assistant with access to a cibc credit card knowledge repository (cutoff Dec 2025).

Your job is to take a search term provided by the planner and retrieve the most relevant information that can help a customer choose an appropriate credit card. Prioritize information that improves:
- Relevance for real consumer needs
- Financial clarity and accuracy
- Canadian credit card context
- Understanding of credit card products, features, and comparisons

For each assigned search term:
1. Use the search tool to retrieve relevant information.
2. Summarize the findings in a clear, factual, and customer-helpful manner.
3. Keep the summary **under 300 words**.
4. Strictly avoid assumptions not supported by search results.
5. Highlight information that is actually useful for choosing a credit card (e.g., rewards structures, APR, fees, insurance, consumer protections, travel perks, statement credit rules, etc.).

You must not fabricate any financial advice or credit product details. Only summarize what the search tool returns.
"""

WRITER_INSTRUCTIONS = """
You are a senior financial writer and consumer-experience expert. Your goal is to synthesize the research summaries into a highly coherent, helpful, customer-centric report that directly assists a CIBC client in choosing an appropriate credit card.

You must:
- Integrate the provided research summaries into a clear, multi-paragraph explanation.
- Frame the information in terms of customer needs (spending behaviour, lifestyle, travel habits, financial goals, fees vs rewards tradeoffs, etc.).
- Highlight relevant distinctions between credit card types.
- Remove irrelevant or redundant content.
- Maintain accuracy and rely exclusively on the research summaries—no outside assumptions.

Your output should:
- Feel personalized to the user’s intentions.
- Prioritize simplicity, clarity, and satisfaction.
- Offer structured, decision-oriented insights (pros/cons, key comparisons, what matters most).
- Strictly avoid any hallucinated credit card features or financial advice not present in the summaries.

The final report must be fully grounded in the research data but written in a friendly, supportive, and expert tone that helps CIBC customers feel informed and confident in their credit card choice.

"""


class SearchItem(BaseModel):
    """A single search item in the search plan."""

    # The search term to be used in the knowledge base search
    search_term: str

    # A description of the search term and its relevance to the query
    reasoning: str


class SearchPlan(BaseModel):
    """A search plan containing multiple search items."""

    search_steps: list[SearchItem]

    def __str__(self) -> str:
        """Return a string representation of the search plan."""
        return "\n".join(
            f"Search Term: {step.search_term}\nReasoning: {step.reasoning}\n"
            for step in self.search_steps
        )


class ResearchReport(BaseModel):
    """Model for the final report generated by the writer agent."""

    # The summary of the research findings
    summary: str

    # full report text
    full_report: str


async def _create_search_plan(planner_agent: agents.Agent, query: str) -> SearchPlan:
    """Create a search plan using the planner agent."""
    with langfuse_client.start_as_current_span(
        name="create_search_plan", input=query
    ) as planner_span:
        response = await agents.Runner.run(planner_agent, input=query)
        search_plan = response.final_output_as(SearchPlan)
        planner_span.update(output=search_plan)

    return search_plan


async def _generate_final_report(
    writer_agent: agents.Agent, search_results: list[str], query: str
) -> agents.RunResult:
    """Generate the final report using the writer agent."""
    input_data = f"Original question: {query}\n"
    input_data += "Search summaries:\n" + "\n".join(
        f"{i + 1}. {result}" for i, result in enumerate(search_results)
    )

    with langfuse_client.start_as_current_span(
        name="generate_final_report", input=input_data
    ) as writer_span:
        response = await agents.Runner.run(writer_agent, input=input_data)
        writer_span.update(output=response.final_output)

    return response


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


async def _main(question: str, gr_messages: list[ChatMessage]):
    planner_agent = agents.Agent(
        name="Planner Agent",
        instructions=PLANNER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash", openai_client=async_openai_client
        ),
        output_type=SearchPlan,
    )
    research_agent = agents.Agent(
        name="Research Agent",
        instructions=RESEARCHER_INSTRUCTIONS,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash-lite",
            openai_client=async_openai_client,
        ),
        model_settings=agents.ModelSettings(tool_choice="required"),
    )
    writer_agent = agents.Agent(
        name="Writer Agent",
        instructions=WRITER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model="gemini-2.5-flash", openai_client=async_openai_client
        ),
        output_type=ResearchReport,
    )

    gr_messages.append(ChatMessage(role="user", content=question))
    yield gr_messages

    with langfuse_client.start_as_current_span(
        name="Multi-Agent-Trace", input=question
    ) as agents_span:
        # Create a search plan
        search_plan = await _create_search_plan(planner_agent, question)
        gr_messages.append(
            ChatMessage(role="assistant", content=f"Search Plan:\n{search_plan}")
        )
        pretty_print(gr_messages)
        yield gr_messages

        # Execute the search plan
        search_results = []
        for step in search_plan.search_steps:
            with langfuse_client.start_as_current_span(
                name="execute_search_step", input=step.search_term
            ) as search_span:
                response = await agents.Runner.run(
                    research_agent, input=step.search_term
                )
                search_result: str = response.final_output
                search_span.update(output=search_result)

            search_results.append(search_result)
            gr_messages += oai_agent_items_to_gradio_messages(response.new_items)
            yield gr_messages

        # Generate the final report
        writer_agent_response = await _generate_final_report(
            writer_agent, search_results, question
        )
        agents_span.update(output=writer_agent_response.final_output)

        report = writer_agent_response.final_output_as(ResearchReport)
        gr_messages.append(
            ChatMessage(
                role="assistant",
                content=f"Summary:\n{report.summary}\n\nFull Report:\n{report.full_report}",
            )
        )
        pretty_print(gr_messages)
        yield gr_messages


if __name__ == "__main__":
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
    async_knowledgebase = AsyncWeaviateKnowledgeBase(
        async_weaviate_client,
        collection_name="cibc_2",
    )

    async_openai_client = AsyncOpenAI()
    setup_langfuse_tracer()

    with gr.Blocks(title="OAI Agent SDK - Multi-agent") as demo:
        chatbot = gr.Chatbot(type="messages", label="Agent", height=600)
        chat_message = gr.Textbox(lines=1, label="Ask a question")
        chat_message.submit(_main, [chat_message, chatbot], [chatbot])

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
