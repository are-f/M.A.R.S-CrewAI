import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain import hub
from crewai import Agent

load_dotenv(dotenv_path="../../.env")

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
def qna_agent(user_input: str) -> str:
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    
    search_tool = DuckDuckGoSearchRun()
    tool = Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to look up information on the web ONLY when needed."
    )

    return Agent(role="QnA_Agent",
                 goal="To answer user query using llm and search tool. Use search tool ONLY when needed.",
                 tools=[tool],
                 memory=memory,
                 verbose=True)
