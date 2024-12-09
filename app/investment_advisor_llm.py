import os

from langchain_openai import ChatOpenAI

from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai import Agent, Task, Crew
from crewai.process import Process

from langchain.agents import Tool
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from pydantic import BaseModel, Field

import yfinance as yf

import mistune

# Pull the OpenAI key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not set in environment variables.")
# Pull the Serper key from the environment variable
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    raise ValueError("Serper API Key not set in environment variables.")
    
    
def get_llm(temperature, model):
  return ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=model,
    temperature=temperature
  )
llm = get_llm(temperature=0, model="gpt-4o")

 
#===========================================================================================
# TOOLS
#===========================================================================================

# Initialize the web search and scrape tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define the RAG capabilities (accessing investment-related papers)

# Load the investment-related papers
loader = DirectoryLoader("docs", glob="**/*", loader_cls=PyPDFLoader)
documents = loader.load()

# Split the article into chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
chunks = text_splitter.split_documents(documents)

# Store the chunks as embeddings within a vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Initialize the OpenAI instance and set up a chain for Q&A from an LLM and a vector score
rag_llm = get_llm(temperature=0, model="gpt-4o")
retriever = vector_store.as_retriever()
rag_chain = RetrievalQA.from_chain_type(rag_llm, retriever=retriever)

# Define the input for the RAG tool
class rag_toolInput(BaseModel):
    rag_chain: RetrievalQA = Field(..., description="")
    query: str = Field(..., description="")
# Function for accessing investment strategy-related papers
def access_papers(inputs: rag_toolInput) -> str:
    """Useful for answering questions by accessing investment-related papers."""
    result = inputs.rag_chain({"query": inputs.query})
    return result['result']
# Store the function in the globals() dictionary
globals()['access_papers'] = access_papers
# Create a RAG Tool
rag_tool = Tool(
    name="rag_tool",
    func=globals()['access_papers'],
    description="Useful for answering questions by retrieving information from investment-related papers.",
    input_schema=rag_toolInput
) 


#===========================================================================================
# AGENTS
#===========================================================================================

# Define the Investment Manager
investment_manager = Agent(
    role="Investment Manager",
    goal="Determine what type of issue or inquiry the client has based on their query: {user_query} and coordinate with appropriate investment specialists to provide advice",
    backstory="You are an experienced investment advisor working with a team of investment specialists "
              "that focus on the U.S. equity market along with basic financial planning frameworks. "
              "You are the first point of contact for clients. These clients may have "
              "specific questions about investment strategies, questions about their existing "
              "portfolio, or may want general information related to market news or investing principles. "
              "You dynamically delegate tasks to your investment specialists. "
              "You engage with clients in a polite and friendly manner. You have intimate knowledge "
              "of the expertise of your investment specialists, soliciting information from them to "
              "provide answers to client questions.",
    allow_delegation=False,
    verbose=False,
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    llm = "gpt-4o"
)

# Define the Basic Financial Advisor
financial_advisor = Agent(
    role="Basic Financial Advisor",
    goal="Provide basic financial advice related to client objectives and constraints",
    backstory="You are a basic financial advisor on an investment management team. "
              "You receive instructions on what to do from the Investment Manager. "
              "You provide basic advice about investment strategies that can be used by clients "
              "to reach their desired goals. For example, a client may want an aggressive strategy "
              "that allows them to make a certain profit over 10 years given a certain disposable "
              "income level. You pay close attention to the clarity of your advice, providing "
              "recommendations as easy to follow steps that a lay person would understand. "
              "You are cognizant of the risks associated with certain recommendations and "
              "acknolwedge the uncertainty in achieving the stated objectives. ",
    tools=[search_tool, scrape_tool, rag_tool],
    allow_delegation=False,
    verbose=False,
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    llm = "gpt-4o"
)

# Topic Definition Agent
topic_agent = Agent(
    role="Topic Definition Agent",
    goal="Identify and filter relevant topics for news coverage.",
    backstory="You help the investment management team identify specific topics or entities that are referenced  "
              "by the given client. These could include specific references to companies, industries, or macro events. "
              "You receive instructions on what to do from the Investment Manager. ",
    allow_delegation=False,
	  verbose=False,
    llm = "gpt-4o"
)

# Define the News Coverage Specialist
news_agent = Agent(
    role="News Coverage Specialist",
    goal="Gather and summarize relevant news coverage for provided topics",
    backstory="You receive a list of topics from the Topic Definition Agent and gather news coverage "
              "related to those topics. The coverage is summarized to highligh pertinent information. "
              "Relevant news coverage could include macro-trend news from economics journals, "
              "analysis of earnings reports, firm-specific news, or industry-specific news from  "
              "financial outlets. The news coverage you provide supplements advice provided "
              "by other investment specialists and helps inform clients about potential opportunites "
              "or risks. You are cognizant of the sources you use, ensuring that you pull information "
              "from reputable sources. You make sure to provide references to the sources used.",
    tools=[search_tool, scrape_tool],
    allow_delegation=False,
	  verbose=False,
    llm = "gpt-4o"
)

# Define the Quantitative Investment Specialist
quant_researcher = Agent(
    role="Quant Investment Specialist",
    goal="You provide advice related to quantiative investment strategies",
    backstory="You are an investment specialist that focuses on quantitative strategies. "
              "You receive instructions on what to do from the Investment Manager. "
              "You rely on the existing literature that shows strong support for different investment strategies "
              "including momentum, quality, and size related strategies. You are intimately familiar "
              "with the foundational investment knowledge (e.g. the efficient frontier, tangency portfolio, "
              "the CAPM, the Fama-French Five Factor Model, trading costs). You provide clear explanations "
              "of technical concepts when necessary. You either provide information related to general inquiries "
              "or advice related to investment decisions.",
    tools=[search_tool, scrape_tool, rag_tool],
    allow_delegation=False,
	  verbose=False,
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    llm = "gpt-4o"
)

# Define the Quantitative Investment Reviewer
quant_reviewer = Agent(
    role="Quantitative Investment Reviewer",
    goal="You review and revise the technical investment-related information or advice",
    backstory="You review and revise investment-related information or advice that is provided by "
              "the Quant Investment Specialist. You ensure that the information provided is clear and "
              "concise so that a lay person with basic investing knowledge can understand the main principles. "
              "You assess the validity of the statements and comment on whether or not there are prevailing "
              "competing theories on particular topics.",
    tools=[search_tool, scrape_tool, rag_tool],
    allow_delegation=False,
	  verbose=False,
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    llm = "gpt-4o"
)

# Define the Data Analyst
data_analyst = Agent(
    role="Data Analyst",
    goal="You pull stock-related data from Yahoo Finance",
    backstory="You are adept at retrieving financial data from Yahoo Finance. When given a stock "
              "(ticker symbol or company name) you provide relevant information like price, volume, and market cap. "
              "You receive instructions on what to do from the Investment Manager or other investment specialists.",
    allow_delegation=False,
	  verbose=False,
    llm = "gpt-4o"
)

# Define the Information Aggregator
info_aggregator = Agent(
    role="Info Aggregator",
    goal="You aggregate information from the investment specialists and respond to the client query: {user_query}",
    backstory="You are an investment advisor that recieves information from different investment specialists. "
              "You collate this information into presentable analyses and recommendations for the client."
              "You are adept at consolidating this information into a response that a lay person would understand. "
              "You engage with clients in a polite and friendly manner.",
    allow_delegation=False,
	  verbose=False,
    llm = "gpt-4o"
)


#===========================================================================================
# TASKS
#===========================================================================================

# Define the initial task for handling user queries for Investment Manager
manager_task = Task(
    description="Analyze the user's query: {user_query} and delegate tasks for each agent aside from the Info Aggregator.",
    expected_output="A list of agents where each agent has assigned tasks to complete. "
                    "The agents that you can provide tasks to are listed here: "
                    "Basic Financial Advisor, Topic Definition Agent, Quant Investment Specialist, Data Analyst",
    agent=investment_manager,
)

# Task for Basic Financial Advisor
financial_advisor_task = Task(
    description="Analyze the instructions from the Investment Manager and provide a detailed response.",
    expected_output="A detailed response pertaining to the assigned tasks.",
    context=[manager_task],
    agent=financial_advisor,
)

# Task for Topic Definition Agent
define_topics = Task(
    description="Analyze the user query: {user_query} and identify relevant topics for news coverage.",
    expected_output="A list of specific topics deemed relevant for news coverage.",
    agent=topic_agent,
)

# Task for News Coverage Specialist
fetch_news = Task(
    description="Fetch and summarize recent financial news related to the provided topics from the Topic Agent.",
    expected_output="A concise summary of recent news about the provided topics where each topic is "
    "given its own section and used news stories are referenced.",
    context=[define_topics],
    agent=news_agent,
)

# Function for inferring ticker symbols from a user expression
def infer_tickers_with_llm(user_expression):
    global llm
    # Construct a prompt for the LLM
    prompt = f"""
    Identify the stock ticker symbols for the following companies or phrases:
    {user_expression}

    Output the tickers as a comma-separated list. If no tickers are found, output "None".
    """
    # Call the LLM with the prompt
    response = llm.run(prompt)
    # Process the LLM's response
    if response.lower() == "none":
        return []
    else:
        tickers = [ticker.strip() for ticker in response.split(",")]
        return tickers
# Task for determining ticker symbols
infer_tickers_task = Task(
    description="Infer ticker symbols from instructions given from the Investment Manager.",
    expected_output="A list of inferred ticker symbols.",
    context=[manager_task],
    agent=data_analyst,
    function=infer_tickers_with_llm
)

# Function used to pull stock-related data from Yahoo Finance
def get_yahoo_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = {
            "current price": stock.info.get("currentPrice"),
            "day high": stock.info.get("dayHigh"),
            "day low": stock.info.get("dayLow"),
            "volume": stock.info.get("volume"),
            "market_cap": stock.info.get("marketCap"),
            "PE ratio": stock.info.get("trailingPE"),
            "EPS": stock.info.get("trailingEps")
        }
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
# Task for pulling relevant stock-related data from Yahoo Finance
fetch_stock_data = Task(
    description="Retrieve stock data for the necessary ticker symbols from Yahoo Finance.",
    expected_output="A dictionary containing stock data (price, volume, etc).",
    context=[infer_tickers_task],
    agent=data_analyst,
    function=get_yahoo_data
)

# Task for Quantitative Investment Specialist
quant_researcher_task = Task(
    description="Analyze the instructions from the Investment Manager and provide a detailed response.",
    expected_output="A detailed response pertaining to the assigned tasks.",
    context=[manager_task],
    agent=quant_researcher,
)

# Task for Quantitative Investment Reviewer
quant_reviewer_task = Task(
    description="Analyze the response from the Quant Investment Specialist and revise the output.",
    expected_output="A revision that is has more clarity and accuracy.",
    context=[quant_researcher_task],
    agent=quant_reviewer,
)

# Task for Information Aggregator
info_aggregator_task = Task(
    description="Analyze and aggregate the responses from the other agents, keeping only information related to the client query: {user_query}.",
    expected_output="An cogent response that is easy to understand.",
    context=[financial_advisor_task, fetch_news, fetch_stock_data, quant_reviewer_task],
    agent=info_aggregator,
)


#===========================================================================================
# CREW
#===========================================================================================

# Create the Crew
crew = Crew(
    agents=[
        investment_manager,
        financial_advisor,
        topic_agent,
        news_agent,
        quant_researcher,
        quant_reviewer,
        data_analyst,
        info_aggregator
    ],
    process=Process.sequential,
    verbose=True,
    tasks=[
        manager_task,
        financial_advisor_task,
        define_topics,
        fetch_news,
        infer_tickers_task,
        fetch_stock_data,
        quant_researcher_task,
        quant_reviewer_task,
        info_aggregator_task
    ]
)


#===========================================================================================
# PROCESSING FUNCTION
#===========================================================================================
    
# Main processing function
def process_query(query):
    """
    Process the user's query and return a response.
    """
    try:   
        # Kick off the workflow
        response = crew.kickoff(inputs={"user_query": query})
        # Extract the raw string output
        if hasattr(response, "raw"):
            response = response.raw
        else:
            raise ValueError("Invalid response format from crew.kickoff")
        # Convert the Markdown response to HTML
        response = mistune.create_markdown()(response)
        # Return the aggregated response from the Info Aggregator
        return response
    except Exception as e:
        # Log and return any errors
        print(f"Error during query processing: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"

