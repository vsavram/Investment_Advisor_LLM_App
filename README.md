# Investment Advisor LLM Application

### Overview
The Investment Advisor LLM App is an AI-powered application designed to provide personalized financial insights and investment strategies. 
Users can submit queries related to financial planning, investment strategies, or market data, and the app leverages a multi-agent LLM workflow 
to provide detailed, actionable responses.

## Motivation

Most retail investors (or people in general) lack the resources to properly achieve financial goals. They lack the knowledge for creating a financial plan or investment strategy and lack the resources for efficiently accessing investing-related data. The Investment Advisor LLM is meant to provide detailed information and actionable recommendations that lay people can understand. 

## Links for Web Application

* The Github repository can be found at [https://github.com/vsavram/Investment\_Advisor\_LLM\_App](https://github.com/vsavram/Investment_Advisor_LLM_App)

* The web application can be accessed [https://investment-advisor-app.onrender.com/](https://investment-advisor-app.onrender.com/)

## Example Cases

1. What is the current price of Tesla stock? Is this a good time to invest in Tesla?
2. I want to save \$200,000 over the next 5 years. How can I do this?
3. I have invested in an AI/tech ETF that tracks most Fortune 500 tech companies. Will this continue to do well over the next year?

## Workflow

**Overview:**

The Investment Advisor LLM leverages a multi-agent system implemented using the CrewAI framework. The crew is meant to leverage a hierarchical structure where the Investment Manager serves as the first point of contact for clients. However, a hierarchical structure is simulated using a sequential structure where the manager delegates tasks to investment specialists based on their respective areas of expertise (task dependencies replicate a hierarchical structure with delegation). The results from the investment specialists are are sent to an Information Aggregator that consolidates the results into easily digestible, actionable insights that are presented to the client.

An overview of the agents and their respective capabilities/tools is provided below.

* **Investment Advisor** 
	* **Description:** This agent is the first point of contact for clients. The manager has intimiate knowledge of the varying expertise of the investment specialists.
	* **Capabilities/Tools:** Delegation to other investment specialists.
* **Basic Financial Advisor** 
	* **Description:** This agent is capable of providing information and recommendations related to basic financial planning, understanding the given client's goals and constraints.
	* **Capabilities/Tools:** RAG for investment strategy-related papers (using OpenAI embeddings and FAISS vector-store)
* **Topic Agent** 
	* **Description:** This agent defines a set of topics related to the client's query and provides these topics to the News Coverage Agent.
	* **Capabilities/Tools:** NA
* **News Coverage Agent** 
	* **Description:** This agent pulls relevant news coverage pertaining to the client's query that can help answer questions or provide rationale for investment strategies.
	* **Capabilities/Tools:** Web search/scrape
* **Quantitative Investment Specialist** 
	* **Description:** This agent has a deep understanding of sophisticated investment strategies and can provide recommendations for a given client's objectives.
	* **Capabilities/Tools:** RAG for investment strategy-related papers (using OpenAI embeddings and FAISS vector-store)
* **Quantitative Investment Reviewer** 
	* **Description:** This agent reviews and revises the output from the Quantitative Investment Specialist, ensuring that the output is clear and correct.
	* **Capabilities/Tools:** RAG for investment strategy-related papers (using OpenAI embeddings and FAISS vector-store)
* **Data Analyst** 
	* **Description:** This agent is able to pull relevant stock-related data from Yahoo Finance including current price, daily low/high, daily volume, market cap, PE ratio, and EPS.
	* **Capabilities/Tools:** API access for Yahoo Finance

<div style="text-align: center;">
<img src="workflow.png" alt="" width="700">
</div>

## Software Used

**LangChain** 
[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

**CrewAI** 
[https://github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)

**OpenAI** 

* GPT-4o [https://arxiv.org/pdf/2410.21276](https://arxiv.org/pdf/2410.21276)
* API [https://platform.openai.com/docs/overview](https://platform.openai.com/docs/overview) 

**FAISS** 
[https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

**pydantic**
[https://github.com/pydantic/pydantic](https://github.com/pydantic/pydantic)

**yfinance (Yahoo Finance API)**
[https://github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)

**SERPER (Google Search API)**
[https://serper.dev/](https://serper.dev/)

**Render (Web Hosting)**
[https://render.com/docs](https://render.com/docs)
