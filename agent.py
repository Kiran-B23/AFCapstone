import yfinance as yf, pandas as pd
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import get_retriever
from checker import flag_transactions

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
retriever = get_retriever()

def market_data(ticker: str) -> str:                    # Tool 1 — yfinance
    t = yf.Ticker(ticker.strip().upper())
    hist = t.history(period="1mo")
    if hist.empty:
        return f"No market data for {ticker}."
    price, month_ago = hist["Close"].iloc[-1], hist["Close"].iloc[0]
    change = (price - month_ago) / month_ago * 100
    return f"{ticker}: price ${price:.2f}, 1-month change {change:+.1f}%."

def research(query: str) -> str:                        # Tool 2 — RAG
    hits = retriever.invoke(query)
    return "\n\n".join(f"[source] {d.page_content}" for d in hits) or "No relevant documents."

# The transactions are loaded per request; the tool reads the redacted CSV path from the app.
def check_uploaded(_input: str) -> str:                 # Tool 3 — transaction checker
    df = pd.read_csv("data/_current_upload.csv")
    flagged = flag_transactions(df)
    if flagged.empty:
        return "No suspicious transactions found."
    return "Suspicious transactions:\n" + flagged.to_string(index=False)

TOOLS = [
    Tool(name="MarketData", func=market_data,
         description="Get live price and 1-month trend for a stock ticker like AAPL."),
    Tool(name="Research", func=research,
         description="Look up finance documents to ground an answer about a company or topic."),
    Tool(name="CheckTransactions", func=check_uploaded,
         description="Scan the user's uploaded transactions for suspicious activity."),
]

agent = initialize_agent(
    TOOLS, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,   # the agent reasons, then picks a tool
    verbose=True,                                  # prints its reasoning (also shows in LangSmith)
)