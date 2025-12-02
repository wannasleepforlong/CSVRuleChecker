import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

csv_file_path = "customers.csv"
df = pd.read_csv(csv_file_path)

hard_rules = [
    "Number of rows should be more than 1000",
    "Customer IDs must be unique",
]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="openai-functions",
    allow_dangerous_code=True  
)

if __name__ == "__main__":
    response = agent.invoke("Number of customers with names starting from 'A'?")    
    print("\nAgent Response:")
    print(response["output"])