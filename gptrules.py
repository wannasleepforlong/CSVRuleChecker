import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load CSV
csv_file_path = "customers.csv"
df = pd.read_csv(csv_file_path)

# Define rules
hard_rules = [
    "Number of rows should be less than 1000",
    "Customer IDs must be unique",
]

soft_rules = [
    "Customer names must start with a capital letter",
    "Email addresses must contain '@'",
]

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create Pandas agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    agent_type="openai-functions",
    allow_dangerous_code=True
)

# --- Evaluate hard rules ---
results = []
for rule in hard_rules:
    prompt = f"Check this hard rule on the dataframe. Respond only with 'Yes' or 'No': {rule}"
    response = agent.invoke(prompt)
    result_text = response["output"].strip()
    results.append("Yes" if "yes" in result_text.lower() else "No")

# --- Evaluate soft rules ---
for i, rule in enumerate(soft_rules, 1):
    prompt = f"Find rows violating this soft rule and return a dataframe named 'violating_rows': {rule}"
    response = agent.invoke(prompt)
    local_vars = {"df": df.copy(), "pd": pd}
    violation_exists = False
    try:
        exec(response["output"], {}, local_vars)
        if "violating_rows" in local_vars and not local_vars["violating_rows"].empty:
            violating_df = local_vars["violating_rows"]
            file_name = f"softrule{i}.csv"
            violating_df.to_csv(file_name, index=False)
            print(f"Violations for soft rule {i} saved to {file_name}")
            violation_exists = True
    except Exception as e:
        print(f"No violations found or error for soft rule {i}: {e}")
    
    # Append Yes/No for soft rule summary
    results.append("Yes" if not violation_exists else "No")

# --- Create combined summary for hard + soft rules ---
all_rules = hard_rules + soft_rules
summary_df = pd.DataFrame({
    "Rule": all_rules,
    "Result": results
})

summary_df.to_csv("rules_check.csv", index=False)
print("Summary saved to rules_check.csv")
print(summary_df)
