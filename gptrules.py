#Hard rules don't make csvs.
#Soft rules make csvs.

import os, json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

os.makedirs("results", exist_ok=True)
csv_file_path = "inputs/customers.csv"
df = pd.read_csv(csv_file_path)

rules_file_path = "inputs/rules.json"
with open(rules_file_path, 'r') as f:
    rules_config = json.load(f)

hard_rules = rules_config.get("hard_rules", [])
soft_rules = rules_config.get("soft_rules", [])

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


results = []
for rule in hard_rules:
    prompt = f"""
Evaluate the following rule on the dataframe carefully:

Rule: {rule}

You MUST respond with ONLY ONE WORD:
- 'Pass'  → if the dataframe satisfies the rule
- 'Fail'  → if the dataframe violates the rule
"""
    response = agent.invoke(prompt)
    result_text = response["output"].strip()
    results.append("Pass" if "pass" in result_text.lower() else "Fail")


for i, rule in enumerate(soft_rules, 1):
    prompt = f"""
You must return ONLY raw Python code.

Task:
- Using the dataframe 'df', find all rows violating this rule: {rule}
- Store the result strictly in a pandas dataframe VARIABLE named exactly: violating_rows
- Do NOT print anything.
- Do NOT return markdown.
- Do NOT add explanations.
- Do NOT return comments.
- Your entire response MUST be ONLY executable Python code defining 'violating_rows'.

Example format (for reference only):
violating_rows = df[<condition>]
"""

    response = agent.invoke(prompt)
    local_vars = {"df": df.copy(), "pd": pd}
    violation_exists = False
    try:
        exec(response["output"], {}, local_vars)
        if "violating_rows" in local_vars and not local_vars["violating_rows"].empty:
            violating_df = local_vars["violating_rows"]
            file_name = f"results/softrule{i}.csv"
            violating_df.to_csv(file_name, index=False)
            print(f"Violations for soft rule {i} saved to {file_name}")
            violation_exists = True
    except Exception as e:
        print(f"No violations found or error for soft rule {i}: {e}")
    
    # Append Yes/No for soft rule summary
    results.append("Pass" if not violation_exists else "Fail")

# --- Create combined summary for hard + soft rules ---
all_rules = hard_rules + soft_rules
summary_df = pd.DataFrame({
    "Rule": all_rules,
    "Result": results
})

summary_df.to_csv("results/validation.csv", index=False)
print("Summary saved to validation.csv")
print(summary_df)
