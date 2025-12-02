import os
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.set_page_config(page_title="CSV Rule Checker + Q&A", layout="wide")
st.title("CSV Rule Checker with Hard & Soft Rules + Q&A")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded CSV")
    st.dataframe(df)

    # Define rules
    hard_rules = [
        "Number of rows should be less than 1000",
        "Customer IDs must be unique",
    ]

    soft_rules = [
        "Customer names must start with a capital letter",
        "Email addresses must contain '@'",
    ]

    # OpenAI API Key
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

        # Create Pandas agent
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            agent_type="openai-functions",
            allow_dangerous_code=True
        )

        # --- Hard rules ---
        st.subheader("Checking Hard Rules...")
        results = []
        for rule in hard_rules:
            prompt = f"Check this hard rule on the dataframe. Respond only with 'Yes' or 'No': {rule}"
            response = agent.invoke(prompt)
            result_text = response["output"].strip()
            results.append("Yes" if "yes" in result_text.lower() else "No")

        # --- Soft rules ---
        st.subheader("Checking Soft Rules...")
        for i, rule in enumerate(soft_rules, 1):
            prompt = f"Find rows violating this soft rule and return a dataframe named 'violating_rows': {rule}"
            response = agent.invoke(prompt)
            local_vars = {"df": df.copy(), "pd": pd}
            violation_exists = False
            try:
                exec(response["output"], {}, local_vars)
                if "violating_rows" in local_vars and not local_vars["violating_rows"].empty:
                    violating_df = local_vars["violating_rows"]
                    st.write(f"Violations for Soft Rule {i}: {rule}")
                    st.dataframe(violating_df)
                    file_name = f"softrule{i}.csv"
                    violating_df.to_csv(file_name, index=False)
                    st.success(f"Violations saved to {file_name}")
                    violation_exists = True
            except Exception as e:
                st.info(f"No violations found or error for soft rule {i}: {e}")

            # Append Yes/No for soft rule summary
            results.append("Yes" if not violation_exists else "No")

        # --- Summary ---
        all_rules = hard_rules + soft_rules
        summary_df = pd.DataFrame({"Rule": all_rules, "Result": results})
        st.subheader("Summary of Hard + Soft Rules")
        st.dataframe(summary_df)
        summary_df.to_csv("rules_check.csv", index=False)
        st.success("Summary saved to rules_check.csv")

        # --- Q&A Section ---
        st.subheader("Ask Questions About the CSV")
        user_question = st.text_input("Enter your question (e.g., 'Number of customers with names starting from A?')")
        if st.button("Get Answer") and user_question:
            try:
                response = agent.invoke(user_question)
                st.write("**Answer:**")
                st.write(response["output"])
            except Exception as e:
                st.error(f"Error processing question: {e}")
