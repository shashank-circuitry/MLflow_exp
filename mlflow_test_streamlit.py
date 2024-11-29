import streamlit as st
import mlflow
import openai
import os
import pandas as pd
import dagshub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Agentic Evaluation") 

# Excel Upload Section
st.sidebar.header("Upload Excel for Input Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

# Default Evaluation Data
default_eval_data = pd.DataFrame(uploaded_file)

# Use uploaded data or default data
if uploaded_file:
    eval_data = pd.read_excel(uploaded_file)
    st.success("Excel file uploaded successfully!")
else:
    eval_data = default_eval_data
    st.info("Using default evaluation data.")

# Experiment Section
st.markdown("Run the MLflow Experiment")
mlflow.set_experiment("LLM Evaluation")

if st.button("Start Experiment"):
    with st.spinner("Experiment is running..."):
        try:
            with mlflow.start_run() as run:
                st.success("Experiment started successfully!")
                system_prompt = "Answer the following question in two sentences"

                # Log OpenAI GPT-4 model to MLflow
                st.write("Logging GPT-4 model...")
                logged_model_info = mlflow.openai.log_model(
                    model="gpt-4",
                    task=openai.chat.completions,
                    artifact_path="model",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "{question}"},
                    ],
                )

                # Evaluate the model
                st.write("Evaluating the model...")
                results = mlflow.evaluate(
                    logged_model_info.model_uri,
                    eval_data,
                    targets="ground_truth",
                    model_type="question-answering",
                    extra_metrics=[
                        mlflow.metrics.toxicity(),
                        mlflow.metrics.genai.answer_similarity(),
                        mlflow.metrics.genai.answer_correctness(),
                        mlflow.metrics.genai.answer_relevance(),
                        # mlflow.metrics.genai.faithfulness()   
                    ],
                )
                st.success("Evaluation completed successfully!") 
                
                # Display Evaluation Results Table
                st.markdown("Evaluation Results Table")
                eval_table = results.tables["eval_results_table"]
                eval_df = pd.DataFrame(eval_table)
                eval_df.columns = [col.replace('_', ' ').title() for col in eval_df.columns] 
                eval_df = eval_df.drop(columns=["Flesch Kincaid Grade Level/V1/Score", "Ari Grade Level/V1/Score"]) 

                # Display DataFrame without any styling
                st.dataframe(eval_df)

                # Save Results to Excel
                eval_df.to_excel(
                    r"C:\Users\ShashankMutyam\OneDrive - Circuitry.ai\Desktop\Code\eval_results.xlsx",
                    index=False,
                )
                st.success("Results saved to eval_results.xlsx")
        except Exception as e:
            st.error(f"Error during evaluation: {e}") 
