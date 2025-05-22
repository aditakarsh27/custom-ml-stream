import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import io
import base64
import matplotlib
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Data Chat & Visualize",
    page_icon="📊",
    layout="wide"
)

# --- Session State Initialization ---
for key, default in [
    ("messages", [{"role": "assistant", "content": "👋 Welcome! Upload a CSV or Excel file to get started, or ask me a question about your data."}]),
    ("df", None),
    ("df_description", ""),
    ("code_output", ""),
    ("interpreted_output", ""),
    ("df_info", ""),
    ("df_context", {}),
    ("file_uploader_key", 0),
    ("awaiting_response", False),
    ("current_question", ""),
    ("code", "")
]:
    if key not in st.session_state:
        st.session_state[key] = default

def generate_df_description(df):
    """Generate a description of the dataframe using the OpenAI API"""
    # Get basic dataframe info
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    
    # Store df_info in session state for future use
    st.session_state.df_info = df_info
    
    # Get sample data
    df_sample = df.head(5).to_string()
    
    # Get column descriptions
    columns_desc = df.describe().to_string()
    
    # Store context in session state
    st.session_state.df_context = {
        "df_info": df_info,
        "df_sample": df_sample,
        "columns_desc": columns_desc,
        "columns": df.dtypes.to_string()
    }
    
    prompt = f"""
    I have a pandas DataFrame with the following information:
    
    DataFrame Info:
    {df_info}
    
    Sample Data (first 5 rows):
    {df_sample}
    
    Summary Statistics:
    {columns_desc}
    
    Please provide a concise summary of this dataset. Include information about:
    1. The type of data and what it might represent
    2. The number of rows and columns
    3. The data types
    4. Any notable patterns or characteristics
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def interpret_code_output(code, output, conversation_history):
    """Use the LLM to interpret the code output in a user-friendly way"""
    # Get dataframe context
    df_info = st.session_state.df_context.get("df_info", "")
    
    # Format conversation history for the prompt
    conversation_text = ""
    for msg in conversation_history:
        conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    
    prompt = f"""
    Given the following conversation history:
    {conversation_text}
    
    The following code was executed on a pandas DataFrame, based on the LLM's response to the user's question:
    
    ```python
    {code}
    ```
    
    The DataFrame has the following information:
    {df_info}
    
    The output of the code execution was:
    ```
    {output}
    ```
    
    Please format and display the output in a clear, concise way that would be helpful for a non-technical user. Also explain any interpretations of the output. Do not include any Certainly! or similar phrases in your response, and only include the output of the code and its explanation if it is relevant to the user's question.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def process_data_query(query, df, conversation_history):
    """Process a data query using the OpenAI API with conversation history"""
    # Use the stored dataframe context
    df_info = st.session_state.df_context.get("df_info", "")
    columns = st.session_state.df_context.get("columns", "")
    df_sample = st.session_state.df_context.get("df_sample", "")
    
    # Format conversation history for the prompt
    messages = []
    
    # System message first
    messages.append({
        "role": "system", 
        "content": f"""You are a data analysis assistant. Help the user analyze their CSV data.
        
        The DataFrame has the following information:
        {df_info}
        
        Column Names and Types:
        {columns}
        
        Sample Data (first 5 rows):
        {df_sample}
        
        If the user's question requires generating a visualization or performing an analysis, provide Python code using pandas, matplotlib, seaborn, or plotly.
        For visualizations, use plt.figure(figsize=(10, 6)) for matplotlib plots to ensure they're readable.
        For any results that need to be shown, print the results.
        Wrap your code in ```python and ``` tags.
        If the question is general or asking for information, provide a helpful response without code.
        Your Python code should reference the DataFrame as 'df'.
        """
    })
    
    # Add previous conversation for context
    for msg in conversation_history:
        messages.append(msg)
    
    # Add the current query
    messages.append({"role": "user", "content": query})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )
    response_text = response.choices[0].message.content

    if "```python" in response_text:
        code_start = response_text.find("```python") + 9
        code_end = response_text.find("```", code_start)
        code = response_text[code_start:code_end].strip()
        explanation = response_text.replace("```python" + code + "```", "").strip()
        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            # Namespace for code execution
            local_vars = {"df": df, "pd": pd, "plt": plt, "px": px, "sns": sns, "np": np, "io": io, "base64": base64, "st": st}
            exec(code, local_vars)
            # After exec(code, local_vars)
            printed_output = mystdout.getvalue()
            sys.stdout = old_stdout

            # ----------- Matplotlib -----------
            figs_matplotlib = []
            for n in plt.get_fignums():
                fig = plt.figure(n)
                figs_matplotlib.append(fig)

            # ----------- Plotly -----------
            fig_plotly = None
            for v in local_vars.values():
                if str(type(v)).startswith("<class 'plotly.graph_objs._figure.Figure"):
                    fig_plotly = v
                    break
            if 'fig' in local_vars:
                try:
                    import plotly.graph_objs
                    if isinstance(local_vars['fig'], plotly.graph_objs.Figure):
                        fig_plotly = local_vars['fig']
                except ImportError:
                    pass

            interpreted_output = ""
            if printed_output.strip():
                interpreted_output = interpret_code_output(code, printed_output, conversation_history)

            return {
                "explanation": explanation,
                "code": code,
                "output": printed_output,
                "interpreted_output": interpreted_output,
                "figs_matplotlib": figs_matplotlib,  # <- list of all figures
                "fig_plotly": fig_plotly
            }


        except Exception as e:
            return {
                "explanation": explanation,
                "code": code,
                "output": f"Error executing code: {str(e)}",
                "interpreted_output": "",
                "fig_matplotlib": None,
                "fig_plotly": None
            }
    else:
        return {
            "explanation": response_text,
            "code": None,
            "output": None,
            "interpreted_output": "",
            "fig_matplotlib": None,
            "fig_plotly": None
        }

def handle_file_upload(uploaded_file):
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_type}")
            
        st.session_state.df = df
        with st.spinner("Analyzing your data..."):
            description = generate_df_description(df)
            st.session_state.df_description = description
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"📊 Successfully loaded **{uploaded_file.name}** with {len(df)} rows and {len(df.columns)} columns.\n\n{description}"
        })
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- Layout Start ---

st.title("Data Chat & Visualize")

# Split layout: Left = Chat, Right = Visualization
left_col, right_col = st.columns([1.7, 1.7], gap="large")

# -- LEFT COLUMN: Chat UI + File Uploader --
with left_col:
    st.header("Chat with your Data")
    chat_container = st.container()
    with chat_container:
        # Limit number of chat messages for display (e.g. last 25)
        for message in st.session_state.messages[-25:]:
            align = "assistant" if message["role"] == "assistant" else "user"
            with st.chat_message(align):
                st.markdown(message["content"])

    chat_input = st.chat_input("Ask a question about your data...")

    upload_col = st.container()
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=["csv", "xlsx", "xls"],
            key=f"file_uploader_{st.session_state.file_uploader_key}",
            help="Upload a CSV or Excel file to analyze"
        )
        if uploaded_file is not None:
            handle_file_upload(uploaded_file)
            st.session_state.file_uploader_key += 1
            st.rerun()

    # Handle chat input
    if chat_input:
        if st.session_state.df is None:
            st.session_state.messages.append({"role": "assistant", "content": "❗ Please upload a data file first."})
            st.rerun()
        else:
            st.session_state.awaiting_response = True
            st.session_state.current_question = chat_input
            st.session_state.messages.append({"role": "user", "content": chat_input})
            st.rerun()

# -- RIGHT COLUMN: Data Viz, Output, Tabs --
with right_col:
    st.header("Data Visualization & Analysis")
    if st.session_state.df is not None:
        viz_tabs = st.tabs(["Visualization", "Interpretation", "Code & Output"])
        # 1. Visualization
        with viz_tabs[0]:
            st.subheader("Visualization")
            # Try to display the most recent figure, if any
            figs_matplotlib = st.session_state.get("figs_matplotlib", [])
            fig_plotly = st.session_state.get("fig_plotly")
            if figs_matplotlib:
                for fig in figs_matplotlib:
                    st.pyplot(fig)
                    plt.close(fig) 
            elif fig_plotly is not None:
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.info("Visualizations will appear here after you ask a question.")

        # 2. Interpretation
        with viz_tabs[1]:
            st.subheader("Interpretation")
            st.write(st.session_state.interpreted_output or "Ask a question to see an interpretation of the results.")
        # 3. Code
        with viz_tabs[2]:
            st.subheader("Code & Output")
            if st.session_state.code:
                with st.expander("View Code", expanded=True):
                    st.code(st.session_state.code, language="python")
            if st.session_state.code_output:
                with st.expander("Raw Output", expanded=True):
                    st.text(st.session_state.code_output)
            if not st.session_state.code and not st.session_state.code_output:
                st.write("Ask a question to see the code and output.")

# --- Process LLM Response ---
if st.session_state.awaiting_response and st.session_state.df is not None:
    user_query = st.session_state.current_question
    st.session_state.awaiting_response = False
    st.session_state.current_question = ""
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
        if msg["role"] in ["user", "assistant"]
    ]
    with st.spinner("Thinking..."):
        st.session_state.fig_matplotlib = None
        st.session_state.fig_plotly = None
        result = process_data_query(user_query, st.session_state.df, conversation_history)
        response_content = result["explanation"]
        if result["output"]:
            st.session_state.code_output = result["output"]
            st.session_state.interpreted_output = result["interpreted_output"]
            st.session_state.code = result["code"]
            response_content += "\n\n(See visualization and interpretation in the tabs above)"
        if result["explanation"]:
            st.write(result["explanation"])

        if result.get("code"):
            st.session_state.code = result["code"]

        # Store output for tabs
        st.session_state.code_output = result.get("output", "")
        st.session_state.interpreted_output = result.get("interpreted_output", "")
        st.session_state.figs_matplotlib = result.get("figs_matplotlib", [])
        st.session_state.fig_plotly = result.get("fig_plotly")

        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.rerun()

if __name__ == "__main__":
    pass