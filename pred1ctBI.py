import streamlit as st
import pandas as pd
import pygwalker as pyg
import os
from groq import Groq

os.environ["GROQ_API_KEY"] = "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"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(
    page_title="Data App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
        .stButton > button {
            width: 100%;
        }
    }
    iframe {
        width: 100% !important;
        height: calc(100vh - 200px) !important; /* Adjusts dynamically to viewport */
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "data" not in st.session_state:
    st.session_state.data = None
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


@st.cache_resource
def generate_pygwalker_html(df):
    pygwalker = pyg.walk(df)
    return pygwalker.to_html()


def generate_explanation(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the explanation: {str(e)}"


def DataCleaning():
    st.title("Data Cleaning")
    st.write("Clean your data here.")

    st.session_state.uploaded_file = st.file_uploader("Choose a file to clean")
    if st.session_state.uploaded_file is not None:
        st.session_state.data = pd.read_csv(st.session_state.uploaded_file)
        st.dataframe(st.session_state.data)

        with st.expander("Summary Statistics"):
            st.write(st.session_state.data.describe())

        with st.expander("Missing Values"):
            st.write(st.session_state.data.isnull().sum())

        with st.expander("Duplicate Rows"):
            st.write(st.session_state.data.duplicated().sum())

        with st.expander("Data Types"):
            st.write(st.session_state.data.dtypes)

    if st.button("Clean Data"):
        if st.session_state.data is not None:
            initial_empty_cells = st.session_state.data.isnull().sum().sum()
            st.session_state.cleaned_data = st.session_state.data.dropna(axis=1)
            final_empty_cells = st.session_state.cleaned_data.isnull().sum().sum()
            empty_cells_removed = initial_empty_cells - final_empty_cells
            st.write(
                f"**Data cleaned successfully. {empty_cells_removed} empty cells removed.**"
            )
            st.dataframe(st.session_state.cleaned_data)
        else:
            st.error("Please upload a file first.")


def Visualization():
    st.title("Data Visualization")
    st.write("Visualize your data here.")

    uploaded_file = st.file_uploader("Choose a file to visualize")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        pygwalker_html = generate_pygwalker_html(df)

        st.components.v1.html(
            f"""
            <div style="width: 100%; height: 100%; overflow: auto;">
                {pygwalker_html}
            </div>
            """,
            height=800,
            scrolling=True,
        )

        if st.button("Generate AI Explanation"):
            explanation = generate_explanation("Explain the key insights from this dataset.")
            st.subheader("AI Explanation")
            st.write(explanation)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Data Cleaning", "Data Visualization"])

    if page == "Data Cleaning":
        DataCleaning()
    elif page == "Data Visualization":
        Visualization()


if __name__ == "__main__":
    main()
