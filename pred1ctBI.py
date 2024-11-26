import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
if "visualization_settings" not in st.session_state:
    st.session_state.visualization_settings = {"x": None, "y": None, "measure": None, "graph": None}


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

    uploaded_file = st.file_uploader("Choose a file to clean")
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
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
    st.title("Data Visualization with AI Interpretation")
    st.write("Visualize your data and get AI insights here.")

    uploaded_file = st.file_uploader("Choose a file to visualize")
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        df = st.session_state.data

        st.sidebar.subheader("Visualization Options")
        x_axis = st.sidebar.selectbox("Select X-axis", options=df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", options=df.columns)
        measure = st.sidebar.selectbox(
            "Measure Values",
            options=["Count", "Sum", "Average", "Min", "Max"]
        )
        graph_type = st.sidebar.selectbox(
            "Select Graph Type",
            options=["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
        )

        st.session_state.visualization_settings.update({
            "x": x_axis,
            "y": y_axis,
            "measure": measure,
            "graph": graph_type,
        })

        if st.button("Generate Visualization & Interpretation"):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Data grouping and plotting logic
            if measure == "Count":
                df_grouped = df.groupby(x_axis).size()
            else:
                if y_axis in df.select_dtypes(include=["number"]).columns:
                    if measure == "Sum":
                        df_grouped = df.groupby(x_axis)[y_axis].sum()
                    elif measure == "Average":
                        df_grouped = df.groupby(x_axis)[y_axis].mean()
                    elif measure == "Min":
                        df_grouped = df.groupby(x_axis)[y_axis].min()
                    elif measure == "Max":
                        df_grouped = df.groupby(x_axis)[y_axis].max()
                else:
                    st.error("Measure values are only applicable to numeric columns.")
                    return

            if graph_type == "Bar Chart":
                df_grouped.plot(kind="bar", ax=ax)
                ax.set_title("Bar Chart")
            elif graph_type == "Line Chart":
                df_grouped.plot(kind="line", ax=ax)
                ax.set_title("Line Chart")
            elif graph_type == "Scatter Plot":
                if y_axis in df.select_dtypes(include=["number"]).columns:
                    df.plot.scatter(x=x_axis, y=y_axis, ax=ax)
                else:
                    st.error("Scatter plots require a numeric column for Y-axis.")
                    return
            elif graph_type == "Histogram":
                if y_axis in df.select_dtypes(include=["number"]).columns:
                    df[y_axis].plot(kind="hist", bins=20, ax=ax)
                else:
                    st.error("Histograms require a numeric column for the selected measure.")
                    return

            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            st.pyplot(fig)

            settings = st.session_state.visualization_settings
            x_data_sample = df[x_axis].head(10).tolist()
            y_data_sample = df[y_axis].head(10).tolist()

            explanation_prompt = (
                f"The visualization is generated based on the following data:\n"
                f"- X-axis (`{settings['x']}`) values: {x_data_sample}\n"
                f"- Y-axis (`{settings['y']}`) values: {y_data_sample}\n"
                f"The measure used is `{settings['measure']}` and the graph type is `{settings['graph']}`. "
                f"Analyze the data and explain the key findings or insights this visualization provides."
            )
            explanation = generate_explanation(explanation_prompt)
            st.subheader("AI Interpretation")
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