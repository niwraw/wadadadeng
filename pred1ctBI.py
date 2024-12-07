import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
import seaborn as sns
import plotly.express as px
import google.generativeai as genai
from fpdf import FPDF
from docx import Document
from docx.shared import Inches
from io import BytesIO

genai.configure(api_key=os.environ.get("PALM_API_KEY"))

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
        height: calc(100vh - 200px) !important;
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
if "interpretation" not in st.session_state:
    st.session_state.interpretation = None
if "fig" not in st.session_state:
    st.session_state.fig = None

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

def generate_explanation(prompt):
    try:
        chat_session = model.start_chat(
            history=[
            ]
        )

        response = chat_session.send_message(prompt)

        return response.text
    except Exception as e:
        return f"An error occurred while generating the explanation: {str(e)}"

def save_as_docx(interpretation, fig):
    doc = Document()
    doc.add_heading("AI Interpretation and Visualization", level=1)

    # Add interpretation text
    doc.add_paragraph(interpretation)

    # Save the plot as an image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png")
        tmpfile.flush()

        # Add the image from the temporary file to the docx
        doc.add_picture(tmpfile.name, width=Inches(6))

    # Save the docx to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmpfile:
        doc.save(tmpfile.name)
        tmpfile.flush()

        # Read the docx back into a BytesIO object
        with open(tmpfile.name, "rb") as f:
            docx_buffer = BytesIO(f.read())

    # Download button to allow the user to download the docx
    st.download_button(
        label="📥 Download DOCX",
        data=docx_buffer,
        file_name="interpretation_and_visualization.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

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
        measure = st.sidebar.selectbox("Measure Values", options=["Count", "Sum", "Average", "Min", "Max"])
        graph_type = st.sidebar.selectbox(
            "Select Graph Type", 
            options=["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Pie Chart", "Box Plot", "Heatmap","Parallel Coordinates", "Area Chart"]
        )

        st.session_state.visualization_settings.update({
            "x": x_axis,
            "y": y_axis,
            "measure": measure,
            "graph": graph_type,
        })

        if st.button("Generate Visualization & Interpretation"):
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.tight_layout()
            fig.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)

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
                df.plot.scatter(x=x_axis, y=y_axis, ax=ax)
                ax.set_title("Scatter Plot")
            elif graph_type == "Histogram":
                df[y_axis].plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Histogram")
            elif graph_type == "Pie Chart":
                df_grouped.plot(kind="pie", ax=ax)
                ax.set_title("Pie Chart")
            elif graph_type == "Box Plot":
                df.boxplot(column=[y_axis], by=x_axis, ax=ax)
                ax.set_title("Box Plot")
            elif graph_type == "Heatmap":
                sns.heatmap(df.corr(), annot=True, ax=ax)
                ax.set_title("Heatmap")
            elif graph_type == "Area Chart":
                df_grouped.plot(kind="area", ax=ax)
                ax.set_title("Area Chart")
            elif graph_type == "Parallel Coordinates":
                pd.plotting.parallel_coordinates(df, x_axis, ax=ax)
                ax.set_title("Parallel Coordinates")

            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            st.pyplot(fig)

            st.session_state.fig = fig

            explanation_prompt = (
                f"The visualization is generated based on the following data:\n"
                f"- X-axis (`{x_axis}`) values: {df[x_axis].tolist()}\n"
                f"- Y-axis (`{y_axis}`) values: {df[y_axis].tolist()}\n"
                f"- Measure Values: `{measure}`\n"
                f"The data provided will be used to generate a `{graph_type}`.\n"
                f"Generate an interpration of the data."
            )

            explanation = generate_explanation(explanation_prompt)
            st.subheader("AI Interpretation")
            st.session_state.interpretation = explanation
            st.write(explanation)

            save_as_docx(explanation, fig)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Data Cleaning", "Data Visualization"])

    if page == "Data Cleaning":
        DataCleaning()
    elif page == "Data Visualization":
        Visualization()

if __name__ == "__main__":
    main()