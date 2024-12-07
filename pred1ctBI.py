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
    st.session_state.visualization_settings = {"graph": None}
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
            history=[]
        )
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the explanation: {str(e)}"

def save_as_docx(interpretation, fig):
    doc = Document()
    doc.add_heading("AI Interpretation and Visualization", level=1)

    doc.add_paragraph(interpretation)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png")
        tmpfile.flush()
        doc.add_picture(tmpfile.name, width=Inches(6))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmpfile:
        doc.save(tmpfile.name)
        tmpfile.flush()

        with open(tmpfile.name, "rb") as f:
            docx_buffer = BytesIO(f.read())

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

        graph_type = st.selectbox(
            "Select Graph Type",
            options=["Bar Chart", "Line Chart", "Pie Chart", "Parallel Coordinates", "Area Chart"]
        )
        st.session_state.visualization_settings["graph"] = graph_type

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

        measure_options = ["Count", "Sum", "Average", "Min", "Max"]

        if graph_type in ["Bar Chart", "Line Chart", "Area Chart"]:
            x_axis = st.selectbox("Select X-axis (Categorical)", options=df.columns)
            y_axis = st.selectbox("Select Y-axis (Numeric)", options=df.columns)

            valid_x = x_axis in df.columns
            valid_y = y_axis in numeric_columns or y_axis == x_axis
            
            if not valid_x or "Unnamed" in x_axis:
                st.error("Please select a valid X-axis column.")
            if not valid_y or "Unnamed" in y_axis:
                st.error("Please select a valid numeric Y-axis column.")

            if valid_y and y_axis in numeric_columns:
                measure = st.selectbox("Measure Values", options=measure_options)
            else:
                measure = "Count" 

        elif graph_type == "Pie Chart":
            category_col = st.selectbox("Select Category Column", options=df.columns)
            valid_cat = category_col in df.columns

            numeric_col_for_pie = st.selectbox("Select Numeric Column (optional)", options=["None"]+numeric_columns)
            if numeric_col_for_pie == "None":
                measure = "Count"
                numeric_col_for_pie = None
            else:
                measure = st.selectbox("Measure Values", options=measure_options)

        elif graph_type == "Parallel Coordinates":
            if len(categorical_columns) == 0:
                st.error("You need at least one categorical column for Parallel Coordinates.")
                return
            if len(numeric_columns) < 2:
                st.error("You need at least two numeric columns for Parallel Coordinates.")
                return
            class_column = st.selectbox("Select Categorical Column for Class", options=categorical_columns)
            numeric_cols_selected = st.multiselect("Select Numeric Columns", options=numeric_columns, default=numeric_columns[:2])
            measure = None 
        else:
            measure = "Count"

        if st.button("Generate Visualization & Interpretation"):
            try:
                fig, ax = plt.subplots(figsize=(10, 10))
                fig.tight_layout()
                fig.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.1)

                if graph_type in ["Bar Chart", "Line Chart", "Area Chart"]:
                    if measure == "Count":
                        df_grouped = df.groupby(x_axis).size()
                        y_label = "Count"
                    else:
                        if y_axis not in numeric_columns:
                            st.error("Selected Y-axis must be numeric for the chosen measure.")
                            return
                        if measure == "Sum":
                            df_grouped = df.groupby(x_axis)[y_axis].sum()
                        elif measure == "Average":
                            df_grouped = df.groupby(x_axis)[y_axis].mean()
                        elif measure == "Min":
                            df_grouped = df.groupby(x_axis)[y_axis].min()
                        elif measure == "Max":
                            df_grouped = df.groupby(x_axis)[y_axis].max()
                        y_label = f"{measure} of {y_axis}"

                    if graph_type == "Bar Chart":
                        df_grouped.plot(kind="bar", ax=ax)
                        ax.set_title("Bar Chart")
                    elif graph_type == "Line Chart":
                        df_grouped.plot(kind="line", ax=ax)
                        ax.set_title("Line Chart")
                    elif graph_type == "Area Chart":
                        df_grouped.plot(kind="area", ax=ax)
                        ax.set_title("Area Chart")

                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_label)

                elif graph_type == "Pie Chart":
                    if not valid_cat:
                        st.error("Please select a valid category column for the Pie Chart.")
                        return
                    if measure == "Count":
                        df_grouped = df.groupby(category_col).size()
                    else:
                        if numeric_col_for_pie not in numeric_columns:
                            st.error("Selected numeric column for Pie Chart must be valid and numeric.")
                            return
                        if measure == "Sum":
                            df_grouped = df.groupby(category_col)[numeric_col_for_pie].sum()
                        elif measure == "Average":
                            df_grouped = df.groupby(category_col)[numeric_col_for_pie].mean()
                        elif measure == "Min":
                            df_grouped = df.groupby(category_col)[numeric_col_for_pie].min()
                        elif measure == "Max":
                            df_grouped = df.groupby(category_col)[numeric_col_for_pie].max()

                    df_grouped.plot(kind="pie", ax=ax, autopct='%1.1f%%')
                    ax.set_title("Pie Chart")
                    ax.set_ylabel("")

                elif graph_type == "Parallel Coordinates":
                    if class_column not in categorical_columns:
                        st.error("Class column must be categorical.")
                        return
                    if len(numeric_cols_selected) < 2:
                        st.error("Please select at least two numeric columns.")
                        return

                    subset_df = df[[class_column] + numeric_cols_selected].dropna()
                    pd.plotting.parallel_coordinates(subset_df, class_column, ax=ax)
                    ax.set_title("Parallel Coordinates")

                st.pyplot(fig)
                st.session_state.fig = fig

            except Exception as e:
                st.error(f"An error occurred while creating the plot: {str(e)}")
                return

            grouped_values = df_grouped.reset_index().values.tolist()

            if graph_type in ["Bar Chart", "Line Chart", "Area Chart"]:
                explanation_prompt = (
                    f"You have created a {graph_type.lower()} using the column '{x_axis}' as the X-axis and "
                    f"applied the measure '{measure}' on the column '{y_axis}'. "
                    f"The aggregated data is as follows (X-axis value, Aggregated result): {grouped_values}. "
                    f"Explain the trends, patterns, and insights visible in this data. "
                    f"Consider how the values vary across different '{x_axis}' categories."
                )

            elif graph_type == "Pie Chart":
                if measure == "Count":
                    explanation_prompt = (
                        f"A pie chart has been created using '{category_col}' as the category dimension, "
                        f"showing the percentage distribution of counts among each category. "
                        f"Here are the values (Category, Count): {grouped_values}. "
                        f"Describe what this data reveals about the relative proportions of the categories."
                    )
                else:
                    explanation_prompt = (
                        f"A pie chart has been created using '{category_col}' as the category dimension. "
                        f"The values represent the {measure.lower()} of '{numeric_col_for_pie}'. "
                        f"Here are the aggregated values (Category, Aggregated result): {grouped_values}. "
                        f"Explain what this data shows about the distribution of these values across the categories."
                    )

            elif graph_type == "Parallel Coordinates":
                sample_values = subset_df.head(100).to_dict(orient='records') 
                explanation_prompt = (
                    f"A parallel coordinates plot has been created using '{class_column}' as the class column "
                    f"and {numeric_cols_selected} as the numeric features. "
                    f"Here is a sample of the data used:\n{sample_values}\n"
                    f"Interpret how different classes differ across these numeric dimensions and describe any patterns you observe."
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