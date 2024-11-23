import streamlit as st
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

def DataCleaning():
    st.title("Data Cleaning")
    st.write("Clean your data here.")

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("Choose a file to clean")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.uploaded_file is not None:
        df = pd.read_csv(st.session_state.uploaded_file)
        st.dataframe(df)
        with st.expander("Summary Statistics"):
            st.write(df.describe())

        with st.expander("Missing Values"):
            st.write(df.isnull().sum())

        with st.expander("Duplicate Rows"):
            st.write(df.duplicated().sum())

        with st.expander("Data Types"):
            st.write(df.dtypes)
    
    if st.button("Clean Data"):
        if st.session_state.uploaded_file is not None:
            df = pd.read_csv(st.session_state.uploaded_file)
            initial_empty_cells = df.isnull().sum().sum()
            df = df.dropna(axis=1)
            final_empty_cells = df.isnull().sum().sum()
            empty_cells_removed = initial_empty_cells - final_empty_cells
            st.write(f"**Data cleaned successfully. {empty_cells_removed} empty cells removed.**")
            st.dataframe(df)    # Display cleaned data
        else:
            st.error("Please upload a file first.")