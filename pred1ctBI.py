import streamlit as st
import pandas as pd
import pygwalker as pyg



st.set_page_config(
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    
    )

def DataCleaning():
    st.title("Data Cleaning")
    st.write("Clean your data here.")

    uploaded_file = st.file_uploader("Choose a file to clean")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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
        if uploaded_file is not None:
            initial_empty_cells = df.isnull().sum().sum()
            df = df.dropna(axis=1)
            final_empty_cells = df.isnull().sum().sum()
            empty_cells_removed = initial_empty_cells - final_empty_cells
            st.write(f"**Data cleaned successfully. {empty_cells_removed} empty cells removed.**")
            st.dataframe(df)    # Display cleaned data
        else:
            st.error("Please upload a file first.")

def Visualization():
    st.title("Data Visualization")
    st.write("Visualize your data here.")

    uploaded_file = st.file_uploader("Choose a file to visualize")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        pygwalker = pyg.walk(df)
        st.components.v1.html(pygwalker.to_html(), height=800, width=1500, scrolling=True)

def main():
    st.sidebar.title("Navigation")  
    page = st.sidebar.selectbox("Go to", ["Data Cleaning", "Data Visualization"])

    if page == "Data Cleaning":
        DataCleaning()
    elif page == "Data Visualization":
        Visualization()

if __name__ == "__main__":
    main()