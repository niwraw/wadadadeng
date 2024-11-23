import streamlit as st
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

def Visualization():
    st.title("Data Visualization")
    st.write("Visualize your data here.")

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    uploaded_file = st.file_uploader("Choose a file to visualize")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.df = pd.read_csv(uploaded_file)

    if st.session_state.df is not None:
        pygwalker = pyg.walk(st.session_state.df)
        st.components.v1.html(pygwalker.to_html(), height=800, width=1500, scrolling=True)

if __name__ == "__main__":
    Visualization()