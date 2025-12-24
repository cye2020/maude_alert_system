import streamlit as st

def get_shared_data():
    if 'shared_data' not in st.session_state:
        st.session_state.shared_data = {}
    return st.session_state.shared_data