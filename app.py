import streamlit as st
import importlib.util

st.set_page_config(
    page_title="AI Payment Intelligence Platform",
    layout="wide"
)

st.title("AI Payment Intelligence Platform")
st.write("Loading payment intelligence recovery module...")

spec = importlib.util.spec_from_file_location(
    "payment_recovery",
    "payment_platform_FINAL_RECOVERY.txt.py"
)

payment_recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payment_recovery)

st.success("Recovery module loaded successfully.")
st.write("The backend module is available. Next, we can connect its functions to this Streamlit interface.")