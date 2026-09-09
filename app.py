import streamlit as st
import importlib.util
import pickle

st.set_page_config(
    page_title="AI Payment Intelligence Platform",
    layout="wide"
)

st.title("AI Payment Intelligence Platform")
st.write("Loading payment intelligence platform...")

# Load recovery module
spec = importlib.util.spec_from_file_location(
    "payment_recovery",
    "payment_platform_FINAL_RECOVERY.txt.py"
)

payment_recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payment_recovery)

# Load deployment state
with open("payment_platform_1M_DEPLOYMENT_STATE.pkl", "rb") as f:
    checkpoint = pickle.load(f)

# Restore runtime state
restore_info = payment_recovery.restore_runtime_state(checkpoint)

st.success("Payment intelligence platform loaded successfully.")

st.subheader("Deployment State")

st.write(restore_info)