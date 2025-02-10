import streamlit as st
import matplotlib.pyplot as plt

st.title("📊 Stock Analysis Insights")

# Check if figure and description exist
if "analysis_fig" in st.session_state and "analysis_description" in st.session_state:
    st.subheader("📉 Stock Price with Bollinger Bands")

    # Display the stored figure
    st.pyplot(st.session_state["analysis_fig"])

    # Show description
    st.subheader("📜 Analysis Insights")
    st.write(st.session_state["analysis_description"])

    # Show investment decision
    if "positive" in st.session_state["analysis_description"].lower():
        st.success("✅ Based on the image analysis, this seems like a good investment opportunity!")
    else:
        st.warning("⚠️ Caution! The insights suggest that this might not be a great investment.")

    # Button to go back
    if st.button("🔙 Back to Main Page"):
        st.switch_page("main.py")  # Ensure correct path

else:
    st.error("🚨 No analysis data found. Please generate insights from the main page.")
