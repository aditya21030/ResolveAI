import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="ResolveAI",
    page_icon="🎫",
    layout="centered"
)

st.title("ResolveAI")
st.caption("AI-powered customer support ticket routing")

ticket = st.text_area(
    "Customer Ticket",
    placeholder="Describe the customer's issue...",
    height=160
)

if st.button("Analyze Ticket", type="primary", use_container_width=True):

    if len(ticket.strip()) < 3:
        st.warning("Please enter a valid customer ticket.")

    else:
        try:
            with st.spinner("Analyzing ticket..."):
                response = requests.post(
                    API_URL,
                    json={"ticket": ticket},
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

            intent = result["intent"].replace("_", " ").title()
            confidence = result["confidence"]
            action = result["action"]

            st.divider()
            st.subheader("Analysis Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Predicted Intent",
                    value=intent
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.2f}%"
                )

            st.progress(confidence)

            if action == "auto_route":
                st.success(
                    f"Auto Route: Send this ticket to the "
                    f"**{intent}** queue."
                )
            else:
                st.warning(
                    "Human Review: Model confidence is too low "
                    "for automatic routing."
                )

        except requests.RequestException:
            st.error(
                "Could not connect to the ResolveAI API. "
                "Make sure the FastAPI server is running."
            )