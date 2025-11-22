import streamlit as st

# Set the title of the page
st.title("Supplier Feedback Questionnaire for Vendor Evaluation")

# Introduction
st.write(
    "As part of our vendor evaluation process, we are seeking feedback on [Vendor Name]. "
    "Your insights will help us assess their reliability and financial responsibility. "
    "Thank you for your valuable input."
)

# Questions
vendor_name = st.text_input("Vendor Name:", placeholder="Enter the vendor name")

if vendor_name:
    st.subheader(f"Feedback for {vendor_name}")

    # Question 1: Business Relationship
    q1 = st.text_input(
        f"How long have you been supplying goods/services to {vendor_name}?",
        placeholder="e.g., 2 years, 5 months",
    )

    # Question 2: Financial Reliability
    q2 = st.radio(
        f"How consistent is {vendor_name} in adhering to payment terms?",
        options=["Always on time", "Usually on time", "Sometimes delayed", "Frequently delayed", "Rarely on time"],
    )

    # Question 3: Communication and Responsiveness
    q3 = st.slider(
        f"How effective is {vendor_name} in communicating and resolving queries?",
        min_value=1,
        max_value=5,
        step=1,
        format="%d",
        help="1 - Poor, 5 - Excellent",
    )

    # Question 4: Order Management
    q4 = st.radio(
        f"Does {vendor_name} place and manage orders efficiently?",
        options=["Yes, always", "Most of the time", "Sometimes", "Rarely", "No"],
    )

    # Question 5: Challenges Faced
    q5 = st.text_area(
        f"Have you faced any challenges in your business relationship with {vendor_name}? "
        "If yes, please specify.",
        placeholder="Describe any specific challenges.",
    )

    # Question 6: Suggestions for Improvement
    q6 = st.text_area(
        f"What improvements, if any, would you suggest for {vendor_name} to enhance their business practices?",
        placeholder="Share your suggestions for improvement.",
    )

    # Submit button
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! It has been recorded.")
        st.write("### Your Responses:")
        st.write(f"1. Duration of Relationship: {q1}")
        st.write(f"2. Payment Adherence: {q2}")
        st.write(f"3. Communication Effectiveness: {q3}/5")
        st.write(f"4. Order Management: {q4}")
        st.write(f"5. Challenges Faced: {q5}")
        st.write(f"6. Suggestions for Improvement: {q6}")
