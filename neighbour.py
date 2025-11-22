import streamlit as st

# Set the title of the page
st.title("Vendor Loan Eligibility Survey")

# Introduction
st.write(
    "As part of the loan evaluation process, we are gathering feedback on [Vendor Name] "
    "to assess their business practices, financial reliability, and community standing. "
    "Your honest feedback is essential. Thank you for your time."
)

# Questions
vendor_name = st.text_input("Vendor Name:", placeholder="Enter the vendor name")

if vendor_name:
    st.subheader(f"Feedback for {vendor_name}")

    # Question 1: Business Operations
    q1 = st.text_input(
        f"How long have you known or worked with {vendor_name}?",
        placeholder="e.g., 2 years, 5 months",
    )

    # Question 2: Financial Practices
    q2 = st.radio(
        f"How consistent is {vendor_name} in fulfilling their financial obligations, such as payments or credits?",
        options=["Always on time", "Usually on time", "Sometimes delayed", "Frequently delayed", "Rarely on time"],
    )

    # Question 3: Business Stability
    q3 = st.radio(
        f"How stable is {vendor_name}'s business in terms of regular operations and customer base?",
        options=["Very stable", "Moderately stable", "Unstable", "Very unstable"],
    )

    # Question 4: Reputation
    q4 = st.slider(
        f"How would you rate {vendor_name}'s reputation in the community and among business associates?",
        min_value=1,
        max_value=5,
        step=1,
        format="%d",
        help="1 - Poor, 5 - Excellent",
    )

    # Question 5: Challenges or Concerns
    q5 = st.text_area(
        f"Have you observed any challenges or concerns regarding {vendor_name}'s ability to manage finances or maintain business stability? "
        "If yes, please provide details.",
        placeholder="Describe any specific concerns.",
    )

    # Question 6: Loan Eligibility Opinion
    q6 = st.radio(
        f"Do you believe {vendor_name} is financially capable and reliable to handle a loan responsibly?",
        options=["Yes", "No", "Not Sure"],
    )

    # Submit button
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! It has been recorded.")
        st.write("### Your Responses:")
        st.write(f"1. Duration of Relationship: {q1}")
        st.write(f"2. Financial Practices: {q2}")
        st.write(f"3. Business Stability: {q3}")
        st.write(f"4. Reputation: {q4}/5")
        st.write(f"5. Challenges or Concerns: {q5}")
        st.write(f"6. Loan Eligibility Opinion: {q6}")
