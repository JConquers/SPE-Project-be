import pandas as pd
import random

# Function to generate synthetic data
def generate_creditworthiness_data(num_entries):
    data = []
    for _ in range(num_entries):
        # Generate data for each column
        community_reputation = random.choice(["Excellent", "Good", "Average", "Poor", "Unknown"])
        business_activity_observations = random.randint(5, 500)  # Number of daily customers
        informal_credit_relationships = random.choice(["Good Credit History", "Has Defaults", "Unknown"])
        asset_ownership = random.choice([
            "Sewing Machine", "Livestock", "Vehicle", "Shop Equipment", "None", 
            "Multiple Assets"
        ])
        utility_and_bill_payment = random.choice([
            "On Time", "Delayed Payments", "No Records", "Partially Paid"
        ])
        local_market_standing = random.choice([
            "Highly Popular", "Moderately Popular", "Average", "Low Footfall", "New Business"
        ])
        participation_in_welfare = random.choice([
            "Enrolled in MUDRA", "Enrolled in SHG", "None", "Multiple Programs"
        ])
        family_and_dependents = random.randint(1, 10)
        seasonal_and_env_impact = random.choice([
            "High Seasonal Variation", "Moderate Seasonal Variation", 
            "Minimal Seasonal Variation", "Weather Sensitive"
        ])
        behavioral_indicators = random.choice([
            "High Risk-Taker", "Cautious", "Good Problem-Solver", "Indecisive"
        ])
        business_location = random.choice([
            "Prime Location", "Moderate Traffic", "Low Traffic Area", 
            "Home-Based", "Remote Area"
        ])
        supplier_review = random.randint(1, 50)  # Number of regular connections
        customer_review = random.randint(1, 50)
        observed_consistency = random.choice(["Regularly Open", "Irregular Operations"])
        training_and_skill_dev = random.choice([
            "Completed Skill Training", "Ongoing Training", 
            "No Formal Training", "Multiple Certifications"
        ])
        debt_and_liabilities = round(random.uniform(1000, 50000), 2)  # INR
        savings_and_insurance = random.choice([
            "Chit Fund Savings", "Insurance Policy", "None", "Both Savings and Insurance"
        ])
        aadhaar_card = random.choice(["Verified", "Not Verified"])
        ration_card = random.choice(["BPL", "APL", "Antyodaya", "Not Provided"])
        voter_id_card = random.choice(["Verified", "Not Verified"])

        # Debt-to-Income Ratio logic
        debt_to_income = random.uniform(0.1, 0.9)  # Random ratio between 0.1 and 0.9

        # Loan Approval Logic with expanded conditions (adjusted for more "Approved" cases)
        if (
            (community_reputation == "Excellent" or community_reputation == "Good") and
            business_activity_observations > 50 and
            (informal_credit_relationships == "Good Credit History" or informal_credit_relationships == "Unknown") and
            utility_and_bill_payment == "On Time" and
            debt_and_liabilities < 30000 and
            (savings_and_insurance == "Both Savings and Insurance" or savings_and_insurance == "Insurance Policy") and
            business_location != "Remote Area" and
            debt_to_income < 0.6 and
            (asset_ownership in ["Vehicle", "Shop Equipment", "Multiple Assets", "Livestock"]) and
            (behavioral_indicators != "High Risk-Taker" or behavioral_indicators == "Cautious") and
            training_and_skill_dev != "No Formal Training" and
            observed_consistency == "Regularly Open"
        ):
            loan_approval = "Approved"
        else:
            loan_approval = "Denied"

        # Create entry dictionary
        entry = {
            "Community Reputation": community_reputation,
            "Business Activity Observations": business_activity_observations,
            "Informal Credit Relationships": informal_credit_relationships,
            "Asset Ownership": asset_ownership,
            "Utility and Bill Payment Patterns": utility_and_bill_payment,
            "Local Market Standing": local_market_standing,
            "Participation in Welfare or Support Programs": participation_in_welfare,
            "Family and Dependents": family_and_dependents,
            "Seasonal and Environmental Impact": seasonal_and_env_impact,
            "Behavioral Indicators": behavioral_indicators,
            "Business Location and Infrastructure": business_location,
            "Customer and Supplier Relationships": customer_supplier_relationships,
            "Observed Consistency": observed_consistency,
            "Training and Skill Development": training_and_skill_dev,
            "Debt and Liabilities": debt_and_liabilities,
            "Savings and Insurance": savings_and_insurance,
            "Aadhaar Card": aadhaar_card,
            "Ration Card": ration_card,
            "Voter ID Card": voter_id_card,
            "Loan Approval": loan_approval  # Loan approval decision
        }
        
        # Append entry to data
        data.append(entry)

    return pd.DataFrame(data)

# Generate synthetic data for 100 entries
df = generate_creditworthiness_data(100000)

# Save to a CSV file
df.to_csv("synthetic_creditworthiness_with_simple_loan_approval.csv", index=False)

# Display the first few rows
print(df.head())
