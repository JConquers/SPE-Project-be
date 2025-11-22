import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate synthetic time series data for one user
def generate_risk_data(num_entries):
    data = []
    
    # Start date for time series
    start_date = datetime(2022, 1, 1)

    for i in range(num_entries):
        # Generate data for each column
        community_reputation = random.choice(["Excellent", "Good", "Average", "Poor", "Unknown"])
        business_activity_observations = random.randint(5, 500)  # Number of daily customers
        informal_credit_relationships = random.choice(["Good Credit History", "Has Defaults", "Unknown"])
        asset_ownership = random.choice([
            "Sewing Machine", "Livestock", "Vehicle", "Shop Equipment", "None", "Multiple Assets"
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
        customer_supplier_relationships = random.randint(1, 50)  # Number of regular connections
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

        # Risk assessment logic (target variable)
        if (
            (community_reputation == "Poor" or debt_and_liabilities > 30000 or debt_to_income > 0.7 or behavioral_indicators == "High Risk-Taker")
        ):
            risk = "Yes"
        else:
            risk = "No"

        # Time-series entry with date
        date = start_date + timedelta(days=i)

        # Create entry dictionary for one time point
        entry = {
            "Date": date,
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
            "Risk": risk  # Target variable: Risk (Yes/No)
        }
        
        # Append entry to data
        data.append(entry)

    return pd.DataFrame(data)

# Generate synthetic time series data for 100 entries
df = generate_risk_data(100)

# Save to a CSV file
df.to_csv("synthetic_risk_assessment_data.csv", index=False)

# Display the first few rows
print(df.head())
