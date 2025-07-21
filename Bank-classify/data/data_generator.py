import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define categories and corresponding description templates
CATEGORIES = {
    "Food": [
        "Zomato Order", "Swiggy Order", "Restaurant Bill", "Cafe Coffee Day", 
        "McDonald's", "Burger King", "Pizza Hut", "Dominos Pizza"
    ],
    "Travel": [
        "Ola Ride", "Uber Ride", "Train Ticket", "Flight Ticket", "Bus Ticket",
        "Hotel Booking", "Goibibo", "MakeMyTrip"
    ],
    "Salary": ["Monthly Salary"],
    "Rent": ["House Rent"],
    "Shopping": [
        "Amazon Purchase", "Flipkart Purchase", "Myntra Shopping", "Zara Store",
        "H&M Store", "Big Bazaar", "Dmart"
    ],
    "Transfer": ["Bank Transfer", "UPI Transfer", "IMPS Transfer"],
    "Others": ["Miscellaneous Expense", "Cash Withdrawal", "ATM Withdrawal"]
}

def generate_synthetic_data(num_records=1200):
    """
    Generates synthetic bank transaction data.

    Args:
        num_records (int): The number of records to generate.

    Returns:
        pd.DataFrame: A DataFrame with synthetic transaction data.
    """
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_records):
        category = random.choice(list(CATEGORIES.keys()))
        description_template = random.choice(CATEGORIES[category])
        date = start_date + timedelta(seconds=random.randint(0, 365*24*60*60))
        amount = round(random.uniform(100, 5000), 2)
        
        # Add some noise/variation to descriptions
        if " " in description_template:
            words = description_template.split()
            if random.random() < 0.3: # 30% chance to add a transaction ID
                words.append(f"ID:{random.randint(10000, 99999)}")
            description = " ".join(words)
        else:
            description = description_template

        data.append({
            "Date": date.strftime('%Y-%m-%d %H:%M:%S'),
            "Description": description,
            "Amount": amount,
            "Category": category
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/transactions.csv", index=False)
    print("Synthetic data generated and saved to data/transactions.csv") 