import pandas as pd
import gender_guesser.detector as gender

"""
LoanProfile.csv Builder with Gender Inference
---------------------------------------------
Purpose:
Transform raw loan data (bigml.csv) into the same categorical format as lender data
(TreeTableLoaners), enabling direct profile matching for recommendations.

Enhancement:
Since raw loan data lacks explicit borrower gender, we infer it from the borrower's
first name using the 'gender-guesser' library. This provides an additional matching
signal for the recommendation system.
"""


def build_LoanProfile_file(data_path):
    bigml_path = data_path + "bigml.csv"
    df = pd.read_csv(bigml_path)

    def loan_amount_bin(amount):
        if amount <= 500: return "1-500"
        elif amount <= 1000: return "501-1000"
        elif amount <= 1500: return "1001-1500"
        elif amount <= 2000: return "1501-2000"
        elif amount <= 2500: return "2001-2500"
        elif amount <= 3000: return "2501-3000"
        elif amount <= 3500: return "3001-3500"
        elif amount <= 4000: return "3501-4000"
        elif amount <= 4500: return "4001-4500"
        elif amount <= 5000: return "4501-5000"
        elif amount <= 9999: return "5001-9999"
        else: return "10000+"

    continent_map = {
        # Africa
        "Uganda": "Africa", "Kenya": "Africa", "Rwanda": "Africa", "Tanzania": "Africa",
        # East Asia
        "Philippines": "East Asia", "Vietnam": "East Asia", "Cambodia": "East Asia",
        # Europe
        "Albania": "Europe", "Ukraine": "Europe", "Kosovo": "Europe",
        # Middle East
        "Lebanon": "Middle East", "Jordan": "Middle East", "Iraq": "Middle East",
        # North and Central America
        "Honduras": "North and Central America", "El Salvador": "North and Central America", "Guatemala": "North and Central America",
        # Oceania
        "Samoa": "Oceania",
        # South America
        "Peru": "South America", "Ecuador": "South America", "Bolivia": "South America"
    }

    detector = gender.Detector(case_sensitive=False)

    def infer_gender(name):
        """Infer borrower gender from first name using gender-guesser."""
        if pd.isna(name) or not isinstance(name, str):
            return "equal"  # unknown
        first_name = name.strip().split()[0]  # take first word
        g = detector.get_gender(first_name)
        if g in ("male", "mostly_male"):
            return "more to men"
        elif g in ("female", "mostly_female"):
            return "more to women"
        else:
            return "equal"

    loan_profile = pd.DataFrame({
        "Loan_ID": df["id"],
        "Gender_Loan_Preference": df["Name"].apply(infer_gender),
        "Average_Age_Range": "unknown",  # age data doesnt exist in loan seekers dataset
        "Average_Loan_Amount_Range": df["Loan Amount"].apply(loan_amount_bin),
        "Loan_Sector": df["Sector"],
        "Borrower_Continent": df["Country"].map(continent_map).fillna("Other")
    })

    loan_profile.to_csv(data_path + "LoanProfile.csv", index=False)
    print("LoanProfile.csv created with", len(loan_profile), "rows, with gender inference.")
