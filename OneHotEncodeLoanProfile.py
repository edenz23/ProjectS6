import pandas as pd

"""
One-Hot Encode Loan Profiles to Match Lender Data

- Load LoanProfile.csv and the lender one-hot file.
- One-hot encode the categorical loan features using the same column names
   as in the lender one-hot file.
- Outputs LoanProfile_OneHot.csv for direct cosine similarity computation.
"""


def build_OneHotLoanProfile_file(data_path, lender_onehot):
    loan_profile = pd.read_csv(data_path + "LoanProfile.csv")

    lender_feature_cols = lender_onehot.columns.drop("Lender_Username")

    loan_onehot = pd.DataFrame(0, index=loan_profile.index, columns=lender_feature_cols)

    # map Loan_Sector
    for col in lender_feature_cols:
        if col.startswith("Loan_Sector_"):
            sector = col.replace("Loan_Sector_", "")
            loan_onehot[col] = (loan_profile["Loan_Sector"] == sector).astype(int)

    # map Borrower_Continent
    for col in lender_feature_cols:
        if col.startswith("Borrower_Continent_"):
            cont = col.replace("Borrower_Continent_", "")
            loan_onehot[col] = (loan_profile["Borrower_Continent"] == cont).astype(int)

    # map Average_Loan_Amount_Range
    for col in lender_feature_cols:
        if col.startswith("Average_Loan_Amount_Range_"):
            rng = col.replace("Average_Loan_Amount_Range_", "")
            loan_onehot[col] = (loan_profile["Average_Loan_Amount_Range"] == rng).astype(int)

    # map Gender_Loan_Preference
    for col in lender_feature_cols:
        if col.startswith("Gender_Loan_Preference_"):
            gender_val = col.replace("Gender_Loan_Preference_", "")
            loan_onehot[col] = (loan_profile["Gender_Loan_Preference"] == gender_val).astype(int)

    loan_onehot.insert(0, "Loan_ID", loan_profile["Loan_ID"])
    loan_onehot.to_csv(data_path + "LoanProfile_OneHot.csv", index=False)
    print("LoanProfile_OneHot.csv created with", loan_onehot.shape[0], "rows and", loan_onehot.shape[1], "columns.")
