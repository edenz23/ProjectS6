import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class LoanRecommender:
    def __init__(self, data_path):
        """
        LoanRecommender centralizes all recommenders and data.
        Loads:
        - Lenders one-hot encoded data
        - Loan profiles one-hot encoded data
        - K-means cluster profiles
        """
        self.data_path = data_path

        # Load data
        self.lenders = pd.read_csv(data_path + "One-Hot Encode - TreeTableLoaners - Category.csv")
        self.loans = pd.read_csv(data_path + "LoanProfile_OneHot.csv")
        self.clusters = pd.read_csv(data_path + "K_Means_Groups_Analysis_With_Lender_Type.csv")

        self.clf = None
        self.common_cols = None

    def recommend_rule_based(self, lender_username, top_n=5):
        lender_numeric = self.lenders.drop(columns=["Lender_Username"]).select_dtypes(include=['number'])
        loan_numeric = self.loans.drop(columns=["Loan_ID"]).select_dtypes(include=['number'])

        common_cols = lender_numeric.columns.intersection(loan_numeric.columns)
        lender_vec = lender_numeric[self.lenders["Lender_Username"] == lender_username][common_cols].iloc[0]
        scores = (loan_numeric[common_cols] * lender_vec).sum(axis=1)

        result = self.loans.copy()
        result["score"] = scores
        return result.sort_values("score", ascending=False).head(top_n)

    def recommend_cosine(self, lender_username, top_n=5):
        lender_numeric = self.lenders.drop(columns=["Lender_Username"]).select_dtypes(include=['number'])
        loan_numeric = self.loans.drop(columns=["Loan_ID"]).select_dtypes(include=['number'])

        common_cols = lender_numeric.columns.intersection(loan_numeric.columns)
        lender_vec = lender_numeric[self.lenders["Lender_Username"] == lender_username][common_cols].values
        sims = cosine_similarity(lender_vec, loan_numeric[common_cols].values)[0]

        result = self.loans.copy()
        result["similarity"] = sims
        return result.sort_values("similarity", ascending=False).head(top_n)

    def _assign_lenders_to_clusters(self):
        cluster_numeric = self.clusters.drop(columns=[self.clusters.columns[-1]])  # drop text summary
        cluster_ids = cluster_numeric['cluster']
        cluster_numeric = cluster_numeric.drop(columns=['cluster']).select_dtypes(include=['number'])

        lender_numeric = self.lenders.drop(columns=["Lender_Username"]).select_dtypes(include=['number'])
        assignments = []

        for _, lender_vec in lender_numeric.iterrows():
            sims = cosine_similarity([lender_vec], cluster_numeric.values)[0]
            best_cluster = cluster_ids.iloc[np.argmax(sims)]
            assignments.append(best_cluster)

        self.lenders['AssignedCluster'] = assignments

    def train_classifier(self):
        self._assign_lenders_to_clusters()

        lender_numeric = self.lenders.drop(columns=["Lender_Username", "AssignedCluster"]).select_dtypes(include=['number'])
        loan_numeric = self.loans.drop(columns=["Loan_ID"]).select_dtypes(include=['number'])
        common_cols = lender_numeric.columns.intersection(loan_numeric.columns)
        self.common_cols = common_cols

        X, y = [], []

        for _, lender_row in self.lenders.iterrows():
            lender_vec = lender_row[common_cols].values
            lender_cluster = lender_row['AssignedCluster']

            cluster_pref = self.clusters[self.clusters['cluster'] == lender_cluster] \
                .drop(columns=[self.clusters.columns[-1], 'cluster']) \
                .select_dtypes(include=['number']).iloc[0]

            # Align with loan feature columns
            cluster_pref = cluster_pref.reindex(loan_numeric.columns, fill_value=0)

            matches = (loan_numeric.values * (cluster_pref > 0.5).astype(int)).sum(axis=1)
            threshold = np.median(matches)
            labels = (matches >= threshold).astype(int)

            for loan_vec, label in zip(loan_numeric.values, labels):
                X.append(np.concatenate([lender_vec, loan_vec]))
                y.append(label)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"Classifier trained. Test accuracy: {acc:.3f}")

        self.clf = clf

    def recommend_classifier(self, lender_username, top_n=5):
        if self.clf is None or self.common_cols is None:
            raise ValueError("Classifier not trained. Call train_classifier() first.")

        lender_row = self.lenders[self.lenders["Lender_Username"] == lender_username]
        lender_numeric = lender_row.drop(columns=["Lender_Username", "AssignedCluster"]) \
                                   .select_dtypes(include=['number'])[self.common_cols].values
        loan_numeric = self.loans.drop(columns=["Loan_ID"]).select_dtypes(include=['number'])[self.common_cols].values

        X_pred = [np.concatenate([lender_numeric.flatten(), loan_vec]) for loan_vec in loan_numeric]
        probs = self.clf.predict_proba(X_pred)[:, 1]

        result = self.loans.copy()
        result["probability"] = probs
        return result.sort_values("probability", ascending=False).head(top_n)

