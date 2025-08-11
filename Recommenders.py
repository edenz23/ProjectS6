import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import LoanProfileBuilder
import OneHotEncodeLoanProfile


class LoanRecommender:
    def __init__(self, data_path):
        """
        LoanRecommender centralizes all recommenders and data.

        Loads:
        - Lenders one-hot encoded data
        - Loan profiles one-hot encoded data
        - K-means cluster profiles (each row describes a cluster's preference profile)

        Automatically assigns each lender to the most similar cluster.
        """
        self.data_path = data_path

        # Load data
        self.lenders = pd.read_csv(data_path + "One-Hot Encode - TreeTableLoaners - Category.csv")
        self.clusters = pd.read_csv(data_path + "K_Means_Groups_Analysis_With_Lender_Type.csv")
        try:
            self.loans = pd.read_csv(data_path + "LoanProfile_OneHot.csv")
        except Exception:
            LoanProfileBuilder.build_LoanProfile_file(data_path)
            OneHotEncodeLoanProfile.build_OneHotLoanProfile_file(data_path, self.lenders)
            self.loans = pd.read_csv(data_path + "LoanProfile_OneHot.csv")

        self.classifier = None
        self.common_cols = None

        # Assign clusters at initialization
        self._assign_lenders_to_clusters()

    # ------------------ Rule-Based Recommender ------------------ #
    def recommend_rule_based(self, lender_username, top_n=5):
        """Match loans to a lender based on direct feature overlap (weighted sum)."""
        lender_numeric = self.lenders.drop(columns=["Lender_Username", "AssignedCluster"]) \
                                     .select_dtypes(include=['number'])
        loan_numeric = self.loans.drop(columns=["Loan_ID"]).select_dtypes(include=['number'])

        common_cols = lender_numeric.columns.intersection(loan_numeric.columns)
        lender_vec = lender_numeric[self.lenders["Lender_Username"] == lender_username][common_cols].iloc[0]
        scores = (loan_numeric[common_cols] * lender_vec).sum(axis=1)

        result = self.loans.copy()
        result["score"] = scores
        return result.sort_values("score", ascending=False).head(top_n)

    # ------------------ Cosine Similarity Recommender ------------------ #
    def recommend_cosine(self, lender_username, top_n=5):
        """Match loans to a lender using cosine similarity between feature vectors."""
        lender_numeric = self.lenders.drop(columns=["Lender_Username", "AssignedCluster"]) \
                                     .select_dtypes(include=['number'])
        loan_numeric = self.loans.drop(columns=["Loan_ID"]).select_dtypes(include=['number'])

        common_cols = lender_numeric.columns.intersection(loan_numeric.columns)
        lender_vec = lender_numeric[self.lenders["Lender_Username"] == lender_username][common_cols].values
        sims = cosine_similarity(lender_vec, loan_numeric[common_cols].values)[0]

        result = self.loans.copy()
        result["similarity"] = sims
        return result.sort_values("similarity", ascending=False).head(top_n)

    # ------------------ Helper: Assign Lenders to Clusters ------------------ #
    def _assign_lenders_to_clusters(self):
        """
        Assign each lender to the most similar cluster profile using cosine similarity.
        """
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

    def show_training_progress(self, current, total):
        bar_size = 20
        step_size = total // bar_size
        current_progress = current // step_size
        bat = ''
        print(f"\r")


    # ------------------ ML Classifier Training ------------------ #
    def train_classifier(self, max_loans_per_lender=50):
        print("in trainer")
        """
        Train a RandomForestClassifier to predict lender-loan matches.
        Uses sampling to avoid generating billions of pairs.
        """

        common_cols = self.lenders.drop(columns=["Lender_Username", "AssignedCluster"], errors='ignore') \
            .select_dtypes(include=['number']).columns.intersection(
            self.loans.drop(columns=["Loan_ID"], errors='ignore') \
                .select_dtypes(include=['number']).columns
        )

        X, y = [], []

        loan_numeric = self.loans[common_cols]

        for i, lender_row in self.lenders.iterrows():
            if i % (len(self.lenders) // 20) == 0:  # here to show progress of training
                print(f"lender {i} out of {len(self.lenders)}")
            lender_cluster = lender_row["AssignedCluster"]
            cluster_pref = self.clusters[self.clusters['cluster'] == lender_cluster] \
                .drop(columns=[self.clusters.columns[-1], 'cluster'], errors='ignore') \
                .select_dtypes(include=['number']).iloc[0] \
                .reindex(common_cols, fill_value=0)

            cluster_pref_bin = (cluster_pref > 0.5).astype(int).values
            lender_vec = lender_row[common_cols].values

            # Score all loans once
            scores = loan_numeric.values @ cluster_pref_bin

            # Positive matches: top-N
            pos_idx = np.argsort(scores)[-max_loans_per_lender:]
            # Negative matches: random from the rest
            neg_candidates = np.setdiff1d(np.arange(len(scores)), pos_idx)
            neg_idx = np.random.choice(neg_candidates,
                                       size=min(max_loans_per_lender, len(neg_candidates)),
                                       replace=False)

            selected_idx = np.concatenate([pos_idx, neg_idx])

            for idx in selected_idx:
                loan_vec = loan_numeric.iloc[idx].values
                is_match = 1 if idx in pos_idx else 0
                X.append(np.concatenate([lender_vec, loan_vec]))
                y.append(is_match)

        X = np.array(X)
        y = np.array(y)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        self.classifier = clf
        self.common_cols = common_cols

    def recommend_ml(self, lender_username, top_n=5, prefilter_ratio=0.1):
        """
        Hybrid recommendation:
        1. Prefilter loans using cosine similarity (top prefilter_ratio %).
        2. Rank prefiltered loans with the ML classifier.

        prefilter_ratio: fraction of loans to keep before ML scoring (0.1 = 10%).
        """
        if not hasattr(self, "classifier"):
            raise ValueError("You must train the classifier first.")

        lender_row = self.lenders[self.lenders["Lender_Username"] == lender_username]
        if lender_row.empty:
            raise ValueError(f"Lender '{lender_username}' not found.")

        lender_vec = lender_row[self.common_cols].values[0]  # (num_features,)
        loan_matrix = self.loans[self.common_cols].values  # (num_loans, num_features)

        # Step 1: cosine similarity prefilter
        sims = cosine_similarity(lender_vec.reshape(1, -1), loan_matrix)[0]
        loans_with_sims = self.loans.copy()
        loans_with_sims["similarity"] = sims

        # Keep top X% by similarity
        prefilter_n = max(1, int(len(loans_with_sims) * prefilter_ratio))
        prefiltered_loans = loans_with_sims.nlargest(prefilter_n, "similarity")

        # Step 2: ML scoring on prefiltered loans
        lender_repeated = np.tile(lender_vec, (prefilter_n, 1))
        X_input = np.hstack([lender_repeated, prefiltered_loans[self.common_cols].values])
        proba = self.classifier.predict_proba(X_input)[:, 1]

        prefiltered_loans["ml_score"] = proba

        return prefiltered_loans.sort_values("ml_score", ascending=False).head(top_n)
