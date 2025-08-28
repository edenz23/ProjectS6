import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import LoanProfileBuilder
import OneHotEncodeLoanProfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score

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
        except FileNotFoundError:
            LoanProfileBuilder.build_LoanProfile_file(data_path)
            OneHotEncodeLoanProfile.build_OneHotLoanProfile_file(data_path, self.lenders)
            self.loans = pd.read_csv(data_path + "LoanProfile_OneHot.csv")

        self.common_cols = None
        self.rf_clf = None
        self.lr_clf = None

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
    def train_classifier(self, max_loans_per_lender=50, models=("rf", "lr")):
        """
        Train ML models to predict lender-loan matches using sampled pairs.
        models: tuple containing any of ("rf","lr") to control which models to train.
        """
        # Ensure clusters are assigned
        if 'AssignedCluster' not in self.lenders.columns:
            self._assign_lenders_to_clusters()

        # Align common numeric columns once
        common_cols = self.lenders.drop(columns=["Lender_Username", "AssignedCluster"], errors='ignore') \
            .select_dtypes(include=['number']).columns.intersection(
            self.loans.drop(columns=["Loan_ID"], errors='ignore') \
                .select_dtypes(include=['number']).columns
        )
        self.common_cols = common_cols

        X, y = [], []
        loan_numeric = self.loans[common_cols]

        for i, lender_row in self.lenders.iterrows():
            # (Optional tiny progress prints are ok with big loops)
            if i % max(1, (len(self.lenders) // 20)) == 0:
                print(f"Sampling training pairs: lender {i} / {len(self.lenders)}")

            lender_cluster = lender_row["AssignedCluster"]
            cluster_pref = self.clusters[self.clusters['cluster'] == lender_cluster] \
                .drop(columns=[self.clusters.columns[-1], 'cluster'], errors='ignore') \
                .select_dtypes(include=['number']).iloc[0] \
                .reindex(common_cols, fill_value=0)

            cluster_pref_bin = (cluster_pref > 0.5).astype(int).values
            lender_vec = lender_row[common_cols].values

            # Score all loans once for this cluster profile
            scores = loan_numeric.values @ cluster_pref_bin

            # Positive matches: top-N
            pos_idx = np.argsort(scores)[-max_loans_per_lender:]
            # Negative matches: random from remaining
            neg_candidates = np.setdiff1d(np.arange(len(scores)), pos_idx)
            if len(neg_candidates) == 0:
                selected_idx = pos_idx
                neg_idx = np.array([], dtype=int)
            else:
                neg_idx = np.random.choice(
                    neg_candidates,
                    size=min(max_loans_per_lender, len(neg_candidates)),
                    replace=False
                )
                selected_idx = np.concatenate([pos_idx, neg_idx])

            # Build rows
            lender_repeated = np.tile(lender_vec, (selected_idx.shape[0], 1))
            loans_block = loan_numeric.iloc[selected_idx].values
            X_block = np.hstack([lender_repeated, loans_block])

            # Labels: 1 for positives, 0 for negatives
            y_block = np.zeros(selected_idx.shape[0], dtype=int)
            y_block[:len(pos_idx)] = 1  # first |pos_idx| rows correspond to positives

            X.append(X_block)
            y.append(y_block)

        # Stack once
        X = np.vstack(X)
        y = np.concatenate(y)

        # Train requested models
        if "rf" in models:
            self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.rf_clf.fit(X, y)
        if "lr" in models:
            # Logistic Regression with saga (handles large/dense/sparse), balanced classes, and more iters
            self.lr_clf = LogisticRegression(
                solver="saga",
                max_iter=1000,
                n_jobs=-1,
                class_weight="balanced"
            )
            self.lr_clf.fit(X, y)

        print(
            f"Trained models: {', '.join([m for m in models if (m == 'rf' and self.rf_clf) or (m == 'lr' and self.lr_clf)])}")

    def recommend_ml(self, lender_username, top_n=5, prefilter_ratio=0.1, model="rf"):
        """
        Hybrid recommendation:
        1) Cosine prefilter (top prefilter_ratio % by similarity)
        2) ML scoring using chosen model: "rf" (RandomForest) or "lr" (LogisticRegression)
        """
        if model == "rf" and self.rf_clf is None:
            raise ValueError("RandomForest not trained. Call train_classifier(models=('rf', ...)) first.")
        if model == "lr" and self.lr_clf is None:
            raise ValueError("LogisticRegression not trained. Call train_classifier(models=('lr', ...)) first.")
        if self.common_cols is None:
            raise ValueError("Model not initialized with common_cols. Train first.")

        lender_row = self.lenders[self.lenders["Lender_Username"] == lender_username]
        if lender_row.empty:
            raise ValueError(f"Lender '{lender_username}' not found.")

        lender_vec = lender_row[self.common_cols].values[0]
        loan_matrix = self.loans[self.common_cols].values

        # 1) Prefilter by cosine sim
        sims = cosine_similarity(lender_vec.reshape(1, -1), loan_matrix)[0]
        loans_with_sims = self.loans.copy()
        loans_with_sims["similarity"] = sims
        prefilter_n = max(1, int(len(loans_with_sims) * prefilter_ratio))
        prefiltered = loans_with_sims.nlargest(prefilter_n, "similarity")

        # 2) Vectorized ML scoring on prefiltered set
        lender_repeated = np.tile(lender_vec, (prefilter_n, 1))
        X_input = np.hstack([lender_repeated, prefiltered[self.common_cols].values])

        if model == "rf":
            proba = self.rf_clf.predict_proba(X_input)[:, 1]
        else:
            proba = self.lr_clf.predict_proba(X_input)[:, 1]

        prefiltered["ml_score"] = proba
        return prefiltered.sort_values("ml_score", ascending=False).head(top_n)


    def evaluate_recommender(self, lender_username, recommender_fn, k=10):
        """
        Evaluate a given recommender function using cluster preferences as ground truth.
        recommender_fn: function(lender_username, top_n) -> DataFrame with 'Loan_ID'
        """
        lender_row = self.lenders[self.lenders["Lender_Username"] == lender_username]
        if lender_row.empty:
            raise ValueError(f"Lender '{lender_username}' not found.")

        lender_cluster = lender_row["AssignedCluster"].iloc[0]
        common_cols = self.common_cols

        # cluster preference vector
        cluster_pref = self.clusters[self.clusters['cluster'] == lender_cluster] \
            .drop(columns=[self.clusters.columns[-1], 'cluster'], errors='ignore') \
            .select_dtypes(include=['number']).iloc[0] \
            .reindex(common_cols, fill_value=0)

        cluster_pref_bin = (cluster_pref > 0.5).astype(int).values
        loan_matrix = self.loans[common_cols].values

        # ground truth relevance: 1 if matches cluster prefs, else 0
        scores = loan_matrix @ cluster_pref_bin
        y_true = (scores > 0).astype(int)

        # recommendations
        recs = recommender_fn(lender_username, top_n=k)
        recommended_ids = recs["Loan_ID"].values
        y_pred = self.loans.set_index("Loan_ID").loc[recommended_ids]
        y_pred_labels = (y_pred[common_cols].values @ cluster_pref_bin > 0).astype(int)

        precision = y_pred_labels.sum() / k
        recall = y_pred_labels.sum() / max(1, y_true.sum())
        ndcg = ndcg_score([y_true], [scores])  # over full ranking

        return {"precision@k": precision, "recall@k": recall, "ndcg": ndcg}
