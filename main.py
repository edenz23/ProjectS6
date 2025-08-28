from Recommenders import LoanRecommender

data_path = "data/"
rec = LoanRecommender(data_path)

print("\n--- Rule Based ---")
print(rec.recommend_rule_based("2184", top_n=5))

print("\n--- Cosine Similarity ---")
print(rec.recommend_cosine("2184", top_n=5))

# ML: train both RF and Logistic
rec.train_classifier(max_loans_per_lender=50, models=("rf","lr"))

print("\n--- ML (RandomForest) ---")
print(rec.recommend_ml("2184", top_n=5, prefilter_ratio=0.05, model="rf"))

print("\n--- ML (Logistic Regression) ---")
print(rec.recommend_ml("2184", top_n=5, prefilter_ratio=0.05, model="lr"))