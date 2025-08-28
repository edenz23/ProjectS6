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

rec = LoanRecommender("data/")
rec.train_classifier(models=("rf","lr"))

print("--- Evaluation for lender 2184 ---")
print("Rule-based:", rec.evaluate_recommender("2184", rec.recommend_rule_based, k=10))
print("Cosine:", rec.evaluate_recommender("2184", rec.recommend_cosine, k=10))
print("ML-RF:", rec.evaluate_recommender("2184", lambda u,k: rec.recommend_ml(u, top_n=k, model="rf"), k=10))
print("ML-LR:", rec.evaluate_recommender("2184", lambda u,k: rec.recommend_ml(u, top_n=k, model="lr"), k=10))
