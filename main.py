from Recommenders import LoanRecommender

data_path = "data/"
rec = LoanRecommender(data_path)

#print("\n--- Rule Based ---")
#print(rec.recommend_rule_based("2184", top_n=5))

#print("\n--- Cosine Similarity ---")
#print(rec.recommend_cosine("2184", top_n=5))

print("\n--- ML Classifier ---")
rec.train_classifier()
print(rec.recommend_ml("2184", top_n=5))
