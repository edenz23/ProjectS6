from Recommenders import LoanRecommender

rec = LoanRecommender(data_path="data/")

#print("\n--- Rule Based ---")
#print(rec.recommend_rule_based("2184"))

#print("\n--- Cosine Similarity ---")
#print(rec.recommend_cosine("2184"))

print("\n--- ML Classifier ---")
rec.train_classifier()
print(rec.recommend_classifier("2184"))
