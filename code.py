from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text=["London Paris London","Paris Paris London"]
cv=CountVectorizer()
count_matrix=cv.fit_transform(text)
simillarity_scores=cosine_similarity(count_matrix)
print(simillarity_scores)
