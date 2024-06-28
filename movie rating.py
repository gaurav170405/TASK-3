import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Create the user-item rating matrix
user_item_matrix = pd.DataFrame({
    'SHERSHAAH': [4, 0, 0, 5],
    'SAM BAHADUR': [5, 5, 0, 0],
    'MIRZAPUR': [0, 4, 0, 0],
    'VIVAH': [5, 4, 0, 0]
})

# Calculate the item-item similarity matrix using cosine similarity
item_similarity_matrix = pd.DataFrame(cosine_similarity(user_item_matrix.T), 
                                      index=user_item_matrix.columns, 
                                      columns=user_item_matrix.columns)

# Define the recommendation function
def get_recommendations(user_id):
    # Get the user's ratings
    user_ratings = user_item_matrix.iloc[user_id]
    # Compute the weighted ratings by multiplying user ratings with item similarity
    weighted_ratings = user_ratings.dot(item_similarity_matrix)
    # Sort the weighted ratings in descending order and get the movie titles
    recommendations = weighted_ratings.sort_values(ascending=False).index.tolist()
    return recommendations

# Get recommendations for a specific user
user_id = 3
recommendations = get_recommendations(user_id)
print(f"Recommendations for User {user_id}: {recommendations}")