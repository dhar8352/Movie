import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dump import dump, load

# Load your movie rating dataset (assuming it has columns: userId, movieId, rating)
# Replace 'ratings.csv' with your dataset filename
df = pd.read_csv('ratings.csv')

# Create a Surprise Reader object to specify the rating scale
reader = Reader(rating_scale=(0.5, 5))

# Load the dataset into the Surprise format
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize the User-Based Collaborative Filtering model
sim_options = {
    'name': 'cosine',  # Similarity metric (can also use 'pearson' or others)
    'user_based': True  # User-Based Collaborative Filtering
}
model = KNNBasic(sim_options=sim_options)

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model's performance
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# Replace 'your_user_id' with the actual user ID for recommendations
user_id = 'your_user_id'
user_ratings = df[df['userId'] == user_id][['movieId', 'rating']]

# Get movie recommendations for the user
top_n = 10  # Number of recommendations
user_movies = set(user_ratings['movieId'])
not_rated_movies = [movie for movie in trainset.all_items() if movie not in user_movies]

# Predict ratings for unrated movies
predictions = [model.predict(user_id, movie) for movie in not_rated_movies]

# Sort predictions by estimated rating
predictions.sort(key=lambda x: x.est, reverse=True)

# Get top-rated movie recommendations
top_movie_ids = [prediction.iid for prediction in predictions[:top_n]]

# Print recommended movie IDs
print(f'Recommended Movie IDs for User {user_id}: {top_movie_ids}')

# Save the trained model to a file (optional)
model_file = 'movie_recommendation_model.pkl'
dump(model_file, algo=model)

# Load the model from a file (optional)
loaded_model = load(model_file)
