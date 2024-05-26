"""
Movie Recommendation System
"""
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unittest
from unittest.mock import patch

# Load the movie dataset
movies = pd.read_csv(r"E:/movie_recommendation_system/NetFlix movie recomendation system non-web/NetFlix_movie_recomendation_system-non-web--main/Netflix_movie_recommendation_system_non-web/dataset_used/dataset.csv")

# Data exploration
print("Dataset description:")
print(movies.describe())
print("\nNull values count:")
print(movies.isnull().sum())
print("\nDataset columns:")
print(movies.columns)

print("\n" + "-" * 80 + "\n")

# Select relevant features for movie recommendation
movies = movies[['id', 'title', 'overview', 'genre']]
movies['tags'] = movies['overview'] + movies['genre']  # Combine 'overview' and 'genre' into a single 'tags' column
newdata = movies.drop(columns=['overview', 'genre'])   # Drop original 'overview' and 'genre' columns

print("-------------------------------MOVIE RECOMMENDATION SYSTEM-------------------------------")
print("\n")

# Text vectorization using CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(newdata['tags'].values.astype('U')).toarray()  # Convert text data to vector
print("Vectorized data shape:", vector.shape)

print("\n" + "-" * 80 + "\n")

# Calculate cosine similarity between movie vectors
similarity = cosine_similarity(vector)

# Movie recommendation function
def recommand(movie_title):
    """
    Recommends similar movies based on the input movie title.
    
    Args:
        movie_title (str): The title of the movie to find recommendations for.
        
    Returns:
        list: A list of recommended movie titles, or None if no recommendations are found.
    """
    index = newdata[newdata['title'] == movie_title].index  # Find index of the input movie title
    if not index.empty:
        index = index[0]
        if index < len(similarity):
            distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
            recommended_movies = [newdata.iloc[i[0]].title for i in distance[1:6]]  # Exclude the input movie
            return recommended_movies
    return None

# Function to list a sample of movie titles for testing
def list_random_movie_names(n=20):
    """
    Returns a random list of movie titles from the dataset.
    
    Args:
        n (int): The number of movie titles to return (default is 20).
        
    Returns:
        list: A random list of movie titles.
    """
    return newdata['title'].sample(n=n).tolist()

# Function to calculate accuracy of recommendations
def calculate_accuracy(test_movie_titles):
    """
    Calculates the accuracy of the recommendation system.
    
    Args:
        test_movie_titles (list): A list of movie titles to test.
        
    Returns:
        float: The accuracy of the recommendations.
    """
    correct_recommendations = 0
    total_recommendations = 0

    # Evaluate each test movie title using confusion matrix
    for title in test_movie_titles:
        recommended_movies = recommand(title)
        if recommended_movies:
            input_movie_genres = set(movies[movies['title'] == title]['tags'].values[0].split())
            for rec_movie in recommended_movies:
                rec_movie_genres = set(movies[movies['title'] == rec_movie]['tags'].values[0].split())
                if input_movie_genres & rec_movie_genres:
                    correct_recommendations += 1
            total_recommendations += len(recommended_movies)
    
    accuracy = (correct_recommendations / total_recommendations) * 100 if total_recommendations > 0 else 0
    return accuracy

# Testing the recommendation system
print("Total of 10000 movie titles are here! For testing, 20 random movies are listed below:")
print("-" * 80)
random_movie_names = list_random_movie_names()
for name in random_movie_names:
    print("- " + name)

print("-" * 80)
user_input = input("Which movie would you like to watch? ")

recommended_movies = recommand(user_input)

if recommended_movies:
    print("You might also like these movies:")
    print("-" * 80)
    for movie in recommended_movies:
        print("- " + movie.upper())
else:
    print("Sorry, no recommendations found.")

print("-" * 80)
print("------------------------------------------------------------------------------------")

# Unit testing and integration testing

class TestMovieRecommendationSystem(unittest.TestCase):

    def test_list_random_movie_names(self):
        """Test the list_random_movie_names function."""
        movie_names = list_random_movie_names(10)
        self.assertEqual(len(movie_names), 10)
        for name in movie_names:
            self.assertIn(name, newdata['title'].values)
    
    @patch('builtins.input', return_value='The Matrix')
    def test_recommand(self, mock_input):
        """Test the recommand function."""
        recommended_movies = recommand('The Matrix')
        self.assertIsNotNone(recommended_movies)
        self.assertTrue(len(recommended_movies) > 0)
    
    @patch('builtins.input', return_value='Non-Existent Movie')
    def test_recommand_no_recommendations(self, mock_input):
        """Test the recommand function with a non-existent movie."""
        recommended_movies = recommand('Non-Existent Movie')
        self.assertIsNone(recommended_movies)
    
    def test_calculate_accuracy(self):
        """Test the calculate_accuracy function."""
        test_titles = list_random_movie_names(10)
        accuracy = calculate_accuracy(test_titles)
        self.assertTrue(0 <= accuracy <= 100)

def run_tests():
    """Run unit and integration tests and display the results."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestMovieRecommendationSystem)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    total_tests = test_result.testsRun
    failures = len(test_result.failures)
    errors = len(test_result.errors)
    passed = total_tests - (failures + errors)
    accuracy = (passed / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "-" * 80 + "\n")
    print("Unit and Integration Test Results:")
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 80)

    # Calculate and display model accuracy
    print("Calculating model accuracy based on genre similarity...")
    test_titles = list_random_movie_names(100)
    model_accuracy = calculate_accuracy(test_titles)
    print(f"Model accuracy: {model_accuracy:.2f}%")

if __name__ == '__main__':
    run_tests()
