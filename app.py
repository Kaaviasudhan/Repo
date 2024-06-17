from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

app = Flask(__name__)

# Load TF-IDF Vectorizer and dataset
with open('models/tfv.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('models/tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)

with open('models/data.pkl', 'rb') as file:
    data = pickle.load(file)

# Function to recommend courses based on user input
def recommend_courses(user_input, data, top_n=5):
    # Transform the user input
    user_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get top N courses based on similarity
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    # Return the top N courses
    return data.iloc[top_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form['user_input']
        recommended_courses = recommend_courses(user_input, data)
        return render_template('index.html', user_input=user_input, recommended_courses=recommended_courses.to_dict(orient='records'))
    except Exception as e:
        # Log the error for debugging
        print(f"Error: {e}")
        return render_template('index.html', user_input=user_input, recommended_courses=[])

if __name__ == '__main__':
    app.run(debug=True)
