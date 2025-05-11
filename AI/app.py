from flask import Flask, render_template, request, jsonify, send_from_directory
import json
from recommender import (
    knn_recommend,
    predict_preferences,
    apriori_recommend,
    get_kmeans_cluster,
    pca_visualize,
    analyze_poster,
    movie_names
)
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# --- Данные пользователей ---
users_data = {
    "User1": [5, 0, 4, 2, 3, 4, 5, 0, 2, 1],
    "User2": [4, 5, 0, 3, 4, 2, 0, 3, 4, 3],
    "User3": [0, 3, 4, 5, 0, 3, 4, 2, 5, 1],
    "User4": [3, 4, 0, 5, 4, 1, 2, 5, 3, 4],
    "TargetUser": [0] * 10
}

# --- Главная страница ---
@app.route('/')
def index():
    return render_template('index.html', movies=movie_names)

# --- KNN рекомендации ---
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_ratings = request.get_json()
        users_data["TargetUser"] = [user_ratings.get(movie, 0) for movie in movie_names]
        recommendations = knn_recommend(users_data)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Предсказание понравится/не понравится ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_ratings = request.get_json()
        users = users_data.copy()
        users["TargetUser"] = [user_ratings.get(movie, 0) for movie in movie_names]
        film_indices = [i for i, r in enumerate(users["TargetUser"]) if r == 0]
        predictions = predict_preferences(users, film_indices)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Apriori рекомендации ---
@app.route('/apriori', methods=['POST'])
def apriori():
    try:
        user_ratings = request.get_json()
        watched_movies = [movie for movie, rating in user_ratings.items() if rating >= 3]
        transactions = [
            ["Inception", "Titanic", "Matrix", "Avengers", "Toy Story"],
            ["Interstellar", "Titanic", "Toy Story", "Avatar"],
            ["Matrix", "Avengers", "Toy Story", "Joker"],
            ["Inception", "Matrix", "Avengers", "Shrek"],
            ["Titanic", "Toy Story", "Up", "Interstellar"],
            ["Shrek", "Joker", "Up", "Avatar"],
            ["Inception", "Interstellar", "Joker", "Up"]
        ]
        apriori_results = apriori_recommend(transactions, watched=watched_movies)
        return jsonify(apriori_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Анализ постера ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'poster' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['poster']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = analyze_poster(filepath)
        return jsonify({"title": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- K-means кластеризация ---
@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        user_ratings = request.get_json()
        users_data["TargetUser"] = [user_ratings.get(movie, 0) for movie in movie_names]
        cluster_result = get_kmeans_cluster(users_data)
        return jsonify(cluster_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- PCA визуализация ---
@app.route('/pca', methods=['POST'])
def pca():
    try:
        user_ratings = request.get_json()
        users_data["TargetUser"] = [user_ratings.get(movie, 0) for movie in movie_names]
        success = pca_visualize(users_data)
        if success:
            return jsonify({"plot": "/static/pca_plot.png"})
        return jsonify({"error": "PCA visualization failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Serve static files ---
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)