import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import imagehash
from PIL import Image
import os

# --- Список фильмов ---
movie_names = [
    "Inception", "Titanic", "Matrix", "Avengers", "Toy Story",
    "Interstellar", "Shrek", "Joker", "Avatar", "Up"
]

# --- KNN Recommendation ---
def knn_recommend(users, target='TargetUser', n_neighbors=2):
    try:
        if len(users[target]) != len(movie_names):
            return {"error": "Оценки TargetUser должны содержать ровно 10 значений."}
        
        X = np.array(list(users.values()))
        model = NearestNeighbors(n_neighbors=n_neighbors + 1)
        model.fit(X)
        distances, indices = model.kneighbors([users[target]])

        scores = defaultdict(list)
        for idx in indices[0][1:]:
            neighbor = list(users.keys())[idx]
            for i, rating in enumerate(users[neighbor]):
                if users[target][i] == 0:
                    scores[movie_names[i]].append(rating)

        recommended = {movie: round(np.mean(ratings), 2) for movie, ratings in scores.items()}
        return recommended
    except Exception as e:
        return {"error": f"KNN Error: {str(e)}"}

# --- Обучающие данные ---
def get_training_data(users_dict, film_index):
    X, y = [], []
    for user, ratings in users_dict.items():
        if user == "TargetUser":
            continue
        rating = ratings[film_index]
        if rating != 0:
            features = [r for i, r in enumerate(ratings) if i != film_index]
            X.append(features)
            y.append(rating if isinstance(rating, (float, int)) else 0)
    return X, y

def get_target_features(target_ratings, film_index):
    return [[r for i, r in enumerate(target_ratings) if i != film_index]]

# --- Предсказания (все модели) ---
def predict_preferences(users_dict, film_indices):
    results = {}
    try:
        for film_index in film_indices:
            movie = movie_names[film_index]
            results[movie] = {}

            X, y = get_training_data(users_dict, film_index)
            if len(X) == 0:
                results[movie]["error"] = "Недостаточно данных"
                continue

            target = get_target_features(users_dict["TargetUser"], film_index)

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X, y)
            pred_score = lr.predict(target)[0]
            results[movie]["Linear Regression"] = f"{round(pred_score, 2)}"

            # Классы: 1 если рейтинг >= 3, иначе 0
            y_binary = [1 if val >= 3 else 0 for val in y]

            if len(set(y_binary)) < 2:
                results[movie]["error"] = "Недостаточно разнообразия в данных для классификации"
                continue

            classifiers = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(max_depth=3),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(kernel='linear', probability=True),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            for name, clf in classifiers.items():
                clf.fit(X, y_binary)
                prediction = clf.predict(target)[0]
                results[movie][name] = "Понравится" if prediction == 1 else "Не понравится"

        return results
    except Exception as e:
        return {"error": f"Prediction Error: {str(e)}"}

# --- Apriori Algorithm ---
def apriori_recommend(transactions, watched=None, min_support=0.2, min_confidence=0.3):
    try:
        all_items = sorted(set(item for t in transactions for item in t))
        encoded_rows = []
        for transaction in transactions:
            encoded = {item: (item in transaction) for item in all_items}
            encoded_rows.append(encoded)

        df = pd.DataFrame(encoded_rows)

        freq_items = apriori(df, min_support=min_support, use_colnames=True)
        if freq_items.empty:
            return {"message": "Недостаточно совпадений. Попробуй снизить min_support или добавить больше данных."}

        rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
        if rules.empty:
            return {"message": "Правила не найдены. Попробуй снизить пороги."}

        recommendations = set()
        if watched:
            for _, row in rules.iterrows():
                if row['antecedents'].issubset(set(watched)):
                    recommendations.update(row['consequents'])

        recommendations = [movie for movie in recommendations if movie not in watched]
        return {
            "rules_found": len(rules),
            "recommendations": list(recommendations) if recommendations else ["Новых рекомендаций нет"]
        }
    except Exception as e:
        return {"error": f"Apriori Error: {str(e)}"}

# --- Анализ постера (CV + imagehash) ---
reference_dir = "reference_posters"
known_posters = {
    "Inception": "inception.jpg",
    "Titanic": "titanic.jpg",
    "Matrix": "matrix.jpg",
    "Avengers": "avengers.jpg",
    "Toy Story": "toystory.jpg",
    "Interstellar": "interstellar.jpg",
    "Shrek": "shrek.jpg",
    "Joker": "joker.jpg",
    "Avatar": "avatar.jpg",
    "Up": "up.jpg"
}

hashes = {}
try:
    for title, filename in known_posters.items():
        path = os.path.join(reference_dir, filename)
        if os.path.exists(path):
            img = Image.open(path)
            hashes[title] = imagehash.average_hash(img)
except Exception as e:
    print(f"Error loading reference posters: {str(e)}")

def analyze_poster(image_path):
    try:
        uploaded_hash = imagehash.average_hash(Image.open(image_path))
        min_distance = float('inf')
        best_match = None

        for title, ref_hash in hashes.items():
            dist = uploaded_hash - ref_hash
            if dist < min_distance:
                min_distance = dist
                best_match = title

        return best_match if best_match else "Фильм не найден"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# --- K-means кластеризация ---
def get_kmeans_cluster(users):
    try:
        data = np.array(list(users.values()))
        model = KMeans(n_clusters=2, random_state=0)
        model.fit(data)
        cluster = model.predict([users["TargetUser"]])[0]
        return {"cluster": int(cluster)}
    except Exception as e:
        return {"error": f"Clustering Error: {str(e)}"}

# --- PCA визуализация ---
def pca_visualize(users):
    try:
        data = np.array(list(users.values()))
        names = list(users.keys())
        pca = PCA(n_components=2)
        result = pca.fit_transform(data)

        plt.figure(figsize=(8, 6))
        for i, name in enumerate(names):
            x, y = result[i]
            plt.scatter(x, y)
            plt.text(x + 0.05, y + 0.05, name)

        plt.title("PCA — визуализация предпочтений пользователей")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("static", exist_ok=True)
        plt.savefig("static/pca_plot.png")
        plt.close()
        return True
    except Exception as e:
        print(f"PCA Error: {str(e)}")
        return False