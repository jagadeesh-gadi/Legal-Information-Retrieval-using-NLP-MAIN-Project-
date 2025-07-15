import os
import json
import random
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
SUMMARY_FOLDER = "summaries"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

# Load pre-trained embeddings
sentence_embeddings = torch.load(r"C:\Users\jagadeesh\OneDrive\Desktop\All projects\main project\Main-project\legalbert_sentence_embeddings.pt")
word_embeddings = torch.load(r"C:\Users\jagadeesh\OneDrive\Desktop\All projects\main project\Main-project\legal_text_embedding1.pt")
topic_embeddings = torch.load(r"C:\Users\jagadeesh\OneDrive\Desktop\All projects\main project\Main-project\legal_text_embeddingtopic.pt")

# Load TF-IDF weights
try:
    with open("tfidf_weights111.json", "r", encoding="utf-8") as f:
        tfidf_weights = json.load(f)
    tfidf_weights = np.array(tfidf_weights, dtype=np.float32)
except:
    tfidf_weights = np.random.rand(50).astype(np.float32)

# Ensure valid TF-IDF weights
if tfidf_weights.ndim != 1:
    raise ValueError("TF-IDF Weights should be a 1D array!")

# Function to generate a summary
def generate_summary(text):
    sentences = text.split(". ")
    paragraphs = [sentences[i:i+5] for i in range(0, len(sentences), 5)]
    summary_indices = sorted(random.sample(range(len(paragraphs)), min(3, len(paragraphs))))
    return "\n".join([" ".join(paragraphs[i]) for i in summary_indices])

# Function to compute similarity
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            summary = generate_summary(text_content)
            summary_path = os.path.join(SUMMARY_FOLDER, f"{file.filename}_summary.txt")
            
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)

            similarity_score = calculate_cosine_similarity(text_content, summary)
            return render_template("result.html", filename=file.filename, summary=summary, similarity_score=similarity_score)
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
