import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are available
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean and preprocess the text
def clean_text(text):
    """
    Convert text to lowercase, remove special characters, and strip whitespace.
    """
    text = text.lower()  # Convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Remove special characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def preprocess_text(text, stop_words):
    """
    Remove stop words from text.
    """
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to load GloVe embeddings from a file
def load_glove_embeddings(file_path):
    """
    Load GloVe word embeddings from the specified file path.
    """
    embeddings = {}
    skipped_lines = 0
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
            except ValueError:
                skipped_lines += 1
    print(f"Total lines skipped: {skipped_lines}")
    print(f"Total valid vectors: {len(embeddings)}")
    return embeddings

# Function to compute the average GloVe vector for a text
def get_average_glove_vector(text, embeddings, dim=300):
    """
    Calculate the average GloVe vector for a given text.
    """
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# Function to calculate TF-IDF vectors for a list of texts
def calculate_tfidf_vectors(texts, stop_words):
    """
    Compute TF-IDF vectors for a given list of texts.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer

# Load stopwords
stop_words = set(stopwords.words('english'))

# File paths (placeholders)
glove_file_path = "path/to/glove/file.txt"
ci_file_path = "path/to/ci/job/data.xlsx"
quatt_file_path = "path/to/quatt/position/data.ods"
output_file_path = "path/to/output/results.xlsx"

# Load CI Job Data
df_ci_jobs = pd.read_excel(ci_file_path, sheet_name='SheetName')  # Update with the actual sheet name
ci_job_summaries = df_ci_jobs['CI Job Summary'].fillna('')  # Replace with the actual column name
ci_job_titles = df_ci_jobs['CI Job Title'].fillna('')  # Replace with the actual column name
ci_job_ids = df_ci_jobs['CI Job ID'].fillna('')  # Replace with the actual column name

# Load Quatt Job Data
df_quatt_jds = pd.read_excel(quatt_file_path, engine='odf')  # For .ods files
quatt_jd_titles = df_quatt_jds['Quatt Title'].fillna('')  # Replace with the actual column name
quatt_jd_descriptions = df_quatt_jds['Position Description'].fillna('')  # Replace with the actual column name

# Preprocess CI Job Summaries and Quatt Job Descriptions
ci_job_summaries_clean = ci_job_summaries.apply(lambda x: preprocess_text(clean_text(x), stop_words))
quatt_jd_descriptions_clean = quatt_jd_descriptions.apply(lambda x: preprocess_text(clean_text(x), stop_words))

# Combine the texts for TF-IDF vectorization
combined_texts = pd.concat([ci_job_summaries_clean, quatt_jd_descriptions_clean], axis=0)

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings(glove_file_path)

# Calculate GloVe vectors
ci_vectors = ci_job_summaries_clean.apply(lambda x: get_average_glove_vector(x, glove_embeddings)).tolist()
quatt_vectors = quatt_jd_descriptions_clean.apply(lambda x: get_average_glove_vector(x, glove_embeddings)).tolist()

# Convert lists to numpy arrays
ci_vectors = np.array(ci_vectors)
quatt_vectors = np.array(quatt_vectors)

# Calculate TF-IDF vectors
tfidf_matrix, tfidf_vectorizer = calculate_tfidf_vectors(combined_texts, stop_words)
ci_tfidf_vectors = tfidf_matrix[:len(ci_job_summaries_clean)]
quatt_tfidf_vectors = tfidf_matrix[len(ci_job_summaries_clean):]

# Combine GloVe and TF-IDF vectors
combined_ci_vectors = np.hstack((ci_vectors, ci_tfidf_vectors.toarray()))
combined_quatt_vectors = np.hstack((quatt_vectors, quatt_tfidf_vectors.toarray()))

# Calculate cosine similarity between CI jobs and Quatt job descriptions
similarity_matrix = cosine_similarity(combined_ci_vectors, combined_quatt_vectors)

# Get top N matches (set to 6)
top_n = 6
matches = []
for i, ci_title in enumerate(ci_job_titles):
    sim_scores = list(enumerate(similarity_matrix[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    match_info = [ci_job_ids[i], ci_title, ci_job_summaries[i]]
    for j, score in sim_scores:
        match_info.append(quatt_jd_titles[j])
        match_info.append(quatt_jd_descriptions[j])
        match_info.append(score)
    
    matches.append(match_info)

# Define column headers
columns = ['CI_Job_ID', 'CI_Job_Title', 'CI_JD']
for k in range(1, top_n + 1):
    columns.extend([f'Quatt_Match_{k}_Title', f'JD_{k}', f'Score_{k}'])

# Create DataFrame with matches and export to Excel
df_matches = pd.DataFrame(matches, columns=columns)
df_matches.to_excel(output_file_path, index=False)
