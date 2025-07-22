"""
Compute similarities, blend vectors with desired weights and use cosine similarity 
between features extracted from user profile vector vs. each opportunity.
"""
import pandas as pd
import json
import os
import pickle
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import numpy as np

print("[Step 0] Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load and Clean Data ===
file_path = "opportunities.json"
print("[Step 1] Loading opportunity data...")

with open(file_path, 'r') as f:
    data = json.load(f)

fields_to_delete = [
    "location", "imageUrl", "volunteersNeeded", 
    "status", "points", "organizationName", "when"
]
for item in data:
    for field in fields_to_delete:
        item.pop(field, None)

for index, dictionary in enumerate(data):
    dictionary["id"] = index + 1

output_file_path = 'final_opps.json'
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=2)
print(f"[Step 1.5] Cleaned data saved to {output_file_path}")

# === Load Final Cleaned Data ===
print("[Step 2] Loading cleaned opportunities...")
with open(output_file_path, 'r') as f:
    opps = json.load(f)

# === Build Text Fields ===
def build_text(opportunity):
    return (
        " ".join(opportunity["tags"] + opportunity["skills"]) + " " +
        opportunity["description"] + " " + opportunity["name"]
    ).lower()

print("[Step 3] Generating text fields for vectorization...")
for opp in opps:
    opp["text"] = build_text(opp)

# === Input Prompt ===
print("[Step 4] Asking user for search input...")
kw_model = KeyBERT(model)

search_prompt = input("Describe what kind of opportunities you're looking for: ")
# Extract top keywords (can be 1-word or phrases)
keywords = kw_model.extract_keywords(
    search_prompt,
    keyphrase_ngram_range=(1, 2),  # 1 to 2 word phrases
    stop_words='english',         # uses a solid default list
    top_n=5                       # top N phrases to keep
)
search_keywords = [kw for kw, _ in keywords]

print(f"[Step 4.5] Processed search keywords: {search_keywords}")

# === User Profile ===
user = {
    "skills": [
        "tutoring",
        "child care",
        "mentoring",
        "community support",
        "elder care"
    ],
    "training": [
        "child education",
        "first aid",
        "special needs care"
    ],
    "interests": [
        "education",
        "community outreach",
        "volunteering",
        "supporting families"
    ],
    "saved_opportunities": [] 
}




print("[Step 5] Building user profile vector...")

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Prepare texts
skills_text = " ".join(user.get("skills", []))
training_text = " ".join(user.get("training", []))
interests_text = " ".join(user.get("interests", []))

saved_texts = [opp["text"] for opp in opps if opp["id"] in user.get("saved_opportunities", [])]
saved_combined = " ".join(saved_texts)


# Encode each separately (use zeros if empty)
skills_vec = normalize(model.encode(skills_text, convert_to_numpy=True)) if skills_text else np.zeros(384)
training_vec = normalize(model.encode(training_text, convert_to_numpy=True)) if training_text else np.zeros_like(skills_vec)
interests_vec = normalize(model.encode(interests_text, convert_to_numpy=True)) if interests_text else np.zeros_like(skills_vec)
saved_vec = normalize(model.encode(saved_combined, convert_to_numpy=True)) if saved_combined else np.zeros_like(skills_vec)

# Blend user profile vectors with weights
user_vec = 0.25 * skills_vec + 0.15 * training_vec + 0.6 * interests_vec
user_vec = normalize(user_vec)

full_user_vec = 0.9 * user_vec + 0.1 * saved_vec
full_user_vec = normalize(full_user_vec)

# Now encode search prompt (do this after checking if empty if you want)
search_prompt = search_prompt.strip().lower()
search_vec = normalize(model.encode(search_prompt, convert_to_numpy=True)) if search_prompt else None

if search_prompt == "":
    # No search input: boost user profile and saved opps, ignore search prompt
    combined_vec = full_user_vec
    print("[Info] Empty search — using user profile + saved opportunities with weighted vectors.")
else:
    # Encode search vector and use it alone (or blend as you prefer)
    combined_vec = normalize(full_user_vec * 0.1 + search_vec * 0.9)
    print("[Info] Using search prompt for recommendations.")

# === Sentence Transformer ===
print("[Step 6] Loading SentenceTransformer model...")
# Only recompute if the file doesn't exist
embedding_path = "opportunity_vecs.pkl"

if os.path.exists(embedding_path):
    print("[[Step 7] Loading precomputed opportunity vectors...")
    with open(embedding_path, "rb") as f:
        opportunity_vecs = pickle.load(f)
else:
    print("[Step 7] Computing and caching opportunity vectors...")
    texts = [opp["text"] for opp in opps]
    opportunity_vecs = model.encode(texts, convert_to_numpy=True)

    with open(embedding_path, "wb") as f:
        pickle.dump(opportunity_vecs, f)

print("[Step 8] Computing cosine similarity...")
vector_scores = util.pytorch_cos_sim(combined_vec, opportunity_vecs).squeeze().tolist()

# === Rank Results by Vector Similarity ===
print("[Step 9] Ranking opportunities by semantic similarity...")
top_indices = sorted(range(len(vector_scores)), key=lambda i: vector_scores[i], reverse=True)

# === Keyword Matching ===
def keyword_match_score(opp, keywords):
    text = (opp["name"] + " " + opp["description"] + " " + " ".join(opp["tags"])).lower()
    return sum(1 for kw in keywords if kw in text)

print("[Step 10] Scoring keyword relevance...")
keyword_scores = [keyword_match_score(opp, search_keywords) for opp in opps]
max_keyword_score = max(keyword_scores) or 1

# === Combine Vector + Keyword Scores ===
print("[Step 11] Blending vector + keyword scores...")
final_scores = [
    s + 0 * (k / max_keyword_score)
    for s, k in zip(vector_scores, keyword_scores)
]

# === Final Ranking ===
print("[Step 12] Sorting final ranked results...")
top_indices = sorted(
    range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
)
recommended = [
    opps[i] for i in top_indices if opps[i]["id"] not in user["saved_opportunities"]
]

# === Output Top 5 Recommendations ===
def truncate_text(text, max_words=100):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

print("\n[Step 13] Top Recommendations:\n")
for i, r in enumerate(recommended[:5]):
    print(f"Rank {i+1} — ID {r['id']} — {r['name']}")
    print(f"Description: {truncate_text(r['description'], 100)}")
    original_index = top_indices[i]
    print(f"Score: {final_scores[original_index]:.4f}")
    print("-" * 50)
