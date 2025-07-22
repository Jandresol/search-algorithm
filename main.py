from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Optional, List, Dict

app = FastAPI()

# Allow CORS so your frontend (likely running on localhost:3000 or similar) can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only, restrict in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserProfile(BaseModel):
    skills: List[str]
    training: List[str]
    interests: List[str]
    saved_opportunities: List[int]

# Request body model
class SearchRequest(BaseModel):
    search_prompt: str
    user_profile: Optional[UserProfile] = None

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Load your opportunities data on startup
with open("final_opps.json", "r") as f:
    opps = json.load(f)

# Load or compute embeddings on startup
embedding_path = "opportunity_vecs.pkl"
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(embedding_path):
    with open(embedding_path, "rb") as f:
        opportunity_vecs = pickle.load(f)
else:
    texts = [opp["text"] for opp in opps]
    opportunity_vecs = model.encode(texts, convert_to_numpy=True)
    with open(embedding_path, "wb") as f:
        pickle.dump(opportunity_vecs, f)

# Example user profile (you can replace or expand as needed)
user = {
    "skills": ["tutoring", "child care", "mentoring"],
    "training": ["child education", "first aid"],
    "interests": ["education", "community outreach"],
    "saved_opportunities": [],
}

def build_user_vector(search_prompt: str, user_profile: UserProfile):
    skills_text = " ".join(user_profile.skills)
    training_text = " ".join(user_profile.training)
    interests_text = " ".join(user_profile.interests)

    skills_vec = normalize(model.encode(skills_text, convert_to_numpy=True)) if skills_text else np.zeros(384)
    training_vec = normalize(model.encode(training_text, convert_to_numpy=True)) if training_text else np.zeros_like(skills_vec)
    interests_vec = normalize(model.encode(interests_text, convert_to_numpy=True)) if interests_text else np.zeros_like(skills_vec)

    user_vec = 0.25 * skills_vec + 0.15 * training_vec + 0.6 * interests_vec
    user_vec = normalize(user_vec)

    saved_texts = [opp["text"] for opp in opps if opp["id"] in user_profile.saved_opportunities]
    saved_combined = " ".join(saved_texts)
    saved_vec = normalize(model.encode(saved_combined, convert_to_numpy=True)) if saved_combined else np.zeros_like(skills_vec)

    # Combine user vector with saved opps vector
    full_user_vec = 0.9 * user_vec + 0.1 * saved_vec
    full_user_vec = normalize(full_user_vec)

    search_prompt = search_prompt.strip().lower()
    search_vec = normalize(model.encode(search_prompt, convert_to_numpy=True)) if search_prompt else None

    if not search_prompt:
        combined_vec = full_user_vec
    else:
        combined_vec = normalize(full_user_vec * 0.1 + search_vec * 0.9)

    return combined_vec

def keyword_match_score(opp, keywords):
    text = (opp["name"] + " " + opp["description"] + " " + " ".join(opp["tags"])).lower()
    return sum(1 for kw in keywords if kw in text)

@app.post("/search")
async def search(request: SearchRequest):
    user_profile = request.user_profile
    if not user_profile:
        # fallback if frontend didn't send a profile
        user_profile = UserProfile(
            skills=["tutoring", "child care", "mentoring"],
            training=["child education", "first aid"],
            interests=["education", "community outreach"],
            saved_opportunities=[],
        )
    combined_vec = build_user_vector(request.search_prompt, user_profile)
    vector_scores = util.pytorch_cos_sim(combined_vec, opportunity_vecs).squeeze().tolist()

    # Extract keywords simply by splitting words excluding stop words
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    keywords = [
        word for word in request.search_prompt.lower().split()
        if word not in ENGLISH_STOP_WORDS
    ]

    keyword_scores = [keyword_match_score(opp, keywords) for opp in opps]
    max_keyword_score = max(keyword_scores) or 1

    # Combine vector + keyword scores (weight keyword score zero for now)
    final_scores = [s + 0 * (k / max_keyword_score) for s, k in zip(vector_scores, keyword_scores)]

    # Rank by final scores
    top_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)

    # Filter out saved opps
    recommended = [
        opps[i] for i in top_indices if opps[i]["id"] not in user["saved_opportunities"]
    ]

    # Prepare output with scores
    recommendations = []
    for i in top_indices[:10]:
        if opps[i]["id"] not in user["saved_opportunities"]:
            recommendations.append({
                "id": opps[i]["id"],
                "name": opps[i]["name"],
                "description": opps[i]["description"],
                "score": final_scores[i],
            })

    return {"recommendations": recommendations}
