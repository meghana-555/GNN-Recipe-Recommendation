import pandas as pd
import os
import json

# --- Configuration ---
# Direct local path as requested
LOCAL_DATA_PATH = "data/RAW_interactions.csv"
PROJECT_ID = "proj14"

def compute_online_features(user_id):
    """
    Reads the local CSV and extracts real-time features for a specific user.
    """
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"Error: Dataset not found at {LOCAL_DATA_PATH}")
        return None

    print(f"Reading local dataset: {LOCAL_DATA_PATH}")
    
    # Load only necessary columns to save memory if the file is large
    df = pd.read_csv(LOCAL_DATA_PATH)

    # Filter interactions for the target user
    user_interactions = df[df['user_id'] == user_id]

    if user_interactions.empty:
        return {
            "user_id": user_id,
            "status": "user_not_found",
            "features": {}
        }

    # Data Engineering
    # Convert date string to datetime objects
    user_interactions['date'] = pd.to_datetime(user_interactions['date'])
    
    # Sort by date (descending) to get the most recent activity
    user_interactions = user_interactions.sort_values(by='date', ascending=False)
    
    # Feature 1: General Engagement
    avg_rating = user_interactions['rating'].mean()
    total_interactions = len(user_interactions)
    
    # Feature 2: Review Activity (from 'review' column)
    # Count how many non-null reviews this user has written
    review_count = user_interactions['review'].notnull().sum()
    
    # Feature 3: Recent Preference (Top 5 Recipe IDs)
    recent_recipe_ids = user_interactions.head(5)['recipe_id'].tolist()
    
    # Feature 4: Recency (Days since last interaction)
    latest_date = user_interactions.iloc[0]['date']
    days_since_last = (pd.Timestamp.now() - latest_date).days

    return {
        "user_id": user_id,
        "status": "active",
        "metadata": {
            "system": f"mealie_feature_service_{PROJECT_ID}",
            "data_source": "local_filesystem"
        },
        "feature_payload": {
            "interaction_stats": {
                "count": int(total_interactions),
                "average_rating": round(float(avg_rating), 2),
                "reviews_written": int(review_count)
            },
            "behavioral_flags": {
                "is_active_reviewer": bool(review_count > 0),
                "days_since_last_login": int(days_since_last)
            },
            "inference_seeds": {
                "recent_recipe_history": recent_recipe_ids
            }
        }
    }

def run_demo():
    print(f"--- [Mealie Smart Recommendation] Online Feature Path Demo ---")
    
    # Use a known User ID from the Food.com dataset (e.g., 38094)
    target_user = 38094 
    
    print(f"Request: Computing real-time feature vector for User {target_user}...")
    result = compute_online_features(target_user)
    
    if result:
        print("\nJSON PAYLOAD FOR SERVING LAYER:")
        print(json.dumps(result, indent=4))
        print("\n--- Feature Extraction Complete ---")

if __name__ == "__main__":
    run_demo()