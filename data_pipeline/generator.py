"""
Realistic Continuous Traffic Generator for Mealie GNN Recommender.

Simulates multi-dimensional user behavior through Mealie's REST API:
  - Explicit signals: star ratings (1-5)
  - Implicit signals: favorites, meal plan additions, recipe views
  - Temporal patterns: peak hours, weekend spikes
  - User personas: health-conscious, comfort-food lover, adventurous, casual

All interactions land in PostgreSQL and are captured by
ingest_mealie_traffic.py → batch_pipeline.py → S3 → Training loop.

Usage:
    python generator.py                          # run forever, default pace
    python generator.py --duration 1440          # run for 24 hours
    python generator.py --interval 30            # one action every 30 seconds
    python generator.py --users 5                # simulate 5 concurrent personas
"""

import time
import random
import os
import sys
import argparse
import json
import requests
import math
from datetime import datetime

# --- Configuration ---
MEALIE_BASE_URL = os.getenv("MEALIE_BASE_URL", "http://mealie-frontend:9000")
MEALIE_ADMIN_EMAIL = os.getenv("MEALIE_ADMIN_EMAIL", "changeme@example.com")
MEALIE_ADMIN_PASSWORD = os.getenv("MEALIE_ADMIN_PASSWORD", "MyPassword")

# --- User Persona Definitions ---
# Each persona has different rating tendencies and activity patterns
PERSONAS = [
    {
        "name": "health_enthusiast",
        "email_suffix": "health",
        "rating_weights": [2, 5, 15, 40, 38],     # loves healthy recipes (skews 4-5)
        "activity_multiplier": 1.2,                 # more active than average
        "favorite_probability": 0.35,                # often favorites recipes
        "mealplan_probability": 0.25,                # frequently meal plans
        "preferred_keywords": ["salad", "vegetable", "chicken", "fish", "grain"],
    },
    {
        "name": "comfort_food_lover",
        "email_suffix": "comfort",
        "rating_weights": [3, 8, 20, 35, 34],      # mostly positive
        "activity_multiplier": 0.8,                  # slightly less active
        "favorite_probability": 0.20,
        "mealplan_probability": 0.15,
        "preferred_keywords": ["cheese", "butter", "cream", "chocolate", "cake"],
    },
    {
        "name": "adventurous_cook",
        "email_suffix": "adventure",
        "rating_weights": [8, 12, 25, 30, 25],     # wider spread (rates honestly)
        "activity_multiplier": 1.5,                  # very active, tries everything
        "favorite_probability": 0.15,
        "mealplan_probability": 0.10,
        "preferred_keywords": [],                    # no preference, tries randomly
    },
    {
        "name": "casual_browser",
        "email_suffix": "casual",
        "rating_weights": [5, 10, 30, 35, 20],     # rates around 3-4
        "activity_multiplier": 0.5,                  # rarely active
        "favorite_probability": 0.10,
        "mealplan_probability": 0.05,
        "preferred_keywords": ["easy", "quick", "simple"],
    },
    {
        "name": "weekend_warrior",
        "email_suffix": "weekend",
        "rating_weights": [3, 7, 20, 40, 30],      # mostly positive
        "activity_multiplier": 0.3,                  # low base, but spikes on weekends
        "favorite_probability": 0.25,
        "mealplan_probability": 0.40,                # always plans meals
        "preferred_keywords": ["roast", "steak", "pasta", "soup"],
    },
]


def authenticate(email=None, password=None):
    """Get a valid API token from Mealie."""
    email = email or MEALIE_ADMIN_EMAIL
    password = password or MEALIE_ADMIN_PASSWORD
    url = f"{MEALIE_BASE_URL}/api/auth/token"
    data = {"username": email, "password": password}
    try:
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        return token
    except Exception as e:
        return None


def get_invite_token(admin_token):
    """Create a household invitation token via admin API (reuse existing if available)."""
    headers = {"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"}
    try:
        # Check for existing invite tokens
        resp = requests.get(f"{MEALIE_BASE_URL}/api/households/invitations", headers=headers, timeout=10)
        resp.raise_for_status()
        tokens = resp.json()
        for t in tokens:
            if t.get("usesLeft", 0) > 0:
                return t["token"]
        
        # Create a new one with 50 uses
        resp = requests.post(f"{MEALIE_BASE_URL}/api/households/invitations",
                           headers=headers, json={"uses": 50}, timeout=10)
        resp.raise_for_status()
        return resp.json()["token"]
    except Exception as e:
        print(f"    Failed to get invite token: {e}")
        return None


def register_user(invite_token, email, password, full_name):
    """Register a new user via Mealie public API with invite token."""
    payload = {
        "email": email,
        "password": password,
        "passwordConfirm": password,
        "fullName": full_name,
        "username": email.split("@")[0],
        "group_token": invite_token,
    }
    try:
        resp = requests.post(f"{MEALIE_BASE_URL}/api/users/register",
                           json=payload, timeout=10)
        if resp.status_code in [200, 201]:
            print(f"    ✅ Registered {email}")
            return True
        elif "already" in resp.text.lower():
            return True
        else:
            print(f"    Registration: {resp.status_code} - {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"    Registration error: {e}")
        return False


def setup_personas(num_users):
    """Create simulated users and return their auth tokens."""
    print(f"👥 Setting up {num_users} simulated user personas...")
    active_users = []
    
    # Get admin token and create invite token for user registration
    admin_token = authenticate()
    if not admin_token:
        print("❌ Cannot authenticate admin for user creation.")
        return active_users
    
    invite_token = get_invite_token(admin_token)
    if not invite_token:
        print("❌ Cannot get invite token. All personas will use admin account.")

    for i in range(num_users):
        persona = PERSONAS[i % len(PERSONAS)]
        email = f"sim_{persona['email_suffix']}_{i}@mealie.io"
        password = "SimPassword123!"
        full_name = f"Sim {persona['name'].replace('_', ' ').title()} #{i}"

        if invite_token:
            registered = register_user(invite_token, email, password, full_name)
        else:
            registered = False
        token = authenticate(email, password)

        if token:
            # Get user's own ID for rating/favorite APIs
            user_id = None
            try:
                resp = requests.get(f"{MEALIE_BASE_URL}/api/users/self",
                                   headers={"Authorization": f"Bearer {token}"}, timeout=10)
                user_id = resp.json().get("id")
            except: pass
            
            active_users.append({
                "persona": persona,
                "email": email,
                "token": token,
                "user_id": user_id,
                "total_actions": 0,
                "rated_slugs": set(),
                "favorited_slugs": set(),
            })
            print(f"  ✅ {full_name} ({persona['name']}) — ready [uid: {user_id[:8] if user_id else '?'}...]")
        else:
            # Fallback: use admin token with this persona's behavior profile
            admin_token = authenticate()
            admin_id = None
            if admin_token:
                try:
                    resp = requests.get(f"{MEALIE_BASE_URL}/api/users/self",
                                       headers={"Authorization": f"Bearer {admin_token}"}, timeout=10)
                    admin_id = resp.json().get("id")
                except: pass
                active_users.append({
                    "persona": persona,
                    "email": MEALIE_ADMIN_EMAIL,
                    "token": admin_token,
                    "user_id": admin_id,
                    "total_actions": 0,
                    "rated_slugs": set(),
                    "favorited_slugs": set(),
                })
                print(f"  ⚠️  {full_name} — using admin fallback")

    if not active_users:
        print("❌ No users could be set up. Exiting.")
        sys.exit(1)

    print(f"✅ {len(active_users)} personas active.\n")
    return active_users


def fetch_all_recipes(token):
    """Fetch all recipe slugs and names from Mealie API."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{MEALIE_BASE_URL}/api/recipes"
    params = {"page": 1, "perPage": 500}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        recipes = [{"slug": r["slug"], "name": r.get("name", "")} for r in items if r.get("slug")]
        return recipes
    except Exception as e:
        print(f"❌ Failed to fetch recipes: {e}")
        return []


def pick_recipe_for_persona(recipes, persona, rated_slugs):
    """Select a recipe based on persona preferences with some randomness."""
    keywords = persona.get("preferred_keywords", [])
    
    # 60% chance to pick a recipe matching persona preference
    if keywords and random.random() < 0.6:
        matching = [r for r in recipes
                    if any(kw in r["name"].lower() for kw in keywords)
                    and r["slug"] not in rated_slugs]
        if matching:
            return random.choice(matching)

    # Otherwise pick randomly (excluding already rated)
    available = [r for r in recipes if r["slug"] not in rated_slugs]
    if not available:
        available = recipes  # Allow re-rating if pool exhausted
    return random.choice(available)


def action_rate(token, user_id, slug, persona):
    """Rate a recipe via Mealie per-user rating API."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    rating = random.choices([1, 2, 3, 4, 5], weights=persona["rating_weights"])[0]
    try:
        resp = requests.post(
            f"{MEALIE_BASE_URL}/api/users/{user_id}/ratings/{slug}",
            headers=headers, json={"rating": float(rating)}, timeout=10
        )
        resp.raise_for_status()
        return "rate", rating
    except Exception:
        return None, None


def action_favorite(token, user_id, slug):
    """Mark a recipe as favorite via Mealie per-user favorite API."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(
            f"{MEALIE_BASE_URL}/api/users/{user_id}/favorites/{slug}",
            headers=headers, timeout=10
        )
        resp.raise_for_status()
        return "favorite", None
    except Exception:
        return None, None


def action_add_to_mealplan(token, slug, recipe_id=None):
    """Add a recipe to the user's meal plan."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    today = datetime.now().strftime("%Y-%m-%d")
    meal_types = ["breakfast", "lunch", "dinner", "snack"]
    
    # First get the recipe's actual UUID (Mealie needs UUID, not slug)
    if not recipe_id:
        try:
            resp = requests.get(f"{MEALIE_BASE_URL}/api/recipes/{slug}",
                                headers=headers, timeout=10)
            resp.raise_for_status()
            recipe_id = resp.json().get("id")
        except Exception:
            return None, None
    
    payload = {
        "date": today,
        "entryType": random.choice(meal_types),
        "title": "",
        "text": "",
        "recipeId": recipe_id,
    }
    url = f"{MEALIE_BASE_URL}/api/households/mealplans"
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        return "mealplan", None
    except Exception:
        return None, None


def get_temporal_multiplier():
    """
    Simulate realistic temporal patterns:
    - Higher activity 11am-1pm and 5pm-9pm (mealtimes)
    - Lower activity 1am-6am (sleeping)
    - Weekend boost
    """
    hour = datetime.now().hour
    day = datetime.now().weekday()  # 0=Monday, 6=Sunday

    # Base hourly pattern (sinusoidal-like)
    if 1 <= hour <= 6:
        hour_mult = 0.2   # sleeping
    elif 11 <= hour <= 13:
        hour_mult = 1.5    # lunch browsing
    elif 17 <= hour <= 21:
        hour_mult = 1.8    # dinner time peak
    elif 7 <= hour <= 10:
        hour_mult = 0.8    # morning
    else:
        hour_mult = 0.6    # late night

    # Weekend boost
    weekend_mult = 1.4 if day >= 5 else 1.0

    return hour_mult * weekend_mult


def run_generator(duration_minutes=-1, interval_seconds=60, num_users=3):
    """
    Main generator loop simulating realistic multi-user traffic.
    """
    # Setup
    admin_token = authenticate()
    if not admin_token:
        print("❌ Cannot authenticate with Mealie. Exiting.")
        sys.exit(1)

    recipes = fetch_all_recipes(admin_token)
    if not recipes:
        print("❌ No recipes found. Run seed_mealie_recipes.py first.")
        sys.exit(1)

    users = setup_personas(num_users)

    if duration_minutes <= 0:
        end_time = float('inf')
        duration_str = "INDEFINITELY"
    else:
        end_time = time.time() + (duration_minutes * 60)
        duration_str = f"{duration_minutes} minutes"

    print("=" * 65)
    print(f"📊 Realistic Traffic Generator Active")
    print(f"   Duration: {duration_str}")
    print(f"   Base interval: {interval_seconds}s | Users: {len(users)}")
    print(f"   Recipe pool: {len(recipes)} recipes")
    print(f"   Signals: rating + favorite + mealplan")
    print("=" * 65)

    total_actions = 0

    try:
        while time.time() < end_time:
            # Pick a random active user
            user = random.choice(users)
            persona = user["persona"]
            token = user["token"]
            timestamp = datetime.now().strftime('%H:%M:%S')

            # Temporal activity gating
            temporal_mult = get_temporal_multiplier()
            persona_mult = persona["activity_multiplier"]

            # Skip this tick based on combined probability
            if random.random() > (temporal_mult * persona_mult * 0.5):
                time.sleep(interval_seconds)
                continue

            # Pick a recipe suited to this persona
            recipe = pick_recipe_for_persona(recipes, persona, user["rated_slugs"])
            slug = recipe["slug"]
            name_short = recipe["name"][:35]

            # Decide what action to take (weighted by persona)
            roll = random.random()
            fav_prob = persona["favorite_probability"]
            meal_prob = persona["mealplan_probability"]

            if roll < 0.60:
                # 60% chance: Rate the recipe (strongest signal)
                action_type, value = action_rate(token, user["user_id"], slug, persona)
                if action_type:
                    user["rated_slugs"].add(slug)
                    label = f"⭐ {value}/5"
            elif roll < 0.60 + fav_prob:
                # Favorite the recipe (implicit 4.5 rating)
                action_type, _ = action_favorite(token, user["user_id"], slug)
                if action_type:
                    user["favorited_slugs"].add(slug)
                    label = "❤️ favorited (→4.5)"
            else:
                # Add to meal plan (implicit 4.0 rating)
                action_type, _ = action_add_to_mealplan(token, slug)
                label = "🍽️ meal-planned (→4.0)"

            if action_type:
                total_actions += 1
                user["total_actions"] += 1
                print(f"  [{timestamp}] {persona['name']:20s} | {label:18s} | {name_short}  (#{total_actions})")

            # Dynamic interval: faster during peak hours
            adjusted_interval = interval_seconds / max(temporal_mult, 0.3)
            time.sleep(adjusted_interval)

            # Refresh tokens every 200 actions
            if total_actions % 200 == 0 and total_actions > 0:
                print("  🔄 Refreshing all user tokens...")
                for u in users:
                    new_token = authenticate(u["email"], "SimPassword123!")
                    if new_token:
                        u["token"] = new_token

    except KeyboardInterrupt:
        print("\n⏹️ Generator stopped by user.")

    # Summary
    print(f"\n{'=' * 65}")
    print(f"📈 Traffic Generation Summary")
    print(f"   Total actions: {total_actions}")
    for u in users:
        p = u["persona"]
        print(f"   {p['name']:20s}: {u['total_actions']} actions, "
              f"{len(u['rated_slugs'])} rated, {len(u['favorited_slugs'])} favorited")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mealie Realistic Traffic Generator")
    parser.add_argument("--duration", type=int, default=-1,
                        help="Duration in minutes (-1 = forever)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Base seconds between actions (default: 60)")
    parser.add_argument("--users", type=int, default=3,
                        help="Number of simulated user personas (default: 3)")
    args = parser.parse_args()

    run_generator(
        duration_minutes=args.duration,
        interval_seconds=args.interval,
        num_users=args.users,
    )
