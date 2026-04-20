#!/usr/bin/env python
# coding: utf-8
"""
Production Traffic Simulation Script for Mealie Recipe Recommendation System

This script simulates realistic user traffic patterns by:
1. Creating/authenticating users
2. Browsing recipes (views)
3. Rating recipes (feedback)
4. Getting personalized recommendations
5. Acting on recommendations (simulating feedback loop)

Usage:
    python simulate_production_traffic.py --mealie-url http://localhost:9000 --duration 3600
"""

import os
import sys
import time
import random
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('traffic_simulation.log')
    ]
)
logger = logging.getLogger(__name__)


class MealieClient:
    """Client for interacting with Mealie API."""
    
    def __init__(self, base_url: str, admin_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.admin_token = admin_token or os.getenv('MEALIE_ADMIN_TOKEN')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
    def _auth_headers(self, token: str) -> Dict:
        return {'Authorization': f'Bearer {token}'}
    
    def health_check(self) -> bool:
        """Check if Mealie service is available."""
        try:
            resp = self.session.get(f"{self.base_url}/api/app/about", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def create_user(self, username: str, email: str, password: str) -> Optional[Dict]:
        """Create a new user account."""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/users/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password,
                    "passwordConfirm": password,
                }
            )
            if resp.status_code in [200, 201]:
                logger.info(f"Created user: {username}")
                return resp.json()
            elif resp.status_code == 409:
                logger.debug(f"User {username} already exists")
                return None
            else:
                logger.warning(f"Failed to create user {username}: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def login(self, username: str, password: str) -> Optional[str]:
        """Login and get access token."""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/auth/token",
                data={
                    "username": username,
                    "password": password,
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            if resp.status_code == 200:
                token = resp.json().get('access_token')
                logger.debug(f"Login successful for {username}")
                return token
            else:
                logger.warning(f"Login failed for {username}: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Login error: {e}")
            return None
    
    def get_recipes(self, token: str, page: int = 1, per_page: int = 50) -> List[Dict]:
        """Get list of recipes."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/recipes",
                headers=self._auth_headers(token),
                params={"page": page, "perPage": per_page}
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get('items', []) if isinstance(data, dict) else data
            return []
        except Exception as e:
            logger.error(f"Error getting recipes: {e}")
            return []
    
    def get_recipe_detail(self, token: str, recipe_slug: str) -> Optional[Dict]:
        """Get detailed recipe information (simulates viewing a recipe)."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/recipes/{recipe_slug}",
                headers=self._auth_headers(token)
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error(f"Error getting recipe detail: {e}")
            return None
    
    def rate_recipe(self, token: str, recipe_slug: str, rating: int) -> bool:
        """Rate a recipe (1-5 stars)."""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/recipes/{recipe_slug}/rating",
                headers=self._auth_headers(token),
                json={"rating": rating}
            )
            if resp.status_code in [200, 201]:
                logger.info(f"Rated recipe {recipe_slug}: {rating} stars")
                return True
            return False
        except Exception as e:
            logger.error(f"Error rating recipe: {e}")
            return False
    
    def get_recommendations(self, token: str, user_id: int, top_k: int = 10) -> List[Dict]:
        """Get personalized recommendations from the ML model."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/recommendations",
                headers=self._auth_headers(token),
                params={"user_id": user_id, "top_k": top_k}
            )
            if resp.status_code == 200:
                return resp.json().get('recommendations', [])
            return []
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def add_to_meal_plan(self, token: str, recipe_slug: str, date: str) -> bool:
        """Add recipe to meal plan (positive feedback signal)."""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/groups/mealplans",
                headers=self._auth_headers(token),
                json={
                    "date": date,
                    "entryType": "dinner",
                    "recipeId": recipe_slug,
                }
            )
            return resp.status_code in [200, 201]
        except Exception as e:
            logger.error(f"Error adding to meal plan: {e}")
            return False


class TrafficSimulator:
    """Simulates realistic user traffic patterns."""
    
    def __init__(self, client: MealieClient, config: Dict):
        self.client = client
        self.config = config
        self.users = []  # List of (username, password, token) tuples
        self.recipes = []
        self.interaction_log = []
        
    def setup_users(self, num_users: int = 10):
        """Create or login test users."""
        logger.info(f"Setting up {num_users} simulated users...")
        
        for i in range(num_users):
            username = f"sim_user_{i:04d}"
            email = f"{username}@simulation.test"
            password = f"SimPass123!_{i}"
            
            # Try to create user
            self.client.create_user(username, email, password)
            
            # Login
            token = self.client.login(username, password)
            if token:
                self.users.append((username, password, token, i))
                
        logger.info(f"Successfully set up {len(self.users)} users")
    
    def load_recipes(self):
        """Load available recipes."""
        if not self.users:
            logger.error("No users available to fetch recipes")
            return
            
        _, _, token, _ = self.users[0]
        self.recipes = self.client.get_recipes(token, per_page=100)
        logger.info(f"Loaded {len(self.recipes)} recipes")
    
    def simulate_user_session(self, user_idx: int):
        """Simulate a single user browsing session."""
        if user_idx >= len(self.users):
            return
            
        username, _, token, user_id = self.users[user_idx]
        logger.debug(f"Simulating session for {username}")
        
        # Simulate browsing behavior
        actions = []
        
        # 1. Browse some recipes (view 3-8 recipes)
        num_views = random.randint(3, 8)
        viewed_recipes = random.sample(self.recipes, min(num_views, len(self.recipes)))
        
        for recipe in viewed_recipes:
            recipe_slug = recipe.get('slug', recipe.get('id'))
            detail = self.client.get_recipe_detail(token, recipe_slug)
            if detail:
                actions.append({
                    'type': 'view',
                    'user_id': user_id,
                    'recipe_id': recipe_slug,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Small delay between views (simulate reading)
            time.sleep(random.uniform(0.5, 2.0))
        
        # 2. Rate some viewed recipes (30-60% of viewed)
        num_ratings = random.randint(
            int(len(viewed_recipes) * 0.3),
            int(len(viewed_recipes) * 0.6) + 1
        )
        recipes_to_rate = random.sample(viewed_recipes, min(num_ratings, len(viewed_recipes)))
        
        for recipe in recipes_to_rate:
            recipe_slug = recipe.get('slug', recipe.get('id'))
            # Biased rating distribution (more 4s and 5s)
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.15, 0.35, 0.35])[0]
            
            if self.client.rate_recipe(token, recipe_slug, rating):
                actions.append({
                    'type': 'rating',
                    'user_id': user_id,
                    'recipe_id': recipe_slug,
                    'rating': rating,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            time.sleep(random.uniform(0.2, 0.5))
        
        # 3. Get recommendations
        recommendations = self.client.get_recommendations(token, user_id)
        if recommendations:
            actions.append({
                'type': 'recommendations_requested',
                'user_id': user_id,
                'num_recommendations': len(recommendations),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # 4. Act on some recommendations (20-40% click-through)
            if recommendations:
                num_clicks = max(1, int(len(recommendations) * random.uniform(0.2, 0.4)))
                clicked = random.sample(recommendations, min(num_clicks, len(recommendations)))
                
                for rec in clicked:
                    rec_slug = rec.get('slug', rec.get('recipe_id'))
                    self.client.get_recipe_detail(token, rec_slug)
                    actions.append({
                        'type': 'recommendation_clicked',
                        'user_id': user_id,
                        'recipe_id': rec_slug,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    # 50% chance to rate the recommended recipe
                    if random.random() < 0.5:
                        # Higher ratings for recommendations (user chose to click)
                        rating = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
                        if self.client.rate_recipe(token, rec_slug, rating):
                            actions.append({
                                'type': 'recommendation_rated',
                                'user_id': user_id,
                                'recipe_id': rec_slug,
                                'rating': rating,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                    
                    time.sleep(random.uniform(0.3, 1.0))
        
        self.interaction_log.extend(actions)
        return actions
    
    def run(self, duration_seconds: int = 3600, interactions_per_minute: int = 10):
        """Run the traffic simulation for specified duration."""
        logger.info(f"Starting traffic simulation for {duration_seconds}s")
        
        # Setup
        if not self.client.health_check():
            logger.error("Mealie service is not available!")
            return
        
        self.setup_users(num_users=self.config.get('num_users', 20))
        self.load_recipes()
        
        if not self.users or not self.recipes:
            logger.error("Setup failed - no users or recipes available")
            return
        
        start_time = time.time()
        session_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            # Pick a random user
            user_idx = random.randint(0, len(self.users) - 1)
            
            try:
                actions = self.simulate_user_session(user_idx)
                session_count += 1
                
                if session_count % 10 == 0:
                    logger.info(f"Completed {session_count} user sessions, "
                               f"{len(self.interaction_log)} total interactions")
                
            except Exception as e:
                logger.error(f"Error in user session: {e}")
            
            # Wait between sessions (controlled rate)
            delay = 60.0 / interactions_per_minute
            time.sleep(delay * random.uniform(0.5, 1.5))
        
        # Save interaction log
        self.save_interaction_log()
        
        logger.info(f"Simulation complete: {session_count} sessions, "
                   f"{len(self.interaction_log)} interactions")
    
    def save_interaction_log(self, filepath: str = "interaction_log.json"):
        """Save interaction log to file."""
        with open(filepath, 'w') as f:
            json.dump(self.interaction_log, f, indent=2)
        logger.info(f"Saved {len(self.interaction_log)} interactions to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Simulate production traffic for Mealie')
    parser.add_argument('--mealie-url', default=os.getenv('MEALIE_URL', 'http://localhost:9000'),
                       help='Mealie service URL')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Simulation duration in seconds')
    parser.add_argument('--rate', type=int, default=10,
                       help='Target interactions per minute')
    parser.add_argument('--num-users', type=int, default=20,
                       help='Number of simulated users')
    parser.add_argument('--admin-token', default=os.getenv('MEALIE_ADMIN_TOKEN'),
                       help='Admin token for user creation')
    
    args = parser.parse_args()
    
    config = {
        'num_users': args.num_users,
        'interactions_per_minute': args.rate,
    }
    
    client = MealieClient(args.mealie_url, args.admin_token)
    simulator = TrafficSimulator(client, config)
    
    try:
        simulator.run(
            duration_seconds=args.duration,
            interactions_per_minute=args.rate
        )
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        simulator.save_interaction_log()


if __name__ == "__main__":
    main()
