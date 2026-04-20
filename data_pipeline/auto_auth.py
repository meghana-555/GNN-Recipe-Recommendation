import os
import requests
import sys

def get_token():
    # If a static token is explicitly provided, we skip dynamic auth.
    if os.getenv("MEALIE_API_TOKEN"):
        return os.getenv("MEALIE_API_TOKEN")

    email = os.getenv("MEALIE_ADMIN_EMAIL", "").strip('"').strip("'")
    password = os.getenv("MEALIE_ADMIN_PASSWORD", "").strip('"').strip("'")
    url = os.getenv("MEALIE_BASE_URL", "http://mealie-frontend:9000")
    
    if not email or not password:
        print("ERROR: Neither MEALIE_API_TOKEN nor MEALIE_ADMIN_EMAIL credentials were provided.", file=sys.stderr)
        sys.exit(1)

    try:
        r = requests.post(f"{url}/api/auth/token", data={"username": email, "password": password})
        if r.status_code == 200:
            print(r.json()["access_token"])
        else:
            print(f"ERROR: Auth failed [{r.status_code}]: {r.text}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot communicate with Mealie: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    get_token()
