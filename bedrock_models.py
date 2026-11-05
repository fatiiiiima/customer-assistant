
import requests
import json
import getpass

API_KEY = getpass.getpass("Enter your Bedrock API key: ")

url = "https://bedrock-mantle.us-east-1.api.aws/v1/models"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(url, headers=headers, timeout=30)

    print(f"\nHTTP Status: {response.status_code}")

    response.raise_for_status()

    data = response.json()

    print("\nAvailable models:\n")
    print(json.dumps(data, indent=4))

except requests.exceptions.HTTPError:
    print("\nAPI Error:")
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"\nConnection Error: {e}")
