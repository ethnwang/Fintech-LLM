import requests
from datetime import datetime, timedelta

base_url = 'http://localhost:8000'

try:
    # Get stats
    print("\nGetting articles...")
    response = requests.get(f'{base_url}/articles')
    print("Articles:", response.json())
    
    # Get stats
    print("\nGetting stats...")
    response = requests.get(f'{base_url}/stats')
    print("Stats:", response.json())

    # Get all publishers
    print("\nGetting publishers...")
    response = requests.get(f'{base_url}/publishers')
    print("Publishers:", response.json())
    
    # Get all authors
    print("\nGetting authors...")
    response = requests.get(f'{base_url}/authors')
    print("Authors:", response.json())

    # Search for articles with specific text
    print("\nSearching for technology articles...")
    response = requests.get(f'{base_url}/search/text?query=tech')
    print("Search results:", response.json())

    # Get articles from a specific publisher
    print("\nGetting Reuters articles...")
    response = requests.get(f'{base_url}/articles?publisher=Reuters')
    print("Reuters articles:", response.json())

except requests.exceptions.ConnectionError:
    print("Failed to connect to the server. Make sure uvicorn is running (uvicorn api:app --reload)")
except Exception as e:
    print(f"An error occurred: {e}")