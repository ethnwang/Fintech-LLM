import requests
from datetime import datetime
from typing import List, Dict
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import concurrent.futures
import threading
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB setup
try:
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['news_database']
    articles_collection = db['articles']
    # Test connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {e}")
    raise

# Track processed articles
processed_articles = set()
failed_articles = set()
processing_lock = threading.Lock()

def fetch_news_articles(language: str = 'en', limit: int = 100) -> List[Dict]:
    """
    Fetch articles from TheNewsAPI
    """
    url = "https://api.thenewsapi.com/v1/news/all"
    
    params = {
        "api_token": os.getenv("NEWS_API_TOKEN"),
        "language": language,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Fetched {len(data['data'])} articles")
        return data['data']
        
    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
        return []

def process_single_article(article: Dict) -> str:
    """
    Process a single article and store it in MongoDB
    """
    article_id = article['uuid']
    
    # Skip if already processed
    if article_id in processed_articles:
        return f"Already processed article {article_id}"

    try:
        # Combine description and snippet for fuller content
        full_content = " ".join(filter(None, [
            article.get("description", ""),
            article.get("snippet", "")
        ]))
        
        processed_article = {
            "uuid": article_id,
            "publisher": article["source"],
            "title": article["title"],
            "body": full_content,
            "date": article["published_at"],
            "sectors": article.get("categories", []),
            "url": article["url"],
            "processed_at": datetime.utcnow()
        }
        
        # Store in MongoDB
        articles_collection.insert_one(processed_article)
        
        # Track success
        with processing_lock:
            processed_articles.add(article_id)
            with open('successful_articles.txt', 'a') as f:
                f.write(f"{article_id}\n")
        
        return f"Processed article {article_id} successfully"
    
    except Exception as e:
        # Track failure
        with processing_lock:
            failed_articles.add(article_id)
            with open('failed_articles.txt', 'a') as f:
                f.write(f"{article_id}\n")
        
        return f"ERROR processing article {article_id}: {e}"

def parallel_process_articles(articles: List[Dict], max_workers: int = 10) -> None:
    """
    Process multiple articles in parallel using ThreadPoolExecutor
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_article = {
            executor.submit(process_single_article, article): article
            for article in articles
        }

        for future in concurrent.futures.as_completed(future_to_article):
            article = future_to_article[future]
            try:
                result = future.result()
                logger.info(result)

            except Exception as exc:
                logger.error(f'Article {article["uuid"]} generated an exception: {exc}')

def fetch_and_process_articles(batch_size: int = 100, max_workers: int = 10):
    """
    Fetch and process articles in parallel batches
    """
    try:
        # Load previously processed articles
        try:
            with open('successful_articles.txt', 'r') as f:
                processed_articles.update(line.strip() for line in f)
            logger.info(f"Loaded {len(processed_articles)} previously processed articles")
        except FileNotFoundError:
            logger.info("No existing successful articles file")

        # Fetch articles
        articles = fetch_news_articles(limit=batch_size)
        if articles:
            logger.info(f"\nProcessing batch of {len(articles)} articles...")
            parallel_process_articles(articles, max_workers=max_workers)
            logger.info("Batch processing complete")
        else:
            logger.info("No articles to process")

    except Exception as e:
        logger.error(f"Error in fetch and process: {e}")

def start_continuous_processing(batch_size: int = 100, max_workers: int = 10, interval_seconds: int = 60):
    """
    Continuously fetch and process articles
    """
    logger.info(f"Starting continuous processing with {max_workers} workers...")
    logger.info(f"Processing up to {batch_size} articles every {interval_seconds} seconds")
    
    try:
        while True:
            fetch_and_process_articles(batch_size, max_workers)
            
            # Write current state to file after each batch
            logger.info("Writing processed articles to file...")
            all_articles = list(articles_collection.find({}))
            write_articles_to_file(all_articles)
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        # Write final state to file before shutting down
        logger.info("Writing final state to file before shutdown...")
        all_articles = list(articles_collection.find({}))
        write_articles_to_file(all_articles)
        raise
        
def write_articles_to_file(articles: List[Dict], filename: str = "processed_articles.txt"):
    """
    Write new articles to a text file, skipping existing ones
    """
    # Read existing UUIDs from file
    existing_uuids = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("UUID: "):
                    uuid = line.strip().replace("UUID: ", "")
                    existing_uuids.add(uuid)
    except FileNotFoundError:
        # File doesn't exist yet
        pass

    # Open file in append mode instead of write mode
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            # Write header only if file is new (empty)
            if not existing_uuids:
                f.write(f"Articles processed at {datetime.utcnow()}\n")
                f.write("=" * 100 + "\n\n")
            
            # Write only new articles
            new_articles = 0
            for article in articles:
                if article['uuid'] not in existing_uuids:
                    f.write(f"UUID: {article['uuid']}\n")
                    f.write(f"Title: {article['title']}\n")
                    f.write(f"Publisher: {article['publisher']}\n")
                    f.write(f"Author: {article['author']}\n")
                    f.write(f"Date: {article['date']}\n")
                    f.write(f"URL: {article['url']}\n")
                    f.write(f"Sectors: {', '.join(article['sectors']) if article['sectors'] else 'None'}\n")
                    f.write(f"Language: {article['language']}\n")
                    f.write("\nBody:\n{}\n".format(article['body']))
                    f.write("\n" + "=" * 100 + "\n\n")
                    new_articles += 1

            logger.info(f"Added {new_articles} new articles to file")
            
    except Exception as e:
        logger.error(f"Error writing articles to file: {e}")
        

if __name__ == "__main__":
    try:
        logger.info("\nStarting article processing...")
        start_continuous_processing(batch_size=100, max_workers=10, interval_seconds=60)
        
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        
    finally:
        # Try to write articles one last time, even if there was an error
        try:
            logger.info("Performing final write to file...")
            all_articles = list(articles_collection.find({}))
            write_articles_to_file(all_articles)
        except Exception as e:
            logger.error(f"Error writing final file: {e}")