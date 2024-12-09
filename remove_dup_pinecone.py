from pinecone import Pinecone
from typing import Dict, List, Optional
from tqdm import tqdm
import time
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(os.getenv("PINECONE_API_KEY"))

def remove_ticker_duplicates(
    index_name: str,
    api_key: str,
    environment: str,
    namespace: str,
    keep_strategy: str = 'first',
    page_size: int = 1000,
    deletion_batch_size: int = 100
) -> Dict:
    """
    Remove duplicate vectors from Pinecone based on Ticker metadata field.
    
    Args:
        index_name: Name of the Pinecone index
        api_key: Pinecone API key
        environment: Pinecone environment
        namespace: Namespace to deduplicate
        keep_strategy: Strategy for which duplicate to keep ('first' or 'last')
        page_size: Number of vectors to fetch per page
        deletion_batch_size: Number of vectors to delete in a single API call
    
    Returns:
        Dict containing statistics about the deduplication process
    """
    # Initialize Pinecone
    index = pc.Index(index_name)
    
    # Statistics
    stats = {
        "total_vectors": 0,
        "duplicates_removed": 0,
        "processing_time": 0
    }
    
    start_time = time.time()
    
    # Get total vector count
    index_stats = index.describe_index_stats()
    total_vectors = index_stats.namespaces.get(namespace, {}).get('vector_count', 0)
    stats["total_vectors"] = total_vectors
    
    if total_vectors == 0:
        print(f"No vectors found in namespace '{namespace}'")
        return stats

    print(f"Processing {total_vectors} vectors...")
    
    # Track tickers and their associated IDs
    ticker_map = {}
    duplicates_to_remove = set()
    
    # Get dimension of vectors
    dimension = index.describe_index_stats().dimension
    
    # Process vectors in pages
    with tqdm(total=total_vectors) as pbar:
        next_page_token = None
        while True:
            # Fetch a page of vectors
            query_response = index.query(
                vector=[0] * dimension,  # dummy vector for pagination
                namespace=namespace,
                top_k=page_size,
                include_metadata=True,
                include_values=False,  # We don't need vector values
                page_size=page_size,
                page_token=next_page_token
            )
            
            if not query_response.matches:
                break
            
            # Process each vector
            for match in query_response.matches:
                if not match.metadata or 'Ticker' not in match.metadata:
                    continue
                    
                ticker = match.metadata['Ticker']
                
                if ticker in ticker_map:
                    # If we already have this ticker, mark one for deletion
                    if keep_strategy == 'first':
                        duplicates_to_remove.add(match.id)
                    else:  # keep_strategy == 'last'
                        duplicates_to_remove.add(ticker_map[ticker])
                        ticker_map[ticker] = match.id
                else:
                    ticker_map[ticker] = match.id
            
            # Delete duplicates in batches
            duplicate_list = list(duplicates_to_remove)
            for i in range(0, len(duplicate_list), deletion_batch_size):
                batch = duplicate_list[i:i + deletion_batch_size]
                index.delete(ids=batch, namespace=namespace)
                stats["duplicates_removed"] += len(batch)
            
            # Clear the set for the next batch
            duplicates_to_remove.clear()
            
            # Update progress bar
            pbar.update(len(query_response.matches))
            
            # Get next page token
            next_page_token = query_response.page_token
            if not next_page_token:
                break
    
    stats["processing_time"] = time.time() - start_time
    
    print(f"\nDeduplication complete:")
    print(f"Total vectors processed: {stats['total_vectors']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    
    return stats

# Example usage:
if __name__ == "__main__":
    INDEX_NAME = "stocks"
    API_KEY = pc
    ENVIRONMENT = "us-east1-aws"
    NAMESPACE = "stock-descriptions"
    
    stats = remove_ticker_duplicates(
        index_name=INDEX_NAME,
        api_key=API_KEY,
        environment=ENVIRONMENT,
        namespace=NAMESPACE,
        keep_strategy='first'  # Keep the first occurrence of each ticker
    )