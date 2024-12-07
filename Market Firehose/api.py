from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

# run uvicorn api:app --reload --log-level debug to get API running

app = FastAPI()

# Load environment variables
load_dotenv()

# MongoDB connection
client = AsyncIOMotorClient(os.getenv('MONGODB_URI'))
db = client['news_database']
articles_collection = db['articles']

@app.get("/articles")
async def get_articles(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    publisher: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    sector: Optional[str] = None,
    limit: int = Query(default=10, le=100),  # Limit max results to 100
):
    """
    Get articles with multiple filter options
    """
    try:
        query = {}
        
        # Build query based on provided filters
        if start_date and end_date:
            query["date"] = {"$gte": start_date, "$lte": end_date}
        
        if publisher:
            query["publisher"] = {"$regex": publisher, "$options": "i"}  # Case-insensitive search
            
        if title:
            query["title"] = {"$regex": title, "$options": "i"}
            
        if author:
            query["author"] = {"$regex": author, "$options": "i"}
            
        if sector:
            query["sectors"] = sector  # Exact match for sectors

        cursor = articles_collection.find(query).limit(limit)
        articles = await cursor.to_list(length=limit)
        
        # Convert MongoDB _id to string
        for article in articles:
            article["_id"] = str(article["_id"])
            
        return {
            "total": len(articles),
            "articles": articles
        }
        
    except Exception as e:
        print(f"Error in get_articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/publishers")
async def get_publishers():
    """
    Get list of all publishers in the database
    """
    try:
        publishers = await articles_collection.distinct("publisher")
        return {"publishers": sorted(publishers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors")
async def get_sectors():
    """
    Get list of all sectors in the database
    """
    try:
        sectors = await articles_collection.distinct("sectors")
        sectors = [sector for sector in sectors if sector]  # Remove empty sectors
        return {"sectors": sorted(sectors)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/text")
async def search_text(
    query: str,
    limit: int = Query(default=10, le=100)
):
    """
    Search articles by text in title or body
    """
    try:
        search_query = {
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"body": {"$regex": query, "$options": "i"}}
            ]
        }
        
        cursor = articles_collection.find(search_query).limit(limit)
        articles = await cursor.to_list(length=limit)
        
        # Convert MongoDB _id to string
        for article in articles:
            article["_id"] = str(article["_id"])
            
        return {
            "total": len(articles),
            "articles": articles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get comprehensive statistics about stored articles
    """
    try:
        stats = {
            "total_articles": await articles_collection.count_documents({}),
            "total_publishers": len(await articles_collection.distinct("publisher")),
            "total_sectors": len(await articles_collection.distinct("sectors")),
            "publishers": await articles_collection.distinct("publisher"),
            "sectors": await articles_collection.distinct("sectors")
        }
        
        # Get date range
        earliest = await articles_collection.find_one({}, sort=[("date", 1)])
        latest = await articles_collection.find_one({}, sort=[("date", -1)])
        
        if earliest and latest:
            stats["date_range"] = {
                "earliest": earliest["date"],
                "latest": latest["date"]
            }
            
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)