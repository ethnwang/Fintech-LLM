import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load environment variables
load_dotenv()

st.title("Stock Research Assistant")
st.subheader("Ask questions about stocks and get AI-powered insights")

# Initialize clients
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

# Connect to your Pinecone index
pinecone_index = pc.Index("stocks")

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

def perform_stock_rag(query):
    """
    Perform RAG for stock-related queries.
    """
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace="stock-descriptions"
    )

    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = """You are an expert financial analyst specializing in stock research and market analysis.
    Provide detailed, well-reasoned answers about stocks based on the provided context.
    Include relevant metrics, market trends, and business analysis in your responses.
    If specific numbers or metrics are mentioned in the context, include them in your response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat welcome message
with st.chat_message("assistant"):
    st.write("Hello! I'm your Stock Research Assistant. You can ask me questions about companies, sectors, market trends, or specific stocks. For example:")
    st.write("- What are the top companies in the semiconductor sector?")
    st.write("- Tell me about companies with high growth in cloud computing")
    st.write("- Which companies have the highest market cap in the technology sector?")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about stocks:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Perform RAG to get initial context
        rag_response = perform_stock_rag(prompt)
        
        # Create messages array for final response
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful financial analyst. Use the provided context to give comprehensive answers about stocks and market trends."
            },
            {"role": "assistant", "content": rag_response},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ]
        
        # Stream the response
        stream = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            stream=True,
        )

        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        # Update with final response
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add sidebar with additional options
with st.sidebar:
    st.header("Search Filters")
    
    # Add sector filter
    sector = st.selectbox(
        "Filter by Sector",
        ["All Sectors", "Technology", "Healthcare", "Finance", "Consumer Goods", "Energy"]
    )
    
    # Add market cap filter
    market_cap = st.select_slider(
        "Market Capitalization",
        options=["All", "Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]
    )
    
    # Add volume filter
    min_volume = st.number_input("Minimum Daily Volume", min_value=0, value=0)
    
    # Add apply filters button
    if st.button("Apply Filters"):
        st.session_state.messages.append({
            "role": "user", 
            "content": f"Show me stocks in the {sector} sector with {market_cap} market cap and minimum daily volume of {min_volume}"
        })