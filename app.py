import streamlit as st
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from scrapetube import Scraper
from qdrant_client import Qdrant
import requests

# Initialize Qdrant client
qdrant_client = Qdrant(host="localhost", port=6333)

# Function to fetch videos and metadata from YouTube channel
def fetch_videos(channel_id):
    scraper = Scraper()
    videos = scraper.channel(channel_id)
    video_data = []
    for video in videos:
        video_id = video.id
        metadata = YouTubeTranscriptApi.get_transcript(video_id)
        video_data.append({
            "title": video.title,
            "description": video.description,
            "video_id": video_id,
            "transcript": metadata["transcript"]
        })
    return video_data

# Function to create knowledge base with Qdrant
def create_knowledge_base(video_data):
    qdrant_client.create_collection("youtube-channel")
    for video in video_data:
        qdrant_client.insert("youtube-channel", [video])

# Function to retrieve answers with RAG pipeline
def retrieve_answers(query):
    qdrant_client.search("youtube-channel", query)
    results = qdrant_client.retrieve_results()
    answers = []
    for result in results:
        video_id = result["video_id"]
        transcript = result["transcript"]
        answer = generate_answer_with_llama(query, transcript)
        answers.append({
            "video_id": video_id,
            "answer": answer
        })
    return answers

# Function to generate answer with LLaMA model
def generate_answer_with_llama(query, transcript):
    api_key = "YOUR_API_KEY"
    url = "https://api.groq.com/v1/llama3-8b-8192/generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"prompt": query, "max_length": 256, "temperature": 0.5}
    response = requests.post(url, headers=headers, json=data)
    answer = response.json()["text"]
    return answer

# Streamlit app
st.title("YouTube RAG Application")
st.header("Search for answers from your favorite YouTube channel")

channel_id = st.text_input("Enter YouTube channel ID")
query = st.text_input("Enter your query")

if st.button("Search"):
    video_data = fetch_videos(channel_id)
    create_knowledge_base(video_data)
    answers = retrieve_answers(query)
    st.header("Results")
    for answer in answers:
        st.write(f"Video ID: {answer['video_id']}")
        st.write(f"Answer: {answer['answer']}")

# Additional user features
st.header("Advanced Features")
st.write("Coming soon!")
