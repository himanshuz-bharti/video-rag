import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from google import genai
# Add parent directory and ingest directory to Python path for seamless imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
ingest_dir = parent_dir / "ingest"
if str(ingest_dir) not in sys.path:
    sys.path.append(str(ingest_dir))
import config
from embedding import embed_text
load_dotenv()
class VideoRAGQuery:
    def __init__(self, provider: str = None, api_key: str = None):
        self.provider = provider or config.PROVIDER
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Initialize ChromaDB persistent client using config path
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=None
        )
        # Initialize Google GenAI client
        if self.api_key:
            self.gemini_client = genai.Client(api_key=self.api_key)
        else:
            self.gemini_client = None
    def search_similar_frames(self, query_text: str, top_k: int = None) -> dict:
        """Query ChromaDB for most similar frame descriptions."""
        if top_k is None:
            top_k = config.TOP_K_RESULTS
            
        # 1. Embed user query using the same embedding function
        query_embedding = embed_text(query_text, provider=self.provider, api_key=self.api_key)
        if not query_embedding:
            print("Error: Failed to generate query embedding.")
            return {}
        # 2. Query collection using query embeddings
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    def answer_query(self, query_text: str) -> tuple[str, list[dict]]:
        """Perform RAG: Retrieve frames from ChromaDB and answer query using Gemini."""
        print(f"Searching for matches for: '{query_text}'...")
        results = self.search_similar_frames(query_text)
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant video frames found in the database to answer your query.", []
        retrieved_docs = results["documents"][0]
        retrieved_metadatas = results["metadatas"][0]
        retrieved_distances = results["distances"][0]
        # Construct context string for the prompt
        context_blocks = []
        sources = []
        
        for idx, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metadatas, retrieved_distances)):
            timestamp_str = meta.get("timestamp_str", "Unknown")
            frame_ct = meta.get("frame_ct", "Unknown")
            frame_path = meta.get("frame_path", "Unknown")
            video_id = meta.get("video_id", "Unknown")
            
            context_blocks.append(
                f"Source Frame {idx+1} [Timestamp: {timestamp_str}, Frame Index: {frame_ct}, Distance: {dist:.4f}]:\n"
                f"{doc}\n"
            )
            sources.append({
                "index": idx + 1,
                "timestamp_str": timestamp_str,
                "frame_ct": frame_ct,
                "frame_path": frame_path,
                "video_id": video_id,
                "distance": dist
            })
        context_str = "\n".join(context_blocks)
        
        # Build RAG prompt for Gemini
        prompt = f"""
You are an expert video analyst assistant. You are given descriptions of specific keyframes extracted from a video, including their timestamps.
Use ONLY the provided context of retrieved keyframe descriptions to answer the user's question in a clear and detailed manner.
When answering:
1. Reference specific timestamps (e.g., "[at 01:23]") of the frames that support your explanation.
2. If the retrieved keyframes do not contain information related to the question, clearly state that the visual information is not present in the indexed context.
Retrieved Context:
---------------------
{context_str}
---------------------
User Question: {query_text}
Answer:
"""
        # Call Gemini LLM using standard SDK
        try:
            if not self.gemini_client:
                raise ValueError("Gemini API Client is not initialized. Please configure a valid API key in settings.")
            response = self.gemini_client.models.generate_content(
                model=config.VISION_MODEL,
                contents=prompt
            )
            return response.text.strip(), sources
        except Exception as e:
            return f"Error calling Gemini LLM: {e}", sources
def main():
    rag = VideoRAGQuery()
    
    # If query is passed as command line argument
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        answer, sources = rag.answer_query(query_text)
        
        print("\n" + "="*40 + " ANSWER " + "="*40)
        print(answer)
        print("="*88)
        
        print("\nSources / Reference Frames:")
        for source in sources:
            print(f"- [{source['timestamp_str']}] Frame {source['frame_ct']} (Distance: {source['distance']:.3f}) | Path: {source['frame_path']}")
    else:
        # Interactive CLI mode
        print("=== Video RAG Query System CLI ===")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                query_text = input("Ask a question about the video: ").strip()
                if not query_text:
                    continue
                if query_text.lower() in ["exit", "quit"]:
                    break
                
                answer, sources = rag.answer_query(query_text)
                
                print("\n" + "="*40 + " ANSWER " + "="*40)
                print(answer)
                print("="*88)
                
                print("\nSources / Reference Frames:")
                for source in sources:
                    print(f"- [{source['timestamp_str']}] Frame {source['frame_ct']} (Distance: {source['distance']:.3f}) | Path: {source['frame_path']}")
                print("\n" + "-"*88 + "\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}\n")
if __name__ == "__main__":
    main()
