# 🎥 Video RAG: Multi-Modal Video Question Answering Dashboard

Video RAG is a premium, real-time Single-Page Dashboard application that allows you to download YouTube videos, extract keyframes, generate brief visual descriptions, index them in a vector database, and query the video content using Gemini or local Ollama LLMs.

The pipeline is designed to support **real-time extraction preview**, **pause and resume control**, and **instant Q&A querying on partial subsets** as soon as they are indexed.

---

## 🚀 Key Features

* **Multi-Threaded Control Loop**: Run ingestion in the background with the ability to **Pause** extraction mid-run, automatically wrap up and index what was extracted so far, and **Resume** later without losing deduplication history.
* **Smart Keyframe Deduplication**: Evaluates structural changes using HSV Color Histograms to keep only distinct visual scenes (max 10 keyframes per session to save token usage and database space).
* **Flexible Multi-Provider Setup**:
  * **Cloud Mode**: Gemini Vision (`gemini-2.5-flash`) for description generation and Gemini Embeddings (`gemini-embedding-2`) for indexing.
  * **Local Mode**: Ollama Vision (`gemma3:4b` or `llama3.2-vision`) for descriptions and local embeddings (`nomic-embed-text`).
* **Real-time Live Gallery**: Watch keyframes render on your dashboard seconds after they are written to disk.
* **Subset Q&A Querying**: Instantly query indexed keyframes at any stage. You can ask questions even while the extraction is paused.
* **Premium Glassmorphic UI**: Beautiful dashboard styling with dark mode, interactive overlays, live status animations, and standard image modal overlays.

---

## 🛠️ Architecture Workflow

The system operates in two core pipelines: **Ingestion** (extracting, describing, and indexing keyframes) and **Retrieval-Augmented Generation (RAG)** (querying the vector index and generating synthesized responses).

```mermaid
graph TD
    subgraph Ingestion Pipeline
        A[YouTube Video Link] -->|yt-dlp| B(Download MP4)
        B -->|OpenCV Frame Reader| C{Deduplication Check}
        C -->|Histogram Similarity < Threshold| D[Extract Keyframe JPG]
        C -->|Similar| E[Skip Frame]
        D -->|on_frame_extracted callback| F[Show Live in UI Gallery]
        D -->|Vision Model: Gemini/Ollama| G(Generate 20-word visual description)
        G -->|Embedding Model: nomic/gemini| H(Generate 768d vector)
        H -->|ChromaDB| I[(Persistent Vector DB)]
    end
    
    subgraph RAG Q&A Pipeline
        J[User Question] -->|Embedding Model| K(Generate Query Vector)
        K -->|Vector Distance Query| I
        I -->|Retrieve top k matching descriptions| L(Construct Context Prompt)
        L -->|Gemini LLM Synthesis| M[Detailed Answer with timestamp citations]
        M -->|FastAPI JSON Response| N[Render in Chat Interface + Clickable Sources]
    end
```

---

## 📁 Repository Structure

```directory
video-rag/
├── config.py             # Global variables (Thresholds, Providers, Model Settings)
├── main_web.py           # FastAPI Web Server, Control Routes, State Management
├── index.html            # Frontend Dashboard (HTML5, Vanilla CSS, JS Event Loops)
├── ingest/
│   ├── download.py       # yt-dlp downloader, OpenCV frame extraction & histogram comparison
│   ├── describe.py       # Vision API connectors (Gemini / Ollama)
│   ├── embedding.py      # Embedding generation (Gemini / Ollama)
│   └── ingest.py         # Main pipeline coordinator
├── query/
│   └── query.py          # Vector query resolver, Gemini Q&A Prompt Constructor
├── pyproject.toml        # Dependency configurations
└── .env                  # API keys and local environment setups
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
* **Python 3.10 - 3.12**
* **Ollama** (if running models locally): [Download Ollama](https://ollama.com/)
  ```bash
  ollama pull gemma3:4b
  ollama pull nomic-embed-text
  ```
* **Gemini API Key** (if running in cloud mode): Get it from Google AI Studio.

### 2. Install Dependencies
This project uses `uv` for lightning-fast environment setup. Alternatively, you can use `pip`.

```bash
# Clone the repository
git clone https://github.com/himanshuz-bharti/video-rag.git
cd video-rag

# Setup virtual environment and install packages
uv venv
uv pip install -r pyproject.toml
```

### 3. Environment Setup
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Configuration Settings
Open [config.py](file:///c:/video-rag/config.py) to toggle options:
* **`PROVIDER`**: Switch between `"ollama"` (local) and `"gemini"` (cloud).
* **`HIST_THRESHOLD_SAVED`**: Enforce global deduplication sensitivity (default: `0.70`).
* **`HIST_THRESHOLD_PREV`**: Consecutive frame difference sensitivity (default: `0.95`).
* **`VIDEO_FORMAT`**: Standard downloader format (default limits downloads to `<= 720p` mp4 to optimize bandwidth).

---

## 🚀 Running the Project

### Starting the Web Dashboard
```bash
uv run python main_web.py
```
Open **`http://127.0.0.1:8000`** in your browser.

1. **Paste a YouTube URL** -> click **Process Video**.
2. Keyframes will immediately render in the **Extracted Frames Preview** gallery as they are processed.
3. Click **Pause** mid-extraction to halt extraction. Status will update to `indexing` as it completes processing for extracted frames.
4. The Q&A section will unlock. Try asking a question (e.g., *"Is there a car in the video?"*). The response will cite timestamps matching the sources.
5. Click **Resume** to continue extraction from the checkpoint.
6. Click **Clear Session** to wipe caches, directory files, and ChromaDB collections.

### Starting the CLI Query Tool
If you prefer running queries from the terminal against the indexed database:
```bash
uv run python query/query.py "explain is there any scene with a car?"
```
Or run the interactive CLI shell:
```bash
uv run python query/query.py
```
