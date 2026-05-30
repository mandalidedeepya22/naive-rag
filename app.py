"""Main application server for Naive RAG with Gemini and Pinecone."""

import os
import sys
import json
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# HTML, CSS, and JS template for the clean, premium Antigravity web interface
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Naive RAG Engine</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0b10;
            --bg-secondary: #12131a;
            --bg-sidebar: #0f1016;
            --accent-color: #5c62d6;
            --accent-gradient: linear-gradient(135deg, #6366f1, #a855f7);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --success-glow: 0 0 10px rgba(16, 185, 129, 0.4);
            --error-glow: 0 0 10px rgba(239, 68, 68, 0.4);
            --border-color: #262936;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar Styling */
        .sidebar {
            width: 320px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            height: 100%;
            overflow-y: auto;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            font-size: 1.75rem;
        }

        .logo-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-section {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .section-title {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .api-badge {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-dot.active {
            background-color: #10b981;
            box-shadow: var(--success-glow);
        }

        .status-dot.inactive {
            background-color: #ef4444;
            box-shadow: var(--error-glow);
        }

        .instruction-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .instruction-item {
            display: flex;
            gap: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        .instruction-num {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            flex-shrink: 0;
        }

        /* Main Content Styling */
        .main-content {
            flex: 1;
            padding: 3rem;
            height: 100%;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 2.5rem;
            max-width: 1000px;
            margin: 0 auto;
            width: 100%;
        }

        .hero-section {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .hero-subtitle {
            font-size: 1.125rem;
            color: var(--text-secondary);
        }

        .search-container {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .input-wrapper {
            display: flex;
            gap: 1rem;
        }

        .question-input {
            flex: 1;
            background-color: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            font-size: 1rem;
            color: var(--text-primary);
            outline: none;
            transition: all 0.2s ease;
        }

        .question-input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        .ask-button {
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            padding: 0 2rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .ask-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
        }

        .ask-button:active {
            transform: translateY(0);
        }

        /* Loading Indicator */
        .loading-indicator {
            display: none;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
        }

        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(99, 102, 241, 0.1);
            border-top: 3px solid #6366f1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 0.95rem;
            color: var(--text-secondary);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        /* Results Display */
        .results-section {
            display: none;
            flex-direction: column;
            gap: 2rem;
        }

        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .answer-card {
            border-left: 4px solid #a855f7;
        }

        .answer-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #f3f4f6;
            white-space: pre-line;
        }

        .chunks-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .chunk-card {
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.01);
            overflow: hidden;
        }

        .chunk-header {
            padding: 0.75rem 1.25rem;
            background-color: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .chunk-meta {
            display: flex;
            gap: 1rem;
        }

        .badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            background-color: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .chunk-body {
            padding: 1.25rem;
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text-secondary);
        }

        /* Alert Boxes */
        .alert {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            font-size: 0.95rem;
            display: none;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid transparent;
        }

        .alert-error {
            background-color: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }

        .alert-warning {
            background-color: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.2);
            color: #fcd34d;
        }
    </style>
</head>
<body>
    <!-- Sidebar Section -->
    <aside class="sidebar">
        <div class="logo-section">
            <span class="logo-icon">🔍</span>
            <h1 class="logo-title">Naive RAG</h1>
        </div>
        <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5;">
            A beginner-friendly Retrieval-Augmented Generation application powered by Gemini and Pinecone.
        </p>
        
        <hr style="border: 0; border-top: 1px solid var(--border-color);">

        <div class="sidebar-section">
            <h2 class="section-title">API Status</h2>
            <div class="api-badge">
                <span>Gemini API</span>
                <span id="gemini-status" class="status-dot inactive"></span>
            </div>
            <div class="api-badge">
                <span>Pinecone SDK</span>
                <span id="pinecone-status" class="status-dot inactive"></span>
            </div>
        </div>

        <div class="sidebar-section">
            <h2 class="section-title">System Status</h2>
            <div class="api-badge" style="flex-direction: column; align-items: flex-start; gap: 0.25rem;">
                <span style="font-size: 0.75rem; color: var(--text-muted);">Active Index</span>
                <span id="active-index" style="font-weight: 500;">checking...</span>
            </div>
        </div>

        <hr style="border: 0; border-top: 1px solid var(--border-color);">

        <div class="sidebar-section">
            <h2 class="section-title">Ingestion Instructions</h2>
            <div class="instruction-list">
                <div class="instruction-item">
                    <span class="instruction-num">1</span>
                    <span>Add your Gemini and Pinecone API keys to the <code>.env</code> file.</span>
                </div>
                <div class="instruction-item">
                    <span class="instruction-num">2</span>
                    <span>Place your custom text documents in the <code>data/</code> folder.</span>
                </div>
                <div class="instruction-item">
                    <span class="instruction-num">3</span>
                    <span>Run the ingestion pipeline in your terminal: <code>python ingest.py</code></span>
                </div>
                <div class="instruction-item">
                    <span class="instruction-num">4</span>
                    <span>Type a question in the input area and generate grounded answers instantly!</span>
                </div>
            </div>
        </div>
    </aside>

    <!-- Main Content Section -->
    <main class="main-content">
        <header class="hero-section">
            <h1 class="hero-title">RAG Query Hub</h1>
            <p class="hero-subtitle">Ask questions and retrieve semantic context from your knowledge base</p>
        </header>

        <!-- Search input box -->
        <section class="search-container">
            <div class="input-wrapper">
                <input type="text" id="question-input" class="question-input" placeholder="Ask a question about the ingested documents..." autocomplete="off">
                <button id="ask-btn" class="ask-button">Ask AI</button>
            </div>
            <div id="error-alert" class="alert alert-error"></div>
            <div id="warn-alert" class="alert alert-warning"></div>
        </section>

        <!-- Loading spinner -->
        <div id="loading" class="loading-indicator">
            <div class="spinner"></div>
            <span class="loading-text">Retrieving relevant chunks and generating answer...</span>
        </div>

        <!-- RAG output displays -->
        <section id="results" class="results-section">
            <div class="card">
                <h3 class="card-title">User Question</h3>
                <p id="result-question" style="font-size: 1.15rem; font-weight: 500;"></p>
            </div>

            <div class="card answer-card">
                <h3 class="card-title">Generated Answer</h3>
                <p id="result-answer" class="answer-text"></p>
            </div>

            <div class="card" style="background-color: transparent; border: none; box-shadow: none; padding: 0;">
                <h3 class="card-title" style="margin-bottom: 0.75rem;">Retrieved Context Chunks (Top 3)</h3>
                <div id="result-chunks" class="chunks-list"></div>
            </div>
        </section>
    </main>

    <script>
        const questionInput = document.getElementById('question-input');
        const askBtn = document.getElementById('ask-btn');
        const errorAlert = document.getElementById('error-alert');
        const warnAlert = document.getElementById('warn-alert');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        // Elements to update on success
        const resultQuestion = document.getElementById('result-question');
        const resultAnswer = document.getElementById('result-answer');
        const resultChunks = document.getElementById('result-chunks');

        // Sidebar elements
        const geminiStatus = document.getElementById('gemini-status');
        const pineconeStatus = document.getElementById('pinecone-status');
        const activeIndex = document.getElementById('active-index');

        // Fetch API connection status on page load
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.gemini_connected) {
                    geminiStatus.className = 'status-dot active';
                } else {
                    geminiStatus.className = 'status-dot inactive';
                }

                if (data.pinecone_connected) {
                    pineconeStatus.className = 'status-dot active';
                } else {
                    pineconeStatus.className = 'status-dot inactive';
                }

                activeIndex.textContent = data.index_name;
            } catch (err) {
                console.error("Error checking connection status", err);
            }
        }

        // Trigger on click or Enter key
        askBtn.addEventListener('click', performQuery);
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                performQuery();
            }
        });

        async function performQuery() {
            const question = questionInput.value.trim();

            // Clear previous alerts/results
            errorAlert.style.display = 'none';
            warnAlert.style.display = 'none';
            results.style.display = 'none';

            if (!question) {
                warnAlert.textContent = "Please enter a question.";
                warnAlert.style.display = 'flex';
                return;
            }

            // Show loading
            loading.style.display = 'flex';
            askBtn.disabled = true;

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "An unexpected error occurred.");
                }

                // Render success result
                resultQuestion.textContent = data.question;
                resultAnswer.textContent = data.answer;

                // Render chunks
                resultChunks.innerHTML = '';
                data.chunks.forEach((chunk, index) => {
                    const card = document.createElement('div');
                    card.className = 'chunk-card';

                    const header = document.createElement('div');
                    header.className = 'chunk-header';
                    
                    const title = document.createElement('span');
                    title.textContent = `Chunk ${index + 1}`;

                    const meta = document.createElement('div');
                    meta.className = 'chunk-meta';

                    const sourceBadge = document.createElement('span');
                    sourceBadge.className = 'badge';
                    sourceBadge.textContent = `Source: ${chunk.source}`;

                    const scoreBadge = document.createElement('span');
                    scoreBadge.className = 'badge';
                    scoreBadge.textContent = `Score: ${chunk.score.toFixed(4)}`;

                    meta.appendChild(sourceBadge);
                    meta.appendChild(scoreBadge);
                    header.appendChild(title);
                    header.appendChild(meta);

                    const body = document.createElement('div');
                    body.className = 'chunk-body';
                    body.textContent = chunk.text;

                    card.appendChild(header);
                    card.appendChild(body);
                    resultChunks.appendChild(card);
                });

                results.style.display = 'flex';
            } catch (err) {
                // Show errors dynamically
                const msg = err.message;
                if (msg.includes("Please enter a question")) {
                    warnAlert.textContent = msg;
                    warnAlert.style.display = 'flex';
                } else if (msg.includes("Pinecone index not found")) {
                    errorAlert.textContent = msg;
                    errorAlert.style.display = 'flex';
                } else if (msg.includes("No relevant documents found")) {
                    warnAlert.textContent = msg;
                    warnAlert.style.display = 'flex';
                } else {
                    errorAlert.textContent = msg;
                    errorAlert.style.display = 'flex';
                }
            } finally {
                loading.style.display = 'none';
                askBtn.disabled = false;
            }
        }

        // Initialize status
        checkStatus();
    </script>
</body>
</html>
"""


class RAGRequestHandler(BaseHTTPRequestHandler):
    """Custom request handler for the Naive RAG single-page web server."""

    def log_message(self, format, *args):
        """Silence standard console logging of HTTP requests for cleaner output."""
        return

    def do_GET(self):
        """Serve the beautiful single-page dashboard and status APIs."""
        parsed_path = urlparse(self.path)
        if parsed_path.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif parsed_path.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            gemini_key = os.getenv("GEMINI_API_KEY")
            pinecone_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")

            status = {
                "gemini_connected": bool(gemini_key),
                "pinecone_connected": bool(pinecone_key),
                "index_name": index_name or "Not configured"
            }
            self.wfile.write(json.dumps(status).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        """Handle incoming query submissions, running retrieval and generation."""
        if self.path == '/query':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
            except Exception:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid request payload."}).encode('utf-8'))
                return

            question = data.get('question', '').strip()

            # 1. Handle Missing API Keys
            gemini_key = os.getenv("GEMINI_API_KEY")
            pinecone_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")

            if not gemini_key or not pinecone_key:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing API key. Please check your .env file."}).encode('utf-8'))
                return

            # 2. Handle Empty Query
            if not question:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Please enter a question."}).encode('utf-8'))
                return

            try:
                from utils.pinecone_db import init_pinecone
                from utils.retriever import retrieve_chunks
                from utils.generator import build_prompt, generate_answer

                # Initialize database
                pc = init_pinecone()

                # Get existing index names
                try:
                    existing_indexes = pc.list_indexes().names()
                except AttributeError:
                    existing_indexes = [idx.name for idx in pc.list_indexes()]

                # 3. Handle Missing Index
                if not index_name or index_name not in existing_indexes:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Pinecone index not found. Please run ingest.py first."}).encode('utf-8'))
                    return

                index = pc.Index(index_name)

                # Retrieve matching chunks
                chunks = retrieve_chunks(question, index, top_k=3)

                # 4. Handle No Retrieved Documents
                if not chunks:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No relevant documents found. Try rephrasing your question."}).encode('utf-8'))
                    return

                # Build context
                context = "\n\n".join([c["text"] for c in chunks])

                # Build prompt and generate answer
                prompt = build_prompt(context, question)
                answer = generate_answer(prompt)

                # Send success response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "question": question,
                    "chunks": chunks,
                    "answer": answer
                }).encode('utf-8'))

            # 5. Handle Gemini API/Pinecone failures
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_msg = "Gemini API error: " + str(e)
                self.wfile.write(json.dumps({"error": error_msg}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def run_server(port=8080):
    """Start and run the HTTP server on a specified port."""
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, RAGRequestHandler)
    except OSError as e:
        print(f"Port {port} is already in use. Retrying on port {port + 1}...")
        run_server(port + 1)
        return

    print(f"\n==========================================")
    print(f"Naive RAG Engine Server started successfully!")
    print(f"Local Web App Link: http://localhost:{port}")
    print(f"==========================================\n")

    # Automatically launch user browser to experience the premium interface
    try:
        import antigravity
        # Antigravity hook included cleanly as a package dependency trigger
        # which can open the comic, but for pure developer convenience,
        # we make sure the local dashboard starts up immediately
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        webbrowser.open(f"http://localhost:{port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    run_server(8080)