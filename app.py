import os
import sys

# Auto-re-execute using the virtual environment if available and not already in it
if os.name == 'nt':
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
else:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")

if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
    import subprocess
    print(f"Virtual environment detected. Restarting script under: {venv_python}...", flush=True)
    sys.exit(subprocess.call([venv_python, "-u"] + sys.argv))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from flask import Flask, render_template, request, jsonify
from document_processing import DocumentProcessor
from user_view import QueryProcessor
from response_generation import ResponseGenerator
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- Configuration ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys")
if os.path.exists(keys_path):
    try:
        with open(keys_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) >= 1 and not PINECONE_API_KEY:
                PINECONE_API_KEY = lines[0]
            if len(lines) >= 2 and not GROQ_API_KEY:
                GROQ_API_KEY = lines[1]
    except Exception:
        pass

if not PINECONE_API_KEY:
    PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
if not GROQ_API_KEY:
    GROQ_API_KEY = "YOUR_GROQ_API_KEY"

# --- Initialize components once at startup ---
print("Initializing document processor...")
document_processor = DocumentProcessor(pdf_dir="./pdf_files", vector_db_api_key=PINECONE_API_KEY)
document_processor.upload_to_vector_db()

print("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Initializing response generator...")
query_processor = QueryProcessor()
response_generator = ResponseGenerator(vector_db_api_key=PINECONE_API_KEY, llm_api_key=GROQ_API_KEY)

print("Server ready!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("question", "").strip()
    level = data.get("level", "beginner").strip().lower()

    if not user_query:
        return jsonify({"error": "Please enter a question."}), 400

    processed = query_processor.process_query(user_query, level)
    query_embedding = embedding_model.encode(processed["query"])
    response = response_generator.respond_to_user(query_embedding, processed["level"])
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
