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

from document_processing import DocumentProcessor
from user_view import QueryProcessor
from response_generation import ResponseGenerator
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

def main():
    import sys
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    # Retrieve API Keys from environment variables, local file, or placeholders
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")

    keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys")
    if os.path.exists(keys_path):
        try:
            with open(keys_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) >= 1 and not pinecone_api_key:
                    pinecone_api_key = lines[0]
                if len(lines) >= 2 and not groq_api_key:
                    groq_api_key = lines[1]
        except Exception:
            pass

    if not pinecone_api_key:
        pinecone_api_key = "YOUR_PINECONE_API_KEY"
    if not groq_api_key:
        groq_api_key = "YOUR_GROQ_API_KEY"

    # Phase 1: Document Processing
    document_processor = DocumentProcessor(pdf_dir="./pdf_files", vector_db_api_key=pinecone_api_key)
    document_processor.upload_to_vector_db()

    # Phase 2: User Query Input
    query_processor = QueryProcessor()
    user_query = input("Please enter your question: ")
    level = input("Please specify the detail level (beginner, intermediate, expert): ").strip().lower()
    processed_query_data = query_processor.process_query(user_query, level)

    # Generate embedding for the processed query
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(processed_query_data["query"])

    # Phase 3: Response Generation
    response_generator = ResponseGenerator(vector_db_api_key=pinecone_api_key, llm_api_key=groq_api_key)
    response = response_generator.respond_to_user(
        query_embedding, processed_query_data["level"], user_question=user_query
    )

    # Output the final response
    print("\nIntelligent Tutor Response:")
    print(response)

if __name__ == "__main__":
    main()
