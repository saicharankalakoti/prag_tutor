import os
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
import pinecone

class DocumentProcessor:
    def __init__(self, pdf_dir, vector_db_api_key, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_dir = pdf_dir
        self.model = SentenceTransformer(model_name)
        pc = pinecone.Pinecone(api_key=vector_db_api_key)
        self.index = pc.Index("intelligent-tutor")  

    def read_pdf(self, pdf_file):
        text_content = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text()
        return text_content

    def generate_embeddings(self, text):
        sentences = text.split('\n')  
        embeddings = self.model.encode(sentences)
        return sentences, embeddings

    def upload_to_vector_db(self):
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found to process.")
            return

        # Load processed files tracking JSON
        tracking_file = os.path.join(self.pdf_dir, "processed_files.json")
        processed_hashes = {}
        if os.path.exists(tracking_file):
            try:
                with open(tracking_file, "r", encoding="utf-8") as f:
                    processed_hashes = json.load(f)
            except Exception:
                pass

        import hashlib
        updated = False
        print(f"Checking {len(pdf_files)} PDF file(s) for vector database updates...")
        
        for pdf_file in pdf_files:
            filepath = os.path.join(self.pdf_dir, pdf_file)
            
            # Compute MD5 hash of the PDF file
            hasher = hashlib.md5()
            try:
                with open(filepath, "rb") as f:
                    hasher.update(f.read())
                file_hash = hasher.hexdigest()
            except Exception as e:
                print(f"Could not read/hash '{pdf_file}': {e}")
                continue

            # Skip if the file hash matches the stored hash
            if processed_hashes.get(pdf_file) == file_hash:
                print(f"Skipping '{pdf_file}' (already processed and uploaded).")
                continue

            print(f"Reading and processing updated/new file '{pdf_file}'...")
            text = self.read_pdf(filepath)
            print(f"Generating embeddings for '{pdf_file}'...")
            sentences, embeddings = self.generate_embeddings(text)
            print(f"Uploading {len(sentences)} vectors to Pinecone...")
            vectors_to_upsert = []
            for i, embedding in enumerate(embeddings):
                vector = embedding.tolist()
                if len(vector) < 1024:
                    vector = vector + [0.0] * (1024 - len(vector))
                vectors_to_upsert.append((f"{pdf_file}_{i}", vector, {"sentence": sentences[i]}))
                
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
            
            # Update hash cache
            processed_hashes[pdf_file] = file_hash
            updated = True
            print(f"Successfully processed and uploaded '{pdf_file}'.")

        # Save the updated tracking file
        if updated:
            try:
                with open(tracking_file, "w", encoding="utf-8") as f:
                    json.dump(processed_hashes, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not save tracking file: {e}")

if __name__ == "__main__":
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "pcsk_zG6nR_FMnXfTWCTeqwJpVzme97sQrKk1ZY8f5B4SVmQFPacBFqDMuietew7dmScQH4Rvi")
    processor = DocumentProcessor(pdf_dir="./pdf_files", vector_db_api_key=pinecone_api_key)
    processor.upload_to_vector_db()
