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
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "pcsk_2DGukt_8pVeAnDxKKhMFQZ54iar6WcJAyjEUtnNC9w9qRSj8jwg3Xu9QnFUnRNLGGzBwF6")
    processor = DocumentProcessor(pdf_dir="./pdf_files", vector_db_api_key=pinecone_api_key)
    processor.upload_to_vector_db()

# import os
# import json
# import hashlib
# import pdfplumber
# from sentence_transformers import SentenceTransformer
# import pinecone


# class DocumentProcessor:

#     def __init__(
#         self,
#         pdf_dir,
#         vector_db_api_key,
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     ):
#         self.pdf_dir = pdf_dir

#         # Load embedding model
#         self.model = SentenceTransformer(model_name)

#         # Connect to Pinecone
#         pc = pinecone.Pinecone(api_key=vector_db_api_key)

#         self.index = pc.Index("intelligent-tutor")

#     def read_pdf(self, pdf_file):

#         pages = []

#         with pdfplumber.open(pdf_file) as pdf:

#             for page_number, page in enumerate(pdf.pages, start=1):

#                 page_text = page.extract_text()

#                 if page_text:
#                     pages.append({
#                         "page": page_number,
#                         "text": page_text
#                     })

#         return pages

#     def chunk_text(self, text, chunk_size=1000, overlap=200):

#         chunks = []

#         start = 0

#         while start < len(text):

#             end = start + chunk_size

#             chunk = text[start:end]

#             if chunk.strip():
#                 chunks.append(chunk)

#             start += chunk_size - overlap

#         return chunks

#     def generate_embeddings(self, chunks):

#         embeddings = self.model.encode(
#             chunks,
#             show_progress_bar=True
#         )

#         return embeddings

#     def upload_to_vector_db(self):

#         if not os.path.exists(self.pdf_dir):
#             os.makedirs(self.pdf_dir)

#         pdf_files = [
#             f for f in os.listdir(self.pdf_dir)
#             if f.endswith(".pdf")
#         ]

#         if not pdf_files:
#             print("No PDF files found.")
#             return

#         tracking_file = os.path.join(
#             self.pdf_dir,
#             "processed_files.json"
#         )

#         processed_hashes = {}

#         if os.path.exists(tracking_file):

#             try:
#                 with open(
#                     tracking_file,
#                     "r",
#                     encoding="utf-8"
#                 ) as f:

#                     processed_hashes = json.load(f)

#             except Exception:
#                 pass

#         for pdf_file in pdf_files:

#             filepath = os.path.join(
#                 self.pdf_dir,
#                 pdf_file
#             )

#             # Calculate file hash
#             hasher = hashlib.md5()

#             with open(filepath, "rb") as f:
#                 hasher.update(f.read())

#             file_hash = hasher.hexdigest()

#             # Skip already processed files
#             if processed_hashes.get(pdf_file) == file_hash:

#                 print(
#                     f"Skipping {pdf_file} "
#                     f"(already processed)"
#                 )

#                 continue

#             print(f"\nProcessing: {pdf_file}")

#             pages = self.read_pdf(filepath)

#             all_chunks = []

#             # Create chunks page by page
#             for page_data in pages:

#                 page_number = page_data["page"]

#                 chunks = self.chunk_text(
#                     page_data["text"],
#                     chunk_size=1000,
#                     overlap=200
#                 )

#                 for chunk in chunks:

#                     all_chunks.append({
#                         "text": chunk,
#                         "page": page_number
#                     })

#             print(
#                 f"Created {len(all_chunks)} chunks"
#             )

#             texts = [
#                 item["text"]
#                 for item in all_chunks
#             ]

#             print("Generating embeddings...")

#             embeddings = self.generate_embeddings(texts)

#             vectors_to_upsert = []

#             for i, embedding in enumerate(embeddings):

#                 vector_id = f"{pdf_file}_chunk_{i}"

#                 metadata = {
#                     "text": all_chunks[i]["text"],
#                     "source": pdf_file,
#                     "page": all_chunks[i]["page"],
#                     "chunk": i
#                 }

#                 vectors_to_upsert.append({
#                     "id": vector_id,
#                     "values": embedding.tolist(),
#                     "metadata": metadata
#                 })

#                 # Upload in batches
#                 if len(vectors_to_upsert) >= 100:

#                     self.index.upsert(
#                         vectors=vectors_to_upsert
#                     )

#                     vectors_to_upsert = []

#             # Upload remaining vectors
#             if vectors_to_upsert:

#                 self.index.upsert(
#                     vectors=vectors_to_upsert
#                 )

#             # Save processed hash
#             processed_hashes[pdf_file] = file_hash

#             print(
#                 f"Successfully uploaded "
#                 f"{len(all_chunks)} chunks."
#             )

#         # Save tracking file
#         with open(
#             tracking_file,
#             "w",
#             encoding="utf-8"
#         ) as f:

#             json.dump(
#                 processed_hashes,
#                 f,
#                 indent=4
#             )


# if __name__ == "__main__":

#     pinecone_api_key = os.environ.get(
#         "PINECONE_API_KEY"
#     )

#     processor = DocumentProcessor(
#         pdf_dir="./pdf_files",
#         vector_db_api_key=pinecone_api_key
#     )

#     processor.upload_to_vector_db()