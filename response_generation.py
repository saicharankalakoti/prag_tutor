
import os

from groq import Groq
from sentence_transformers import util
import pinecone

class ResponseGenerator:
    def __init__(self, vector_db_api_key, llm_api_key, model_name="qwen/qwen3.6-27b"):
        if not llm_api_key:
            raise ValueError("GROQ_API_KEY is not set. Set it in the environment before starting the app.")
        pc = pinecone.Pinecone(api_key=vector_db_api_key)
        self.index = pc.Index("intelligent-tutor")
        self.groq_client = Groq(api_key=llm_api_key)
        self.model_name = model_name

    def fetch_answer(self, query_embedding):
        # Convert numpy array to list if needed
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        if len(query_embedding) < 1024:
            query_embedding = query_embedding + [0.0] * (1024 - len(query_embedding))
        results = self.index.query(vector=query_embedding, top_k=3, include_metadata=True)
        return results['matches']

    def generate_response(self, matches, level):
        sentences = []
        for match in matches:
            sentence = match.get("metadata", {}).get("sentence", "")
            # Truncate each individual sentence to avoid huge chunks from the PDF
            if len(sentence) > 400:
                sentence = sentence[:400] + "..."
            sentences.append(sentence)
        context = " ".join(sentences)
        # Keep total context well within Groq's request size limits
        if len(context) > 1500:
            context = context[:1500] + "..."

        prompt = f"Answer this question at a {level} level of understanding: {context}"

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful and knowledgeable tutor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
            )
            raw_content = response.choices[0].message.content
            import re
            # Strip completed think blocks
            cleaned = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
            # Strip incomplete/truncated think blocks if they exist
            cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
            return cleaned.strip()
        except Exception as e:
            return (
                f"\n[Warning] Groq API call failed: {e}\n"
                f"However, the system successfully retrieved the following relevant context from the vector database:\n"
                f"--------------------------------------------------------------------------------\n"
                f"{context}\n"
                f"--------------------------------------------------------------------------------"
            )

    def respond_to_user(self, query_embedding, level):
        matches = self.fetch_answer(query_embedding)
        return self.generate_response(matches, level)

if __name__ == "__main__":
    query_embedding = [0.2, 0.1, 0.0]  # example embedding array
    level = "beginner"
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "pcsk_zG6nR_FMnXfTWCTeqwJpVzme97sQrKk1ZY8f5B4SVmQFPacBFqDMuietew7dmScQH4Rvi")
    groq_api_key = "PASTE_YOUR_NEW_GROQ_KEY_HERE"
    generator = ResponseGenerator(vector_db_api_key=pinecone_api_key, llm_api_key=groq_api_key)
    response = generator.respond_to_user(query_embedding, level)
    print(response)
