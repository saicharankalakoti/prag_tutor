import os
import json
import re
from typing import Dict, List, Optional, Any

from groq import Groq
import pinecone


class ResponseGenerator:
    def __init__(
        self,
        vector_db_api_key: str,
        llm_api_key: str,
        model_name: str = "qwen/qwen3.8-27b",
        prerequisites_file: Optional[str] = None,
    ):
        if not llm_api_key:
            raise ValueError("GROQ_API_KEY is not set. Set it in the environment before starting the app.")
        try:
            pc = pinecone.Pinecone(api_key=vector_db_api_key)
            self.index = pc.Index("intelligent-tutor")
        except Exception as e:
            self.index = None
            print(f"[Notice] Pinecone index initialization: {e}")

        try:
            self.groq_client = Groq(api_key=llm_api_key)
        except Exception:
            self.groq_client = None
        self.model_name = model_name

        # Load prerequisites mapping from JSON file (source of truth)
        self.prerequisites_data = self._load_prerequisites(prerequisites_file)

    def _load_prerequisites(self, prerequisites_file: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Loads topic prerequisites from the JSON folder / file.
        Does not hard-code any prerequisites in Python.
        """
        target_path = None
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if prerequisites_file and os.path.exists(prerequisites_file):
            target_path = prerequisites_file
        else:
            candidate = os.path.join(base_dir, "JSONS", "data_structures_prerequisites.json")
            if os.path.exists(candidate):
                target_path = candidate
            else:
                for folder in ["JSONS", "JSON"]:
                    folder_path = os.path.join(base_dir, folder)
                    if os.path.exists(folder_path):
                        for f in os.listdir(folder_path):
                            if f.lower().endswith(".json"):
                                target_path = os.path.join(folder_path, f)
                                break
                    if target_path:
                        break

        if not target_path or not os.path.exists(target_path):
            return {}

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "prerequisites" in data and isinstance(data["prerequisites"], dict):
                raw_prereqs = data["prerequisites"]
            elif isinstance(data, dict):
                raw_prereqs = data
            else:
                raw_prereqs = {}

            cleaned: Dict[str, List[str]] = {}
            for topic, prereqs in raw_prereqs.items():
                topic_str = str(topic).strip()
                if not topic_str:
                    continue
                if isinstance(prereqs, list):
                    cleaned[topic_str] = [str(p).strip() for p in prereqs if str(p).strip()]
                elif prereqs is None:
                    cleaned[topic_str] = []
                else:
                    cleaned[topic_str] = [str(prereqs).strip()]

            return cleaned
        except Exception as e:
            print(f"Warning: Could not load prerequisites from {target_path}: {e}")
            return {}

    def identify_topic_from_rag(
        self,
        matches: List[Any],
        user_question: Optional[str] = None
    ) -> Optional[str]:
        """
        Reuses the topic produced/identified by the existing RAG pipeline.
        First inspects match metadata, then matches against known topics in the
        prerequisites JSON from the user question or retrieved context.
        """
        # 1. Check if RAG match metadata explicitly contains a topic
        for match in matches:
            meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            if isinstance(meta, dict):
                if meta.get("topic"):
                    return str(meta["topic"]).strip()
                if meta.get("title"):
                    return str(meta["title"]).strip()

        # 2. Match known topics from the prerequisites JSON against user question or RAG chunks
        if self.prerequisites_data:
            # Sort topics by descending length so specific terms match before sub-terms
            sorted_topics = sorted(self.prerequisites_data.keys(), key=lambda x: len(x), reverse=True)

            # Check user question if provided
            if user_question:
                q_lower = user_question.lower()
                for t in sorted_topics:
                    pattern = r'\b' + re.escape(t.lower()) + r'\b'
                    if re.search(pattern, q_lower):
                        return t
                    if t.lower() in q_lower:
                        return t

            # Check retrieved sentence context from RAG matches
            for match in matches:
                meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
                sentence = (meta.get("sentence", "") if isinstance(meta, dict) else "").lower()
                for t in sorted_topics:
                    pattern = r'\b' + re.escape(t.lower()) + r'\b'
                    if re.search(pattern, sentence):
                        return t
                    if t.lower() in sentence:
                        return t

        return None

    def get_prerequisites_for_topic(self, topic: Optional[str]) -> List[str]:
        """
        Retrieves only the prerequisites corresponding to the identified topic.
        Returns an empty list if the topic has no prerequisites or is unknown.
        """
        if not topic or not self.prerequisites_data:
            return []

        # Exact match
        if topic in self.prerequisites_data:
            return list(self.prerequisites_data[topic])

        # Case-insensitive match
        topic_lower = topic.strip().lower()
        for t, prereqs in self.prerequisites_data.items():
            if t.lower() == topic_lower:
                return list(prereqs)

        return []

    def fetch_answer(self, query_embedding):
        if self.index is None:
            return []
        # Convert numpy array to list if needed
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        if len(query_embedding) < 1024:
            query_embedding = query_embedding + [0.0] * (1024 - len(query_embedding))
        results = self.index.query(vector=query_embedding, top_k=3, include_metadata=True)
        return results['matches']

    def build_dynamic_prompt(
        self,
        context: str,
        level: str,
        prerequisites: Optional[List[str]] = None,
        user_question: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        """
        Constructs the dynamic prompt with prerequisite context instructions.
        """
        cleaned_prereqs = [p.strip() for p in prerequisites if str(p).strip()] if prerequisites else []
        topic_display = topic or "the requested topic"
        question_header = f"User Question: {user_question}\n" if user_question else ""
        topic_header = f"Identified Topic: {topic}\n" if topic else ""

        if cleaned_prereqs:
            prereqs_str = ", ".join(cleaned_prereqs)
            prompt = (
                f"{question_header}{topic_header}"
                f"Course Material Context:\n{context}\n\n"
                f"Prerequisites for {topic_display}: {prereqs_str}\n\n"
                f"Instructions for Response:\n"
                f"1. Prerequisite Overview:\n"
                f"   - First, provide a very short and basic overview of the prerequisite(s) ({prereqs_str}).\n"
                f"   - Provide ONLY enough background knowledge for the user to understand {topic_display}.\n"
                f"   - Do NOT give lengthy explanations of the prerequisites (1 to 2 concise sentences per prerequisite).\n"
                f"   - Start with: 'Before learning {topic_display}, you should have a basic understanding of:' followed by brief bullet points.\n"
                f"2. Main Topic Explanation:\n"
                f"   - Immediately following the prerequisite overview, continue with the normal, clear explanation of {topic_display} "
                f"at a {level} level of understanding using the provided course context."
            )
        else:
            prompt = (
                f"{question_header}{topic_header}"
                f"Course Material Context:\n{context}\n\n"
                f"Instructions for Response:\n"
                f"Please explain {topic_display} at a {level} level of understanding using the provided course context.\n"
                f"There are no prerequisites required for this topic, so do not generate a prerequisite section. "
                f"Explain the main topic normally."
            )

        return prompt

    def generate_response(
        self,
        matches: List[Any],
        level: str,
        prerequisites: Optional[List[str]] = None,
        user_question: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        sentences = []
        for match in matches:
            sentence = match.get("metadata", {}).get("sentence", "") if isinstance(match, dict) else getattr(match, "metadata", {}).get("sentence", "")
            # Truncate each individual sentence to avoid huge chunks from the PDF
            if len(sentence) > 400:
                sentence = sentence[:400] + "..."
            sentences.append(sentence)
        context = " ".join(sentences)
        # Keep total context well within Groq's request size limits
        if len(context) > 1500:
            context = context[:1500] + "..."

        prompt = self.build_dynamic_prompt(
            context=context,
            level=level,
            prerequisites=prerequisites,
            user_question=user_question,
            topic=topic
        )

        candidate_models = [self.model_name]
        for fallback in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "groq/compound-mini"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model in candidate_models:
            try:
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful and knowledgeable tutor."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=850,
                )
                raw_content = response.choices[0].message.content
                # Strip completed think blocks
                cleaned = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
                # Strip incomplete/truncated think blocks if they exist
                cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
                return cleaned.strip()
            except Exception as e:
                last_error = e
                # If rate limited, try next candidate model
                continue

        # If all candidate models failed, return graceful error notice
        prereq_note = f"\nIdentified Prerequisites: {', '.join(prerequisites)}\n" if prerequisites else ""
        return (
            f"\n[Warning] Groq API call failed: {last_error}\n"
            f"{prereq_note}"
            f"However, the system successfully retrieved the following relevant context from the vector database:\n"
            f"--------------------------------------------------------------------------------\n"
            f"{context}\n"
            f"--------------------------------------------------------------------------------"
        )

    def respond_to_user(
        self,
        query_embedding: Any,
        level: str,
        user_question: Optional[str] = None,
        topic: Optional[str] = None,
        prerequisites: Optional[List[str]] = None
    ) -> str:
        # Step 1: Retrieve RAG matches
        matches = self.fetch_answer(query_embedding)

        # Step 2: Reuse topic identified by RAG pipeline if not explicitly passed
        if not topic:
            topic = self.identify_topic_from_rag(matches, user_question)

        # Step 3: Retrieve only the prerequisites corresponding to that particular topic
        if prerequisites is None:
            prerequisites = self.get_prerequisites_for_topic(topic)

        # Step 4 & 5: Pass retrieved prerequisites to dynamic prompt & response generation
        return self.generate_response(
            matches=matches,
            level=level,
            prerequisites=prerequisites,
            user_question=user_question,
            topic=topic
        )


if __name__ == "__main__":
    query_embedding = [0.2, 0.1, 0.0]  # example embedding array
    level = "beginner"
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY", "PASTE_YOUR_NEW_GROQ_KEY_HERE")

    try:
        generator = ResponseGenerator(vector_db_api_key=pinecone_api_key, llm_api_key=groq_api_key)
        response = generator.respond_to_user(query_embedding, level, user_question="Explain Binary Search")
        print(response)
    except Exception as err:
        print(f"Demo run: {err}")
