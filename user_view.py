import re
import nltk

# Ensure NLTK resources are available
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class QueryProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_query(self, query):
        query = re.sub(r'\W', ' ', query)  # Remove non-alphanumeric characters
        words = word_tokenize(query)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return " ".join(filtered_words)

    def process_query(self, query, level):
        processed_query = self.preprocess_query(query)
        return {"query": processed_query, "level": level}

if __name__ == "__main__":
    query_processor = QueryProcessor()
    user_query = "How does machine learning work?"
    level = "beginner"
    processed_query = query_processor.process_query(user_query, level)
    print(processed_query)
