import os
import re
import random
import io
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentQA:
    def __init__(self):
        self.document_text = ""
        self.sentences = []
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        self.tfidf_matrix = None
        self.topics = [
            "Agriculture",
            "Finance",
            "Education",
            "Health",
            "Infrastructure"
        ]
        self.topic_vectors = None
        self.base_directory = "C:/DocumentQA"

    def upload_document(self, file_path):
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                self.read_pdf(f.read())
        else:
            with open(file_path, 'rb') as f:
                self.read_text(f.read())

    def read_pdf(self, content):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        self.document_text = ""
        for page in pdf_reader.pages:
            self.document_text += page.extract_text()
        print("Successfully read PDF document.")

    def read_text(self, content):
        encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
        for encoding in encodings:
            try:
                self.document_text = content.decode(encoding)
                print(f"Successfully read using {encoding} encoding.")
                return
            except UnicodeDecodeError:
                print(f"Failed to decode with {encoding}.")
        print("All standard decodings failed. Reading as binary and ignoring errors.")
        self.document_text = content.decode('utf-8', errors='ignore')
        print("Document read with potential character loss.")

    def preprocess_text(self):
        self.document_text = re.sub(r'\s+', ' ', self.document_text)  # Remove extra whitespace
        self.document_text = re.sub(r'[^\w\s.()]', '', self.document_text)  # Remove special characters except periods and parentheses
        self.document_text = re.sub(r'\(.*?\)', '', self.document_text)  # Remove text within parentheses

    def process_document(self):
        self.preprocess_text()
        self.sentences = re.split(r'(?<=[.!?])\s+', self.document_text)
        self.sentences = [sent.strip() for sent in self.sentences if len(sent.split()) > 5]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sentences)
        self.topic_vectors = self.tfidf_vectorizer.transform(self.topics)

    def get_topic(self, vector):
        similarities = cosine_similarity(vector, self.topic_vectors).flatten()
        return self.topics[np.argmax(similarities)]

    def answer_question(self, user_input):
        user_vector = self.tfidf_vectorizer.transform([user_input])
        
        # Find the most relevant topic
        topic_similarities = cosine_similarity(user_vector, self.topic_vectors).flatten()
        most_relevant_topic_idx = np.argmax(topic_similarities)
        most_relevant_topic = self.topics[most_relevant_topic_idx]
        
        # Find sentences similar to both the question and the most relevant topic
        sentence_similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        topic_sentence_similarities = cosine_similarity(self.topic_vectors[most_relevant_topic_idx], self.tfidf_matrix).flatten()
        
        # Combine similarities with higher weight on topic relevance
        combined_similarities = (0.3 * sentence_similarities) + (0.7 * topic_sentence_similarities)
        top_indices = combined_similarities.argsort()[-7:][::-1]  # Get top 7 most relevant sentences

        responses = []
        for idx in top_indices:
            if combined_similarities[idx] > 0.005:  # Lower threshold for more results
                responses.append(f"Relevance: {combined_similarities[idx]:.2f}\nA: {self.sentences[idx]}")

        if responses:
            self.save_response(most_relevant_topic, user_input, responses)
            return f"Topic: {most_relevant_topic}\n\n" + "\n\n".join(responses)
        else:
            return "I'm sorry, I couldn't find a relevant answer to your question in the document."

    def get_random_question(self):
        if self.sentences:
            random_sentence = random.choice(self.sentences)
            words = random_sentence.split()
            if len(words) > 3:
                question_start = random.choice(["What is", "Can you explain", "Tell me about"])
                question_end = " ".join(words[:3]).rstrip('.,!?') + "?"
                sentence_vector = self.tfidf_vectorizer.transform([random_sentence])
                topic = self.get_topic(sentence_vector)
                return f"{question_start} {question_end}", random_sentence, topic
        return "No questions available.", "", ""

    def save_response(self, topic, question, responses):
        topic_folder = os.path.join(self.base_directory, topic)
        os.makedirs(topic_folder, exist_ok=True)
        
        file_path = os.path.join(topic_folder, "responses.txt")
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"Q: {question}\n")
            for response in responses:
                file.write(f"{response}\n\n")
            file.write("="*50 + "\n")

def main():
    qa_system = DocumentQA()

    print("Welcome to the Document Q&A System!")
    print("Please provide the path to a text document or PDF to begin.")

    file_path = input("Enter the file path: ").strip()
    qa_system.upload_document(file_path)
    qa_system.process_document()

    print("\nDocument processed. Identified topics:")
    for i, topic in enumerate(qa_system.topics):
        print(f"Topic {i+1}: {topic}")

    print("\nYou can now ask questions or type 'random' for a random question.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() == 'quit':
            print("Thank you for using the Document Q&A System. Goodbye!")
            break
        elif user_input.lower() == 'random':
            question, answer, topic = qa_system.get_random_question()
            print(f"\nRandom Question: {question}")
            print(f"Topic: {topic}")
            print(f"Answer: {answer}")
        else:
            response = qa_system.answer_question(user_input)
            print(response)

if __name__ == "__main__":
    main()
