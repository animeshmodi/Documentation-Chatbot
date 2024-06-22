## Document Q&A System

This Document Q&A System allows you to upload a text document or PDF, process it, and then ask questions related to the document. The system identifies the most relevant answers and saves responses to categorized folders on your system.

## Requirements

- Python 3.6 or higher
- The following Python packages:
  - numpy
  - PyPDF2
  - scikit-learn


  How it Works :
1) Upload and process documents.

    - The system can handle both text documents and PDFs.
    - It reads the content and preprocesses it, removing any extra whitespace, special characters (except periods and parentheses), and text within parentheses.
    - The text is divided into sentences and converted to TF-IDF vectors.
       
2) Question Answering.

    - When a question is asked, the system uses cosine similarity to find the most relevant topic and sentences in the document.
    - The most relevant answers are determined by their combined similarity in terms of topic and sentence relevance.
    
3) Saving responses

    - The system saves each question and its responses in a folder structure that is organized according to the identified topics.
    - The primary directory for saving responses is C:/DocumentQA, with subdirectories for each topic.

