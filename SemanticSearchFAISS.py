from haystack.pipelines import Pipeline
from haystack.schema import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import torch
import os

FaissIndexPath = r".\faiss_index.faiss"
FaissJsonPath = r".\faiss_index.json"
FaissDbPath = r".\faiss_document_store.db"
EmbeddingModelPath = r"C:\Code\Python\Models\all-mpnet-base-v2"

# use GPU if available and drivers are installed
use_gpu = True if torch.cuda.is_available() else False

if os.path.exists(FaissDbPath):
    os.remove(FaissDbPath)

JsonKnowledgeObject = {}
# Adding Content to be indexed
JsonKnowledgeObject[
    "content"
] = """Faiss (Facebook AI Similarity Search) is an open-source library developed by Facebook, designed for efficient similarity searches and clustering of dense vectors. This library addresses challenges commonly encountered in machine learning applications, particularly those involving high-dimensional vectors, such as image recognition and recommendation systems. Its widespread applicability, combined with features like scalability and flexibility, makes it a valuable tool for various machine learning and data analysis tasks, as demonstrated in its real-world application scenarios outlined in the Facebook Engineering blog post. 

Faiss employs advanced techniques like indexing and quantization to accelerate similarity searches in large datasets. Its versatility is evident in its support for both CPU and GPU implementations, ensuring scalability across different hardware configurations. Faiss offers flexibility with options for both exact and approximate similarity searches, allowing users to tailor the level of precision to their specific requirements."""

# Adding Meta Data
JsonKnowledgeObject["meta"] = {}
JsonKnowledgeObject["meta"][
    "title"
] = "Semantic Search With Facebook AI Similarity Search (FAISS)"
JsonKnowledgeObject["meta"]["author"] = "ThreadWaiting"
JsonKnowledgeObject["meta"][
    "link"
] = "https://threadwaiting.com/semantic-search-with-facebook-ai-similarity-search-faiss/"

# Initialize/Reload Document Store
document_store = FAISSDocumentStore(
    similarity="cosine", sql_url="sqlite:///faiss_document_store.db"
)

# Convert Json object to Document object
document = Document(
    content=JsonKnowledgeObject["content"], meta=JsonKnowledgeObject["meta"]
)

# Add document to the document store
document_store.write_documents([document])

# This needs to be executed every time the data gets refreshed
retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model=EmbeddingModelPath, use_gpu=use_gpu
)
document_store.update_embeddings(retriever)
document_store.save(index_path=FaissIndexPath)


# Load the saved index into anew DocumnetStore instance
document_store = FAISSDocumentStore(faiss_index_path=FaissIndexPath)
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=EmbeddingModelPath,
    model_format="sentence_transformers",
    top_k=3,
    use_gpu=use_gpu,
)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])

output = query_pipeline.run(query="What is Faiss?")

results_documents = output["documents"]

if len(results_documents) > 0:
    print("\nMatching Article: \n")
    for doc in results_documents:
        docDoc = doc.to_dict()
        print(docDoc["meta"]["title"])
        print(docDoc["content"])
        score = round(float(str(docDoc["score"] or "0.0")) * 100, 2)
        print("Match score:", score, "%")
