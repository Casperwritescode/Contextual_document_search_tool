from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to find and display relevant documents
def find_relevant_docs(query_embedding, embeddings_with_summaries, top_n=5):
    similarities = []
    
    # Calculate similarities
    for emb in embeddings_with_summaries:
        doc_embedding = emb['embedding']
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities.append((emb['identifier'], similarity, emb['summary']))
    
    # Sort documents by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print the top matching documents
    print("Top matching documents:")
    for identifier, similarity, summary in similarities[:top_n]:  # Top N matches
        print(f"\nDocument: {identifier}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Summary: {summary}")

# Example usage
find_relevant_docs(query_embedding, embeddings_with_summaries)
