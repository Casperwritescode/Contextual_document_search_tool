from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import datetime

# Load BART model and tokenizer for summarization and embeddings
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Set model to evaluation mode (disables dropout)
model.eval()

# Function to generate summary using BART
def generate_summary(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate embeddings using BART
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.model(**inputs)  # Use the underlying model for embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Encode documents with identifier, metadata, and summary
embeddings_with_summaries = []
for i, doc in enumerate(doc_text.toPandas()['value'].tolist()):
    try:
        # Generate metadata
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = doc_text.select('file_name').collect()[i][0]
        identifier = f"{file_name}_{i+1}"
        
        # Generate embedding
        embedding = generate_embedding(doc)
        
        # Generate summary
        summary = generate_summary(doc)
        
        # Combine embedding, summary, and metadata
        embedding_with_metadata = {
            'identifier': identifier,
            'timestamp': timestamp,
            'embedding': embedding,
            'summary': summary
        }
        embeddings_with_summaries.append(embedding_with_metadata)
    except Exception as e:
        print(f"Error processing document {i}: {e}")