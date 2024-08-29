# Contextual_document_search_tool
Contextual search tool with .txt document/query embedding for similarity comparison and document selection + summarization

Code utilizes PySpark framework and locally mounted directory

Workflow:

1. Execute Doc_ingestion.py to ingest documents and collect text
2. Execute BART_transformers_doc_embedding.py to embed documents
3. Execute BART_transformers_query_embedding to embed manually defined query
4. Complete with similarity assessment with Doc_query_similarity_assessment.py