from Retriever.querry_embedder import embed_query
from Retriever.retriever  import retrieve
from Retriever.reranker import reranker

query  = "Which interfaces does the BME280 support, I2C or SPI?"

#embed the query 
embedding = embed_query(query=query)
print("Query embedded")
print("Vector dimension:", len(embedding))

#retive candidate form the Qdrant 
results = retrieve(embedding,top_k=10)
print("\nQdrant results:")
print("Number of candidates:", len(results))


#Reranker 
reranker_result = reranker(
    query=query,
    results=results,
    top_k=3
)
print("\n\n========== RERANKED RESULTS ==========")


for result, score in reranker_result:
    print("\nReranker score:", score)
    print("Original Qdrant score:", result.score)
    print("Source:", result.payload["source"])
    print("Content:", result.payload["content"])