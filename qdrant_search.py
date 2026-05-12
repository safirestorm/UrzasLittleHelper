from qdrant_client import QdrantClient
from VectorTransformer import transformToVector
from PIL import Image #A way to import images
from pathlib import Path

qdrant = QdrantClient(url="http://localhost:6333") 

def makeQuery(query_img):
    query_vector=transformToVector(query_img)

    hits = qdrant.query_points(
        collection_name="mtg_cards",
        query=query_vector,
        limit=5 # returns the 5 closest points
    )

    return hits

