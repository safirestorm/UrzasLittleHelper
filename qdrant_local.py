from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing, CI/CD

# Creating the collection 
def create_collection():
    qdrant.create_collection(
        collection_name="mtg_cards",
        vectors_config=VectorParams(
            size=768, # Needs to be the output from DINOv2
            distance=Distance.COSINE
    )
)



# Store the data
def dataStorage():
    if not qdrant.collection_exists(collection_name={"mtg_cards"}):
        create_collection()

# Step 1 - Dino

# Step 2 - qdrant

