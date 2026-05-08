from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from VectorTransformer import transformToVector
from PIL import Image #A way to import images
from pathlib import Path
import torch.nn.functional as F
import timeit

qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing, CI/CD
database_path = Path("largerSubset/znr")
valid_extensions = ('.jpg', '.png', 'jpeg')
print("Started qdrant, got path and established valid extensions")

# Creating the collection 
#def create_collection():
#    qdrant.create_collection(
#        collection_name="mtg_cards",
#        vectors_config=VectorParams(
#            size=384, # Needs to be the output from DINOv2
#            distance=Distance.COSINE
#    )   
#)


# Store the data
def dataStorage():
    if not qdrant.collection_exists(collection_name="mtg_cards"):
        qdrant.create_collection(
        collection_name="mtg_cards",
        vectors_config=VectorParams(
            size=384, # Needs to be the output from DINOv2
            distance=Distance.COSINE
            )   
        )

    for idx, file_path in enumerate(database_path.glob('*')):
        if file_path.suffix.lower() in valid_extensions:
            start = timeit.default_timer()
            img = Image.open(file_path).convert('RGB')
            vec = transformToVector(img)
            #vectors.append(vec)
            #filenames.append(file_path.name)

        # Step 2 - qdrant
            qdrant.upsert(
                collection_name="mtg_cards",
                points=[
                    PointStruct(
                        id=idx,
                        vector=vec, 
                        payload={"scryfall_id": str(file_path.stem) }
                    )
                ]
            )
            end = timeit.default_timer()
            print(end - start)


dataStorage()

# Perform check here to establish that vectors are added correctly
point_id = 0  # Replace with the specific ID you want to check
result = qdrant.retrieve(
    collection_name="mtg_cards",
    ids=[point_id],
    with_vectors=True, # Set to True to see the actual vector data
    with_payload=True
)

if result:
    print(f"Retrieved point: {result[0]}")
else:
    print("Point not found.")