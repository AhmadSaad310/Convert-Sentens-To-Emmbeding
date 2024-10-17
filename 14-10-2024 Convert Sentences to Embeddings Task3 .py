from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["This tasks is not easy","The first one i did not know what exactly needed", "Ok mybe this is enough"]

embeddings = model.encode(texts)
print("Embeddings shape:", embeddings.shape)
print("First embedding:", embeddings[0])


vectorstore = FAISS.from_documents(texts, embeddings) 
