from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("🧠 Chargement de la base vectorielle...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="../chroma_db", embedding_function=embeddings)

# On configure le retriever pour ramener les 3 meilleurs passages (chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Une question métier très précise
question = "Quelles sont les conditions pour clôturer un compte inactif à l'initiative de la banque ?" 

print(f"\n🔍 Question posée : {question}")
print("📚 Recherche dans les documents...\n")

# Lancement de la recherche
results = retriever.invoke(question)

for i, doc in enumerate(results):
    print(f"--- Résultat {i+1} ---")
    print(doc.page_content)
    print("-" * 50 + "\n")