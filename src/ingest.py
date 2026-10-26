import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("📥 Chargement des circulaires BAM...")
loader = PyPDFDirectoryLoader("../data/bam_circulars")
docs = loader.load()

if not docs:
    print("❌ Erreur : Aucun PDF trouvé.")
    exit()

print(f"📄 {len(docs)} pages chargées.")

# VÉRIFICATION DE SÉCURITÉ : Le PDF contient-il du texte ?
total_text = "".join([doc.page_content for doc in docs])
if len(total_text.strip()) == 0:
    print("❌ Erreur : Le PDF est vide ou c'est un document scanné (composé d'images).")
    print("💡 Solution : Remplace-le par un PDF 'natif' où tu peux sélectionner le texte avec ta souris.")
    exit()

# 2. Découper le texte
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Textes découpés en {len(chunks)} morceaux (chunks).")

# 3. Création des embeddings
print("🧠 Chargement du modèle d'embedding et création des vecteurs...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Sauvegarde dans ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="../chroma_db"
)

print("✅ Ingestion terminée ! La base vectorielle est sauvegardée dans le dossier 'chroma_db'.")