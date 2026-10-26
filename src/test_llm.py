import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Charger la variable d'environnement (GROQ_API_KEY)
load_dotenv("../.env")

print("🧠 Connexion à l'API Groq (Modèle Llama-3 70B)...")

# 2. Initialiser le LLM
# La température à 0 garantit des réponses factuelles (idéal pour la banque)

llm = ChatGroq(
        temperature=0,
        model_name="openai/gpt-oss-120b" 
    )

# 3. Poser une question test basique
question = "Explique en une seule phrase ce qu'est le risque de liquidité pour une banque."
print(f"\n🗣️ Question : {question}")

print("\n⏳ Génération de la réponse...\n")
reponse = llm.invoke(question)

print("🤖 Réponse :")
print(reponse.content)