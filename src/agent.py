import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END

# 1. Charger l'environnement
load_dotenv("../.env")

# 2. Définir le "State"
class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    loop_count: int # Pour éviter les boucles infinies

# 3. Initialiser les outils
print("⚙️ Initialisation des composants...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="../chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")

# 4. Prompts
generate_prompt = PromptTemplate(
    template="""Tu es un assistant strict spécialisé dans la réglementation de Bank Al-Maghrib.
    
    RÈGLES ABSOLUES :
    1. Si l'utilisateur dit juste "bonjour", "salut" ou "hi", réponds poliment : "Bonjour ! Je suis l'agent Bank Al-Maghrib. Quelle est votre question sur la réglementation bancaire ?"
    2. Pour toute autre question, utilise UNIQUEMENT le contexte ci-dessous.
    3. Si la réponse n'est pas dans le contexte ou si la question n'a rien à voir avec la banque, refuse de répondre en disant : "Désolé, je ne réponds qu'aux questions sur la réglementation de Bank Al-Maghrib."

    Contexte :
    {context}
    
    Question: {question}
    Réponse:""",
    input_variables=["context", "question"]
)

grade_prompt = PromptTemplate(
    template="""Tu es un évaluateur strict. Le document ci-dessous contient-il des informations pertinentes pour répondre à la question ? 
    Réponds UNIQUEMENT par 'OUI' ou 'NON'.
    Document: {context}
    Question: {question}
    Évaluation:""",
    input_variables=["context", "question"]
)

rewrite_prompt = PromptTemplate(
    template="""La question suivante n'a pas donné de bons résultats dans notre base de données. 
    Reformule-la pour qu'elle soit plus précise pour une recherche sémantique. Ne renvoie QUE la nouvelle question.
    Question initiale: {question}
    Nouvelle question:""",
    input_variables=["question"]
)

# 5. Nœuds (Nodes)
def retrieve_node(state: AgentState):
    print(f"🔍 [Node: Retrieve] Recherche pour : '{state['question']}'")
    docs = retriever.invoke(state["question"])
    return {"context": docs, "loop_count": state.get("loop_count", 0) + 1}

def grade_documents_node(state: AgentState):
    print("⚖️ [Node: Grade] Évaluation de la pertinence des documents...")
    # On concatène les textes pour l'évaluation
    docs_text = "\n\n".join([doc.page_content for doc in state["context"]])
    chain = grade_prompt | llm
    result = chain.invoke({"context": docs_text, "question": state["question"]})
    
    # On stocke le score (OUI ou NON) temporairement dans le state ou on l'utilise pour le routage
    score = result.content.strip().upper()
    print(f"   -> Verdict du Juge : {score}")
    return {"answer": score} # On utilise "answer" temporairement pour passer le score

def rewrite_question_node(state: AgentState):
    print("🔄 [Node: Rewrite] Reformulation de la question...")
    chain = rewrite_prompt | llm
    new_question = chain.invoke({"question": state["question"]}).content.strip()
    return {"question": new_question}

def generate_node(state: AgentState):
    print("✍️ [Node: Generate] Rédaction de la réponse finale...")
    docs_text = "\n\n".join([doc.page_content for doc in state["context"]])
    chain = generate_prompt | llm
    response = chain.invoke({"context": docs_text, "question": state["question"]})
    return {"answer": response.content}

# 6. Routage Conditionnel (Edge)
def decide_to_generate(state: AgentState):
    """Décide s'il faut générer la réponse ou reformuler la question."""
    score = state["answer"]
    
    if "OUI" in score:
        return "generate"
    else:
        # Sécurité : on arrête après 3 essais pour ne pas bloquer le système
        if state["loop_count"] >= 3:
            print("⚠️ Limite de boucles atteinte, on génère avec ce qu'on a.")
            return "generate"
        return "rewrite"

# 7. Construire le Graphe
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_question", rewrite_question_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# C'est ici que la magie opère : l'embranchement conditionnel
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite_question"
    }
)

workflow.add_edge("rewrite_question", "retrieve") # La Boucle (Cycle) !
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    print("\n🚀 Lancement de l'Agent Autonome...")
    
    # On pose une question volontairement ambiguë ou complexe pour tester le juge
    question = "Je veux arrêter mon contrat bancaire, combien de temps la banque a-t-elle si c'est elle qui décide ?"
    
    inputs = {"question": question, "loop_count": 0}
    result = app.invoke(inputs)
    
    print("\n" + "="*50)
    print(f"🤖 Réponse Finale :\n{result['answer']}")
    print("="*50 + "\n")