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
    loop_count: int

# 3. Initialiser les outils
print("⚙️ Initialisation des composants...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="../chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")

# 4. Prompts
router_prompt = PromptTemplate(
    template="""Tu es un expert en classification. 
    Analyse la demande de l'utilisateur.
    - Si c'est une simple salutation (bonjour, salut, hi, hello) ou une formule de politesse (merci, au revoir), réponds UNIQUEMENT par le mot : BAVARDAGE
    - Si c'est une vraie question ou une recherche d'information (même floue comme "je veux créer un compte"), réponds UNIQUEMENT par le mot : BANQUE
    
    Demande : {question}
    Classification :""",
    input_variables=["question"]
)

generate_prompt = PromptTemplate(
    template="""Tu es un expert strict spécialisé UNIQUEMENT dans la réglementation de Bank Al-Maghrib (Maroc). 
    
    RÈGLE D'OR :
    Réponds à la question en te basant STRICTEMENT sur le contexte fourni. 
    Si la réponse ne se trouve pas clairement dans le contexte, tu ne dois formuler aucune supposition et répondre EXACTEMENT par la phrase suivante : 
    "Désolé, cette information ne figure pas dans les circulaires de Bank Al-Maghrib que j'ai en mémoire."
    
    N'invente aucune procédure et ne mentionne jamais d'autres pays ou juridictions.

    Contexte : {context}
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
    template="""La question suivante n'a pas donné de bons résultats. 
    Reformule-la pour qu'elle soit plus précise pour une recherche dans des circulaires bancaires de Bank Al-Maghrib (Maroc). 
    
    RÈGLE ABSOLUE : Ne mentionne JAMAIS la France ou la réglementation française. La question doit cibler uniquement le droit marocain.
    Ne renvoie QUE la nouvelle question reformulée.
    
    Question initiale: {question}
    Nouvelle question:""",
    input_variables=["question"]
)

# 5. Nœuds (Nodes)
def route_question(state: AgentState):
    """Ce n'est pas un nœud standard, c'est la fonction de routage conditionnel."""
    print("🚦 [Router] Analyse de l'intention de l'utilisateur...")
    chain = router_prompt | llm
    result = chain.invoke({"question": state["question"]}).content.strip().upper()
    
    if "BAVARDAGE" in result:
        print("   -> Intention détectée : Salutations/Bavardage")
        return "chat"
    else:
        print("   -> Intention détectée : Question Métier (Banque)")
        return "retrieve"

def chat_node(state: AgentState):
    """Gère les salutations sans interroger ChromaDB."""
    print("💬 [Node: Chat] Réponse directe de courtoisie...")
    return {"answer": "Bonjour ! Je suis l'agent IA expert en réglementation de Bank Al-Maghrib. Comment puis-je vous aider aujourd'hui ?"}

def retrieve_node(state: AgentState):
    print(f"🔍 [Node: Retrieve] Recherche pour : '{state['question']}'")
    docs = retriever.invoke(state["question"])
    return {"context": docs, "loop_count": state.get("loop_count", 0) + 1}

def grade_documents_node(state: AgentState):
    print("⚖️ [Node: Grade] Évaluation de la pertinence des documents...")
    docs_text = "\n\n".join([doc.page_content for doc in state["context"]])
    chain = grade_prompt | llm
    score = chain.invoke({"context": docs_text, "question": state["question"]}).content.strip().upper()
    print(f"   -> Verdict du Juge : {score}")
    return {"answer": score} 

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

def decide_to_generate(state: AgentState):
    if "OUI" in state["answer"]:
        return "generate"
    else:
        if state["loop_count"] >= 3:
            print("⚠️ Limite de boucles atteinte, on génère avec ce qu'on a.")
            return "generate"
        return "rewrite"

# 6. Construire le Graphe
workflow = StateGraph(AgentState)

# Ajouter tous les nœuds
workflow.add_node("chat", chat_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_question", rewrite_question_node)
workflow.add_node("generate", generate_node)

# Le point d'entrée conditionnel (Le Routeur)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "chat": "chat",
        "retrieve": "retrieve"
    }
)

# Les chemins classiques
workflow.add_edge("chat", END)
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite_question"
    }
)

workflow.add_edge("rewrite_question", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    print("\n🚀 Lancement de l'Agent Autonome avec Routeur...")
    
    # Test 1 : Salutation
    print("\n--- TEST 1 : BAVARDAGE ---")
    inputs_chat = {"question": "Salut !", "loop_count": 0}
    result_chat = app.invoke(inputs_chat)
    print(f"🤖 Agent : {result_chat['answer']}")
    
    # Test 2 : Vraie question
    print("\n--- TEST 2 : QUESTION MÉTIER ---")
    inputs_bank = {"question": "Quel est le délai de préavis pour clôturer un compte ?", "loop_count": 0}
    result_bank = app.invoke(inputs_bank)
    print(f"🤖 Agent : {result_bank['answer']}")