import streamlit as st
from agent import app

# Configuration de la page
st.set_page_config(page_title="Agent BAM", page_icon="🏦", layout="centered")

st.title("🏦 Agent IA - Bank Al-Maghrib")
st.markdown("Posez vos questions sur la réglementation bancaire marocaine. L'agent LangGraph recherchera et vérifiera les circulaires avant de répondre.")

# Initialisation de l'historique de chat dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Champ de saisie utilisateur
if prompt := st.chat_input("Ex: Quel est le délai de préavis pour clôturer un compte ?"):
    
    # Afficher la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lancer l'agent LangGraph
    with st.chat_message("assistant"):
        with st.spinner("🧠 L'agent réfléchit (Recherche, Évaluation et Génération)..."):
            # On appelle notre graphe compilé depuis agent.py
            result = app.invoke({"question": prompt, "loop_count": 0})
            reponse_finale = result["answer"]
            
            # Afficher la réponse
            st.markdown(reponse_finale)
            
    # Sauvegarder la réponse dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse_finale})