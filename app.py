import streamlit as st
import os
from PIL import Image
import io
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import tempfile
import mimetypes
import chromadb
from chromadb.config import Settings
import time

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RAG avec Llama Vision",
    page_icon="🦙",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .upload-section {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #FFFFFF;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration de l'URL Ollama
OLLAMA_BASE_URL = "http://192.168.1.141:11434"

# Configuration de ChromaDB
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Configuration du client ChromaDB
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

def check_ollama_connection(max_retries=3, timeout=5):
    """Vérifie la connexion au serveur Ollama avec plusieurs tentatives."""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
            if response.status_code == 200:
                return True
            else:
                st.error(f"Le serveur Ollama a répondu avec le code {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                st.warning(f"Tentative {attempt + 1}/{max_retries} de connexion au serveur Ollama...")
                time.sleep(1)
            else:
                st.error(f"Impossible de se connecter au serveur Ollama à {OLLAMA_BASE_URL}. Vérifiez que le serveur est en cours d'exécution et accessible.")
                return False
        except requests.exceptions.Timeout:
            st.error(f"Le serveur Ollama ne répond pas (timeout après {timeout}s)")
            return False
        except Exception as e:
            st.error(f"Erreur lors de la connexion au serveur Ollama : {str(e)}")
            return False
    return False

# Initialisation des variables de session
if 'vectorstore' not in st.session_state:
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="llama3.2-vision"
    )
    # Charger la collection existante ou en créer une nouvelle
    try:
        st.session_state.vectorstore = Chroma(
            client=chroma_client,
            collection_name="documents",
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement de la base vectorielle : {str(e)}")
        st.session_state.vectorstore = None

# Liste des fichiers déjà traités
if 'processed_files' not in st.session_state:
    try:
        # Récupérer les métadonnées de tous les documents dans la collection
        collection = chroma_client.get_collection(name="documents")
        metadata_list = collection.get()["metadatas"]
        st.session_state.processed_files = list(set(
            meta.get("source", "") for meta in metadata_list if meta and "source" in meta
        ))
    except Exception as e:
        st.session_state.processed_files = []

def delete_document(filename):
    """Supprime un document de la base vectorielle."""
    try:
        collection = chroma_client.get_collection(name="documents")
        # Récupérer tous les documents
        result = collection.get()
        
        # Trouver les indices des documents à supprimer
        indices_to_delete = [
            i for i, meta in enumerate(result["metadatas"])
            if meta and meta.get("source") == filename
        ]
        
        # Supprimer les documents
        if indices_to_delete:
            ids_to_delete = [result["ids"][i] for i in indices_to_delete]
            collection.delete(ids=ids_to_delete)
            
            # Mettre à jour la liste des fichiers traités
            st.session_state.processed_files.remove(filename)
            return True
    except Exception as e:
        st.error(f"Erreur lors de la suppression du document : {str(e)}")
        return False

def process_file(uploaded_file):
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    try:
        # Détecter le type de fichier basé sur l'extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.txt', '.md', '.csv']:
            loader = TextLoader(file_path)
        else:
            st.error(f"Type de fichier non supporté: {file_extension}")
            return None
        
        documents = loader.load()
        
        # Découpage du texte
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    finally:
        os.unlink(file_path)

def init_vectorstore(texts, source_file):
    """Initialise ou met à jour le vectorstore avec les nouveaux textes."""
    try:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model="llama3.2-vision"
        )
        
        # Ajouter les métadonnées source à chaque document
        for text in texts:
            text.metadata["source"] = source_file
        
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                client=chroma_client,
                collection_name="documents"
            )
        else:
            st.session_state.vectorstore.add_documents(texts)
            
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du vectorstore : {str(e)}")
        raise

def get_system_prompt():
    return """Tu es un assistant IA expert qui répond aux questions en utilisant uniquement le contexte fourni.
Instructions importantes :
1. Base tes réponses UNIQUEMENT sur le contexte donné
2. Si le contexte ne contient pas l'information nécessaire, dis-le clairement
3. Réponds dans la même langue que la question posée
4. Sois précis et concis dans tes réponses
5. Si pertinent, cite les parties spécifiques du contexte
6. N'invente jamais d'informations qui ne sont pas dans le contexte
7. Si la question est ambiguë, demande des précisions

Format de réponse :
- Structure ta réponse de manière claire et logique
- Utilise des listes ou des points si cela améliore la clarté
- Mets en évidence les citations importantes du contexte"""

def query_ollama(question, context):
    try:
        system_prompt = get_system_prompt()
        prompt = f"""System: {system_prompt}

Context: {context}

Question: {question}

Answer: """

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": "llama3.2-vision",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            error_msg = f"Erreur lors de la communication avec le serveur Ollama (code {response.status_code})"
            st.error(error_msg)
            return error_msg
    except requests.exceptions.Timeout:
        error_msg = "Le serveur Ollama ne répond pas (timeout après 30s)"
        st.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Erreur lors de la communication avec le serveur Ollama : {str(e)}"
        st.error(error_msg)
        return error_msg

# Interface utilisateur
st.title("🦙 RAG avec Llama Vision")

# Vérification de la connexion au serveur Ollama
if not check_ollama_connection():
    st.stop()

# Création des onglets
tab_chat, tab_index = st.tabs(["💬 Chat", "📚 Indexation"])

with tab_chat:
    st.header("💭 Chat avec vos documents")
    
    # Paramètres de recherche
    with st.expander("⚙️ Paramètres de recherche", expanded=False):
        k_documents = st.slider("Nombre de documents à utiliser pour le contexte", min_value=1, max_value=10, value=5)

    # Section de question-réponse
    question = st.text_area("Votre question:", height=150, placeholder="Entrez votre question ici...", help="Appuyez sur Ctrl+Enter pour envoyer")

    if question and st.session_state.vectorstore:
        if st.button("Obtenir une réponse", type="primary") or bool(st.session_state.get("last_question") != question and len(question.strip()) > 0):
            st.session_state["last_question"] = question
            if check_ollama_connection():
                with st.spinner("Recherche en cours..."):
                    try:
                        # Récupération du contexte pertinent avec plus de documents
                        docs = st.session_state.vectorstore.similarity_search(question, k=k_documents)
                        
                        # Affichage des documents utilisés
                        st.markdown("### 📚 Documents consultés")
                        doc_sources = set(doc.metadata.get('source', 'Inconnu') for doc in docs)
                        for source in doc_sources:
                            st.write(f"- {source}")
                        
                        # Préparation du contexte avec références
                        context_parts = []
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get('source', 'Inconnu')
                            context_parts.append(f"[Extrait {i} - {source}]\n{doc.page_content}")
                        context = "\n\n".join(context_parts)
                        
                        # Obtention de la réponse
                        response = query_ollama(question, context)
                        
                        # Affichage de la réponse
                        st.markdown("### 💡 Réponse:")
                        st.write(response)
                        
                        # Affichage des extraits utilisés
                        st.markdown("### 📑 Extraits pertinents")
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"Extrait {i} - {doc.metadata.get('source', 'Inconnu')}"):
                                st.markdown(f"```\n{doc.page_content}\n```")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la recherche : {str(e)}")
            else:
                st.error("Impossible d'obtenir une réponse car le serveur Ollama n'est pas accessible.")
    elif question and not st.session_state.vectorstore:
        st.warning("Veuillez d'abord indexer des documents avant de poser une question.")

with tab_index:
    st.header("📁 Indexation des Documents")
    
    # Section d'upload de fichiers
    with st.expander("Ajouter des documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Glissez-déposez vos fichiers ici",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'md', 'csv']
        )

        if uploaded_files:
            # Filtrer les fichiers non traités
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            
            if new_files:
                if st.button(f"Indexer {len(new_files)} nouveaux documents"):
                    if check_ollama_connection():
                        total_files = len(new_files)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file in enumerate(new_files):
                            # Afficher le fichier en cours de traitement
                            status_text.text(f"Traitement de {file.name}...")
                            
                            # Chargement et découpage du document
                            texts = process_file(file)
                            if texts:
                                # Afficher le nombre de chunks créés
                                chunks_count = len(texts)
                                status_text.text(f"Document {file.name} découpé en {chunks_count} segments...")
                                
                                # Indexation des segments
                                with st.spinner(f"Indexation des {chunks_count} segments de {file.name}..."):
                                    try:
                                        init_vectorstore(texts, file.name)
                                        if file.name not in st.session_state.processed_files:
                                            st.session_state.processed_files.append(file.name)
                                    except Exception as e:
                                        st.error(f"Erreur lors de l'indexation de {file.name}: {str(e)}")
                                        break
                            
                            # Mise à jour de la barre de progression
                            progress = (i + 1) / total_files
                            progress_bar.progress(progress)
                            status_text.text(f"Progression : {int(progress * 100)}% ({i + 1}/{total_files} fichiers)")
                        
                        if len(st.session_state.processed_files) >= len(uploaded_files):
                            status_text.text("✅ Indexation terminée!")
                            st.success(f"Tous les nouveaux documents ont été indexés avec succès! ({total_files} fichiers traités)")
                    else:
                        st.error("L'indexation ne peut pas démarrer car le serveur Ollama n'est pas accessible.")
            else:
                st.info("Tous les documents sélectionnés ont déjà été indexés.")

    # Section des fichiers indexés avec option de suppression
    if st.session_state.processed_files:
        st.subheader("📚 Documents indexés")
        for file in st.session_state.processed_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- {file}")
            with col2:
                if st.button("🗑️ Supprimer", key=f"delete_{file}"):
                    if delete_document(file):
                        st.success(f"Document {file} supprimé avec succès!")
                        st.rerun()
