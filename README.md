# RAG Application avec Ollama et Streamlit

Application de Retrieval-Augmented Generation (RAG) utilisant Ollama et Streamlit pour créer un chatbot intelligent capable de répondre à des questions basées sur vos documents.

## Fonctionnalités

- 💬 Interface de chat intuitive
- 📚 Indexation de documents (PDF, TXT, MD, CSV)
- 🔍 Recherche sémantique dans les documents
- 🤖 Utilisation du modèle llama3.2-vision
- 📊 Affichage des sources et extraits pertinents
- 🗑️ Gestion des documents (ajout/suppression)

## Prérequis

- Python 3.8+
- Ollama avec le modèle llama3.2-vision
- Un serveur Ollama accessible

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd OllamaClient
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration

1. Assurez-vous que le serveur Ollama est en cours d'exécution
2. Vérifiez que le modèle llama3.2-vision est installé :
```bash
ollama pull llama3.2-vision
```

## Utilisation

1. Lancer l'application :
```bash
streamlit run app.py
```

2. Accéder à l'interface web :
- Ouvrez votre navigateur à l'adresse : http://localhost:8501

3. Utilisation :
- Onglet "Chat" : Posez vos questions sur vos documents
- Onglet "Indexation" : Gérez vos documents (ajout/suppression)

## Structure du Projet

```
OllamaClient/
├── app.py              # Application principale
├── requirements.txt    # Dépendances Python
├── .gitignore         # Fichiers ignorés par Git
├── README.md          # Documentation
└── chroma_db/         # Base de données vectorielle (généré)
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commit vos changements
4. Push sur la branche
5. Ouvrir une Pull Request
