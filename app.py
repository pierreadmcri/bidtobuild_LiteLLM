import streamlit as st
import pandas as pd
import re
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from litellm import completion, token_counter

# Imports pour lire les vrais fichiers
from pypdf import PdfReader
from docx import Document

# ==========================================
# 0. CONFIGURATION
# ==========================================

load_dotenv()
model_name = os.getenv("MODEL_NAME", "azure/gpt-4.1-mini")

# Configuration de la page
st.set_page_config(page_title="Scanner Local Documents", page_icon="📂", layout="wide")

# ==========================================
# 1. FONCTIONS UTILITAIRES (Lecture Fichiers)
# ==========================================

def read_file_content(filepath):
    """Lit le contenu texte d'un fichier selon son extension."""
    ext = os.path.splitext(filepath)[1].lower()
    content = ""
    
    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        elif ext == ".docx":
            doc = Document(filepath)
            for para in doc.paragraphs:
                content += para.text + "\n"
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            return f"[Format {ext} non supporté]"
    except Exception as e:
        return f"[Erreur de lecture : {str(e)}]"
        
    return content

def scan_directory(directory_path):
    """Scanne un dossier réel pour lister les fichiers."""
    files_data = []
    
    if not os.path.isdir(directory_path):
        return []

    # On parcourt le dossier
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        # On ignore les dossiers, on ne garde que les fichiers
        if os.path.isfile(filepath):
            stats = os.stat(filepath)
            mod_time = datetime.fromtimestamp(stats.st_mtime)
            
            files_data.append({
                "name": filename,
                "path": filepath,
                "date": mod_time,
                "size": stats.st_size
            })
            
    return files_data

def estimate_tokens(text):
    """Estimation simple ou précise via tiktoken."""
    # Note: token_counter de litellm est le plus précis
    return token_counter(model=model_name, text=text)

# ==========================================
# 2. LOGIQUE MÉTIER
# ==========================================

def process_files(selected_folder):
    # 1. Patterns mis à jour
    search_patterns = {
        "RBO": r".*RBO.*",
        "PTC": r".*PTC.*",
        "BCO": r".*BCO.*",
        "BDC": r".*BDC.*"
    }
    
    # 2. Scan du disque
    all_files = scan_directory(selected_folder)
    if not all_files:
        return [], [], "Dossier vide ou introuvable."

    selected_files = []
    logs = []
    seen_files = set()

    # 3. Filtrage intelligent
    for label, regex_pattern in search_patterns.items():
        candidates = [f for f in all_files if re.search(regex_pattern, f['name'], re.IGNORECASE)]
        
        if not candidates:
            logs.append(f"⚠️ Aucun fichier trouvé pour : {label}")
            continue

        # Tri : Le plus récent en premier
        candidates.sort(key=lambda x: x['date'], reverse=True)
        winner = candidates[0]

        if winner['name'] not in seen_files:
            # C'est ici qu'on lit le contenu (seulement pour les gagnants pour gagner du temps)
            winner["content"] = read_file_content(winner["path"])
            selected_files.append(winner)
            seen_files.add(winner['name'])
            logs.append(f"✅ '{label}' -> **{winner['name']}** ({winner['date'].strftime('%d/%m/%Y')})")
        else:
             logs.append(f"ℹ️ '{label}' -> Fichier déjà sélectionné ({winner['name']})")

    return selected_files, logs, None

# ==========================================
# 3. INTERFACE
# ==========================================

st.title("📂 Scanner Automatique RBO/PTC/BCO")

# Zone de sélection du dossier
col_input, col_btn = st.columns([3, 1])
with col_input:
    # Astuce : On peut mettre une valeur par défaut pour faciliter vos tests
    default_path = os.path.join(os.getcwd(), "documents_types") 
    folder_path = st.text_input("Chemin du dossier à analyser :", value=default_path)

# Bouton d'action
start_analysis = st.button("Lancer l'analyse complète", type="primary")

if start_analysis:
    if not folder_path:
        st.error("Veuillez entrer un chemin de dossier.")
    else:
        # A. BARRE DE PROGRESSION
        progress_bar = st.progress(0, text="Initialisation...")
        
        # Etape 1 : Scan et Filtrage
        progress_bar.progress(20, text="Scan du répertoire et filtrage des dates...")
        time.sleep(0.5) # Juste pour l'effet visuel
        
        final_docs, logs, error = process_files(folder_path)
        
        if error:
            st.error(error)
        elif not final_docs:
            progress_bar.progress(100, text="Fini.")
            st.warning("Aucun fichier correspondant aux critères (RBO, PTC, BCO, BDC) n'a été trouvé.")
        else:
            # Affichage des logs de sélection
            with st.expander("Voir le détail de la sélection des fichiers", expanded=True):
                for log in logs:
                    st.markdown(f"- {log}")

            # Etape 2 : Extraction et Calculs
            progress_bar.progress(50, text="Extraction du texte et calcul des tokens...")
            
            # Construction du contexte
            full_context = ""
            total_input_tokens = 0
            
            for doc in final_docs:
                doc_text = f"\n--- DOCUMENT: {doc['name']} ---\n{doc['content']}\n"
                full_context += doc_text
                # Calcul tokens par fichier
                doc['tokens'] = estimate_tokens(doc['content'])
                total_input_tokens += doc['tokens']

            # Etape 3 : Affichage Tableau Recap
            progress_bar.progress(70, text="Préparation de la synthèse...")
            
            st.subheader("📊 Fichiers analysés")
            df = pd.DataFrame(final_docs)
            st.dataframe(
                df[['name', 'date', 'tokens']], 
                column_config={
                    "name": "Nom du fichier",
                    "date": "Date modif.",
                    "tokens": st.column_config.NumberColumn("Tokens (Coût)", format="%d")
                },
                width="stretch"
            )
            
            st.info(f"💰 Total Tokens en entrée : **{total_input_tokens}**")

            # Etape 4 : Appel LLM
            progress_bar.progress(85, text="Interrogation de l'IA (Patience)...")
            
            system_prompt = """
            Tu es un chef de projet expert en analyse de projets IT.
            Fais une synthèse structurée des documents fournis (RBO, PTC, BCO, BDC).
            Pour chaque type de document trouvé, extrais les points clés, les montants financiers et les alertes.
            Si un type de document manque, indique-le.
            """
            
            try:
                response = completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_context}
                    ]
                )
                
                ai_reply = response.choices[0].message.content
                output_tokens = response.usage.completion_tokens
                
                progress_bar.progress(100, text="Terminé !")
                
                col_res1, col_res2 = st.columns([3, 1])
                with col_res1:
                    st.subheader("🧠 Synthèse Générale")
                    st.markdown(ai_reply)
                
                with col_res2:
                    st.metric("Tokens Réponse", output_tokens)
                    st.metric("Total Session", total_input_tokens + output_tokens)

            except Exception as e:
                st.error(f"Erreur lors de l'appel IA : {e}")