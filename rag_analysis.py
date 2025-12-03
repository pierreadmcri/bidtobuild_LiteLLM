import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import time
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from litellm import completion, token_counter, embedding

# Imports pour lecture fichiers
from pypdf import PdfReader
from docx import Document

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================

st.set_page_config(
    page_title="RAG Analyse Pro - RBO/PTC/BCO/BDC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# --- CONSTANTES & CONFIG ---
CACHE_FILE = "vector_store_cache.pkl"   # Fichier de sauvegarde locale
NB_WORKERS = 4                          # Nombre d'appels simultanés vers Azure (Attention Rate Limit)
BATCH_SIZE = 10                         # Taille des lots d'embeddings
MAX_CHUNK_TOKENS = 1500                 # Taille idéale par morceau
TOP_K_CHUNKS = 10                       # Nombre d'extraits utilisés pour l'analyse

# --- CONFIGURATION AZURE ---
model_name = os.getenv("MODEL_NAME", "azure/gpt-4o-mini")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "azure/text-embedding-3-small")

required_env_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    st.error(f"❌ Configuration manquante : {', '.join(missing_vars)}")
    st.info("Vérifiez votre fichier `.env`.")
    st.stop()

# Configuration LiteLLM
for var in required_env_vars:
    os.environ[var] = os.getenv(var)

os.environ['LITELLM_LOG'] = 'ERROR'  # Silence les logs sauf erreurs

# ==========================================
# 1. FONCTIONS UTILITAIRES
# ==========================================

def estimate_tokens(text):
    """Estimation rapide."""
    try:
        return len(text) // 4  # Approximation rapide pour ne pas ralentir
    except:
        return 0

def read_file_content(filepath):
    """Lecture robuste PDF/DOCX/TXT."""
    ext = os.path.splitext(filepath)[1].lower()
    content = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            for page in reader.pages:
                text = page.extract_text()
                if text: content += text + "\n"
        elif ext == ".docx":
            doc = Document(filepath)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        return content
    except Exception as e:
        return f"[Erreur lecture: {str(e)}]"

def scan_directory(directory_path):
    """Scan récursif des fichiers."""
    files_data = []
    allowed_ext = {".pdf", ".docx", ".txt"}
    
    if not os.path.isdir(directory_path):
        return []

    for root, _, files in os.walk(directory_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in allowed_ext:
                filepath = os.path.join(root, filename)
                try:
                    stats = os.stat(filepath)
                    files_data.append({
                        "name": filename,
                        "path": filepath,
                        "date": datetime.fromtimestamp(stats.st_mtime),
                        "size": stats.st_size
                    })
                except:
                    continue
    return files_data

# ==========================================
# 2. LOGIQUE MÉTIER (Smart Chunking)
# ==========================================

def smart_chunk_document(doc, max_tokens=MAX_CHUNK_TOKENS):
    """Découpe par paragraphes pour préserver le sens."""
    content = doc["content"]
    if not content or len(content) < 50:
        return []

    paragraphs = content.split('\n\n') # Séparateur paragraphe
    chunks = []
    current_chunk_text = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para: continue
        
        para_tokens = estimate_tokens(para)

        # Si ajouter ce paragraphe dépasse la limite, on enregistre le chunk actuel
        if current_tokens + para_tokens > max_tokens:
            if current_chunk_text:
                full_text = "\n\n".join(current_chunk_text)
                chunks.append({
                    "doc_name": doc["name"],
                    "doc_type": doc.get("doc_type", "UNKNOWN"),
                    "date": doc["date"],
                    "content": full_text
                })
            current_chunk_text = [para]
            current_tokens = para_tokens
        else:
            current_chunk_text.append(para)
            current_tokens += para_tokens

    # Dernier morceau
    if current_chunk_text:
        chunks.append({
            "doc_name": doc["name"],
            "doc_type": doc.get("doc_type", "UNKNOWN"),
            "date": doc["date"],
            "content": "\n\n".join(current_chunk_text)
        })

    return chunks

def get_embeddings_batch(texts):
    """Wrapper pour appel API avec gestion d'erreur."""
    # Troncature de sécurité pour l'API (limite technique ~8191 tokens)
    safe_texts = [t[:30000] for t in texts] 
    
    try:
        response = embedding(
            model=embedding_model_name,
            input=safe_texts,
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION")
        )
        # Extraction robuste (dict ou object)
        if hasattr(response, 'data'):
            return [d['embedding'] if isinstance(d, dict) else d.embedding for d in response.data]
        elif isinstance(response, dict) and 'data' in response:
            return [d['embedding'] for d in response['data']]
        return [[0.0]*1536 for _ in texts]
    except Exception:
        return [[0.0]*1536 for _ in texts]

# ==========================================
# 3. CORE : CHARGEMENT OPTIMISÉ (Threaded + Persistent)
# ==========================================

@st.cache_resource(show_spinner=False)
def load_and_process_data_optimized(folder_path):
    logs = []
    stats = []
    
    # --- A. VERIFICATION CACHE DISQUE ---
    # Si le fichier .pkl existe, on charge tout instantanément
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                saved_data = pickle.load(f)
            # Vérification basique que c'est le même dossier
            if saved_data.get("folder") == folder_path:
                return saved_data["chunks"], saved_data["embeddings"], ["⚡ Données chargées depuis le disque (Cache local)."], None, saved_data["stats"]
        except Exception as e:
            logs.append(f"⚠️ Cache disque corrompu, nouveau calcul nécessaire.")

    # --- B. SCAN ET LECTURE ---
    search_patterns = {
        "RBO": r".*RBO.*", "PTC": r".*PTC.*", 
        "BCO": r".*BCO.*", "BDC": r".*BDC.*"
    }
    
    all_files = scan_directory(folder_path)
    if not all_files:
        return None, None, logs, "Dossier vide ou introuvable", []

    selected_docs = []
    for label, pattern in search_patterns.items():
        candidates = [f for f in all_files if re.search(pattern, f['name'], re.IGNORECASE)]
        if candidates:
            # On prend le plus récent
            candidates.sort(key=lambda x: x['date'], reverse=True)
            winner = candidates[0]
            winner["doc_type"] = label
            winner["content"] = read_file_content(winner["path"])
            selected_docs.append(winner)
            logs.append(f"✅ **{label}** : {winner['name']} ({winner['date'].strftime('%d/%m')})")
        else:
            logs.append(f"⚠️ **{label}** : Non trouvé")

    if not selected_docs:
        return None, None, logs, "Aucun document RBO/PTC/BCO/BDC détecté.", []

    # --- C. CHUNKING ---
    all_chunks = []
    for doc in selected_docs:
        doc_chunks = smart_chunk_document(doc)
        all_chunks.extend(doc_chunks)
        stats.append({
            "Type": doc["doc_type"],
            "Fichier": doc["name"],
            "Nb Segments": len(doc_chunks),
            "État": "✅ Indexé"
        })
    
    logs.append(f"✂️ Total segments générés : {len(all_chunks)}")

    if not all_chunks:
        return None, None, logs, "Documents vides.", []

    # --- D. EMBEDDINGS PARALLÈLES (MULTI-THREADING) ---
    texts = [c["content"] for c in all_chunks]
    
    # Préparation des lots
    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batches.append((i, texts[i:i+BATCH_SIZE]))

    # Fonction Worker pour le thread
    def process_batch_worker(args):
        idx, batch_txt = args
        time.sleep(0.05) # Petit délai pour ménager l'API
        emb = get_embeddings_batch(batch_txt)
        return idx, emb

    # Exécution parallèle
    results = []
    progress_bar = st.progress(0, text="Calcul vectoriel en parallèle...")
    
    with ThreadPoolExecutor(max_workers=NB_WORKERS) as executor:
        futures = list(executor.map(process_batch_worker, batches))
        # On collecte les résultats
        for i, res in enumerate(futures):
            results.append(res)
            progress_bar.progress(min((i+1)/len(batches), 1.0), text=f"Vectorisation : Batch {i+1}/{len(batches)}")

    progress_bar.empty()

    # Reconstitution ordonnée
    results.sort(key=lambda x: x[0])
    final_embeddings = []
    for _, emb_batch in results:
        final_embeddings.extend(emb_batch)

    np_embeddings = np.array(final_embeddings, dtype=float)

    # --- E. SAUVEGARDE SUR DISQUE ---
    try:
        data_to_save = {
            "folder": folder_path,
            "chunks": all_chunks,
            "embeddings": np_embeddings,
            "stats": stats,
            "folder": folder_path
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data_to_save, f)
        logs.append("💾 Sauvegarde locale créée (chargement instantané au prochain coup).")
    except Exception as e:
        logs.append(f"⚠️ Echec sauvegarde cache: {e}")

    return all_chunks, np_embeddings, logs, None, stats

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    if st.button("🗑️ Vider Cache & Recharger", type="secondary"):
        st.cache_resource.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.rerun()
    
    st.info("""
    **Mode Turbo activé** 🚀
    - Parallélisme : 4 threads
    - Persistance : Oui (Disque)
    - Limite documents : Aucune
    """)

# --- MAIN ---
st.title("📂 Analyseur de Projets IT (RAG)")
st.markdown("Analyse automatique des documents **RBO, PTC, BCO, BDC**.")

col1, col2 = st.columns([3, 1])
with col1:
    default_path = os.path.join(os.getcwd(), "documents_types")
    folder_path = st.text_input("Dossier à analyser :", value=default_path)

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Analyser", type="primary", width="stretch")

if run_btn:
    if not folder_path or not os.path.exists(folder_path):
        st.error("⚠️ Chemin invalide.")
    else:
        # --- PHASE 1 : CHARGEMENT ---
        with st.status("🔍 Analyse des documents...", expanded=True) as status:
            chunks, embeddings, logs, error, stats = load_and_process_data_optimized(folder_path)
            
            for log in logs:
                st.markdown(log)
                
            if error:
                status.update(label="❌ Erreur", state="error")
                st.error(error)
                st.stop()
            else:
                status.update(label="✅ Indexation terminée !", state="complete", expanded=False)

        # Affichage Stats
        if stats:
            with st.expander("📊 Voir les détails de l'indexation"):
                st.dataframe(pd.DataFrame(stats), width="stretch")

        # --- PHASE 2 : RECHERCHE (RAG) ---
        query = """
        Analyse globale du projet :
        1. Contexte & Objectifs
        2. Périmètre Technique & Fonctionnel
        3. Synthèse Financière (Budget, TJM, RAF, BDC)
        4. Risques (Planning, Budget, Tech)
        5. Recommandations
        """

        try:
            # Embedding Requête (Correction Dict/Object incluse)
            q_resp = embedding(
                model=embedding_model_name,
                input=query,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION")
            )
            
            # Extraction sécurisée
            if hasattr(q_resp, 'data'):
                data_item = q_resp.data[0]
            else:
                data_item = q_resp['data'][0]
                
            if hasattr(data_item, 'embedding'):
                q_vec = data_item.embedding
            else:
                q_vec = data_item['embedding']
                
            q_emb = np.array(q_vec)

            # Calcul Similarité
            norm_q = np.linalg.norm(q_emb)
            norm_docs = np.linalg.norm(embeddings, axis=1)
            norm_docs[norm_docs == 0] = 1 # Div/0 protection
            
            similarities = (embeddings @ q_emb) / (norm_docs * norm_q)
            
            top_indices = np.argsort(-similarities)[:TOP_K_CHUNKS]
            selected_chunks = [chunks[i] for i in top_indices]

        except Exception as e:
            st.error(f"Erreur Recherche Vectorielle : {e}")
            st.stop()

        # --- PHASE 3 : GENERATION REPONSE ---
        context_str = ""
        for c in selected_chunks:
            context_str += f"\n--- DOC: {c['doc_type']} ({c['doc_name']}) ---\n{c['content']}\n"

        system_prompt = """Tu es un directeur de projet expert.
        Analyse les documents fournis pour produire une synthèse structurée.
        Cite tes sources (ex: [RBO], [BDC]) quand tu donnes un chiffre ou un fait.
        Si une info manque, dis "Non spécifié".
        Sois professionnel, clair et concis."""

        st.divider()
        st.subheader("📝 Rapport d'Analyse")
        
        with st.spinner("🧠 Rédaction du rapport en cours..."):
            try:
                response = completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"CONTEXTE:\n{context_str}\n\nREQUETE: {query}"}
                    ],
                    api_key=os.getenv("AZURE_API_KEY"),
                    api_base=os.getenv("AZURE_API_BASE"),
                    api_version=os.getenv("AZURE_API_VERSION")
                )
                
                ai_text = response.choices[0].message.content
                st.markdown(ai_text)
                
            except Exception as e:
                st.error(f"Erreur LLM : {e}")

        # --- SOURCES ---
        st.markdown("---")
        with st.expander("🔎 Consulter les extraits sources utilisés"):
            for i, c in enumerate(selected_chunks):
                st.markdown(f"**Source {i+1} : {c['doc_type']}** - {c['doc_name']}")
                st.caption(c['content'][:600] + " [...]")
                st.divider()