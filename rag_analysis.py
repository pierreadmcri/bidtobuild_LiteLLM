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

os.environ["LITELLM_LOG"] = "ERROR"  # Silence les logs sauf erreurs

# ==========================================
# 1. FONCTIONS UTILITAIRES
# ==========================================

def estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (approx)."""
    try:
        return max(len(text) // 4, 1)
    except Exception:
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
                if text:
                    content += text + "\n"
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
                except Exception:
                    continue
    return files_data

# ==========================================
# 2. LOGIQUE MÉTIER (Smart Chunking + Overlap)
# ==========================================

def smart_chunk_document(doc, max_tokens: int, overlap_tokens: int):
    """
    Découpe par paragraphes avec overlap pour préserver le sens.
    - max_tokens : taille cible d'un segment (approx tokens)
    - overlap_tokens : quantité de contexte recopiée du segment précédent
    """
    content = doc["content"]
    if not content or len(content) < 50:
        return []

    # Découpe simple en pseudo-paragraphes
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # Sécurité : si un paragraphe est monstrueux, on tronque
        if para_tokens > max_tokens:
            max_chars = max_tokens * 4
            para = para[:max_chars]
            para_tokens = estimate_tokens(para)

        # Si on dépasserait la limite avec ce paragraphe -> on ferme le chunk
        if current_tokens + para_tokens > max_tokens and current_chunk:
            full_text = "\n\n".join(current_chunk)
            chunks.append({
                "doc_name": doc["name"],
                "doc_type": doc.get("doc_type", "UNKNOWN"),
                "date": doc["date"],
                "content": full_text
            })

            # Construction du nouvel espace avec overlap
            overlap = []
            overlap_used = 0
            for prev_para in reversed(current_chunk):
                t = estimate_tokens(prev_para)
                if overlap_used + t > overlap_tokens:
                    break
                overlap.insert(0, prev_para)
                overlap_used += t

            current_chunk = overlap + [para]
            current_tokens = overlap_used + para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Dernier segment
    if current_chunk:
        chunks.append({
            "doc_name": doc["name"],
            "doc_type": doc.get("doc_type", "UNKNOWN"),
            "date": doc["date"],
            "content": "\n\n".join(current_chunk)
        })

    return chunks

def get_embeddings_batch(texts):
    """Wrapper pour appel API Embedding avec gestion d'erreur."""
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
        if hasattr(response, "data"):
            return [d["embedding"] if isinstance(d, dict) else d.embedding for d in response.data]
        elif isinstance(response, dict) and "data" in response:
            return [d["embedding"] for d in response["data"]]
        return [[0.0] * 1536 for _ in texts]
    except Exception:
        return [[0.0] * 1536 for _ in texts]

# ==========================================
# 3. MMR (Maximal Marginal Relevance)
# ==========================================

def mmr(embeddings: np.ndarray, query_emb: np.ndarray, k: int, lambda_mult: float) -> list:
    """
    MMR sur un sous-ensemble d'embeddings.
    embeddings: (N, d)
    query_emb: (d,)
    k: nombre de documents à retourner
    lambda_mult: trade-off pertinence/diversité
    Retourne une liste d'indices (0..N-1) dans embeddings.
    """
    if embeddings.shape[0] == 0:
        return []

    # Similarité query-doc
    norms_docs = np.linalg.norm(embeddings, axis=1)
    norms_docs[norms_docs == 0] = 1.0
    norm_q = np.linalg.norm(query_emb)
    if norm_q == 0:
        norm_q = 1.0

    sim_query = (embeddings @ query_emb) / (norms_docs * norm_q)

    # Similarité doc-doc
    dot_docs = embeddings @ embeddings.T
    norms_outer = np.outer(norms_docs, norms_docs)
    norms_outer[norms_outer == 0] = 1.0
    sim_docs = dot_docs / norms_outer

    selected = []
    candidates = list(range(embeddings.shape[0]))

    for _ in range(min(k, len(candidates))):
        mmr_scores = []
        for idx in candidates:
            if not selected:
                diversity = 0.0
            else:
                diversity = max(sim_docs[idx, s] for s in selected)
            score = lambda_mult * sim_query[idx] - (1.0 - lambda_mult) * diversity
            mmr_scores.append((score, idx))
        mmr_scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_idx = mmr_scores[0]
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected

# ==========================================
# 4. CORE : CHARGEMENT OPTIMISÉ (Threaded + Persistent)
# ==========================================

@st.cache_resource(show_spinner=False)
def load_and_process_data_optimized(folder_path: str, max_chunk_tokens: int, overlap_tokens: int):
    """
    Charge les documents, les découpe en segments, calcule les embeddings.
    Utilise un cache disque + cache Streamlit.
    """
    logs = []
    stats = []
    
    # --- A. VERIFICATION CACHE DISQUE ---
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                saved_data = pickle.load(f)
            if (
                saved_data.get("folder") == folder_path 
                and saved_data.get("max_chunk_tokens") == max_chunk_tokens
                and saved_data.get("overlap_tokens") == overlap_tokens
            ):
                logs.append("⚡ Données chargées depuis le disque (Cache local).")
                return (
                    saved_data["chunks"],
                    saved_data["embeddings"],
                    logs,
                    None,
                    saved_data["stats"],
                )
            else:
                logs.append("ℹ️ Paramètres chunking différents, recalcul nécessaire.")
        except Exception:
            logs.append("⚠️ Cache disque corrompu, nouveau calcul nécessaire.")

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
        candidates = [f for f in all_files if re.search(pattern, f["name"], re.IGNORECASE)]
        if candidates:
            candidates.sort(key=lambda x: x["date"], reverse=True)
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
        doc_chunks = smart_chunk_document(doc, max_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
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
    
    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batches.append((i, texts[i:i+BATCH_SIZE]))

    def process_batch_worker(args):
        idx, batch_txt = args
        time.sleep(0.05)  # Petit délai pour ménager l'API
        emb = get_embeddings_batch(batch_txt)
        return idx, emb

    results = []
    progress_bar = st.progress(0, text="Calcul vectoriel en parallèle...")
    
    with ThreadPoolExecutor(max_workers=NB_WORKERS) as executor:
        futures = list(executor.map(process_batch_worker, batches))
        for i, res in enumerate(futures):
            results.append(res)
            progress_bar.progress(
                min((i + 1) / len(batches), 1.0),
                text=f"Vectorisation : Batch {i + 1}/{len(batches)}"
            )

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
            "max_chunk_tokens": max_chunk_tokens,
            "overlap_tokens": overlap_tokens,
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data_to_save, f)
        logs.append("💾 Sauvegarde locale créée (chargement instantané au prochain coup).")
    except Exception as e:
        logs.append(f"⚠️ Echec sauvegarde cache: {e}")

    return all_chunks, np_embeddings, logs, None, stats

# ==========================================
# 5. INTERFACE STREAMLIT
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Paramètres")

    if st.button("🗑️ Vider Cache & Recharger", type="secondary"):
        st.cache_resource.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.rerun()

    st.markdown("### ⚙️ RAG - Chunking")
    max_chunk_tokens = st.slider(
        "Taille des segments (tokens approx.)",
        min_value=200,
        max_value=1500,
        value=600,
        step=50,
    )
    overlap_tokens = st.slider(
        "Overlap entre segments (tokens)",
        min_value=0,
        max_value=400,
        value=120,
        step=20,
    )

    st.markdown("### 🔎 RAG - Retrieval")
    top_k_chunks = st.slider(
        "Nombre de segments utilisés (Top-K)",
        min_value=3,
        max_value=20,
        value=6,
        step=1,
    )
    sim_threshold = st.slider(
        "Seuil de similarité minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
    )

    use_mmr = st.checkbox("Activer MMR (diversification)", value=True)
    lambda_mmr = st.slider(
        "MMR λ (pertinence vs diversité)",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.05,
    )

    st.info(
        f"""
**Mode Turbo** 🚀  
- Taille segment ≈ {max_chunk_tokens} tokens  
- Overlap ≈ {overlap_tokens} tokens  
- Top-K = {top_k_chunks}  
- Seuil sim. = {sim_threshold:.2f}  
- MMR = {"ON" if use_mmr else "OFF"} (λ={lambda_mmr:.2f})
"""
    )

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
    run_btn = st.button("🚀 Analyser", type="primary", use_container_width=True)

if run_btn:
    if not folder_path or not os.path.exists(folder_path):
        st.error("⚠️ Chemin invalide.")
    else:
        # --- PHASE 1 : CHARGEMENT & INDEXATION ---
        with st.status("🔍 Analyse des documents...", expanded=True) as status:
            chunks, embeddings, logs, error, stats = load_and_process_data_optimized(
                folder_path,
                max_chunk_tokens=max_chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
            
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
                st.dataframe(pd.DataFrame(stats), use_container_width=True)

        # --- PHASE 2 : RECHERCHE (RAG) ---
        # Ici on n'injecte PAS de requête utilisateur explicite :
        # le LLM sera guidé uniquement par le prompt système.
        try:
            # Embedding d'une "pseudo-question" neutre pour structurer la recherche
            # (optionnel, mais tu peux garder un wording générique)
            neutral_query = "Analyse globale de ce projet IT (contexte, périmètre, finances, risques, recommandations)."

            q_resp = embedding(
                model=embedding_model_name,
                input=neutral_query,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION")
            )
            
            # Extraction sécurisée
            if hasattr(q_resp, "data"):
                data_item = q_resp.data[0]
            else:
                data_item = q_resp["data"][0]
                
            if hasattr(data_item, "embedding"):
                q_vec = data_item.embedding
            else:
                q_vec = data_item["embedding"]
                
            q_emb = np.array(q_vec, dtype=float)

            # Calcul Similarité Cosinus
            norm_q = np.linalg.norm(q_emb)
            if norm_q == 0:
                norm_q = 1.0

            norm_docs = np.linalg.norm(embeddings, axis=1)
            norm_docs[norm_docs == 0] = 1.0
            
            similarities = (embeddings @ q_emb) / (norm_docs * norm_q)

            # Filtrage par seuil
            candidate_indices = np.where(similarities >= sim_threshold)[0]

            if len(candidate_indices) == 0:
                st.warning(
                    "Aucun segment ne dépasse le seuil de similarité configuré. "
                    "Utilisation des meilleurs segments disponibles malgré tout."
                )
                # on prend un pool un peu plus large pour MMR ou tri simple
                candidate_indices = np.argsort(-similarities)[: max(top_k_chunks * 3, top_k_chunks)]
            
            # Sélection finale (MMR ou simple tri)
            if use_mmr:
                sub_emb = embeddings[candidate_indices]
                mmr_indices_sub = mmr(
                    embeddings=sub_emb,
                    query_emb=q_emb,
                    k=top_k_chunks,
                    lambda_mult=lambda_mmr,
                )
                top_indices = [int(candidate_indices[i]) for i in mmr_indices_sub]
            else:
                sorted_candidates = candidate_indices[np.argsort(-similarities[candidate_indices])]
                top_indices = sorted_candidates[:top_k_chunks].tolist()

            selected_chunks = [chunks[i] for i in top_indices]

        except Exception as e:
            st.error(f"Erreur Recherche Vectorielle : {e}")
            st.stop()

        # --- PHASE 3 : GENERATION REPONSE ---
        # Contexte mieux structuré
        context_str = ""
        for i, c in enumerate(selected_chunks):
            date_str = c["date"].strftime("%Y-%m-%d") if isinstance(c["date"], datetime) else str(c["date"])
            context_str += (
                f"\n### SEGMENT {i+1}\n"
                f"[SOURCE: {c['doc_type']} - {c['doc_name']} - {date_str}]\n"
                f"{c['content']}\n"
            )

        system_prompt = """
Tu es un directeur de projet expert en delivery IT.

Tu reçois des extraits de documents de projet (RBO, PTC, BCO, BDC) au format suivant :
### SEGMENT i
[SOURCE: TYPE - NOM_FICHIER - DATE]
CONTENU...

RÈGLES :
- Réponds UNIQUEMENT en t'appuyant sur les segments fournis.
- Quand tu cites un chiffre ou un fait, indique la source au format [RBO], [PTC], [BCO], [BDC].
- Si une information n'apparaît pas clairement dans les segments, répond "Non spécifié".
- Si un type de document n'est pas présent (par ex. pas de BDC), précise-le explicitement.

Produit une synthèse structurée en 5 parties :
1. Contexte & Objectifs
2. Périmètre Technique & Fonctionnel
3. Synthèse Financière (Budget, TJM, RAF, BDC)
4. Risques (Planning, Budget, Tech)
5. Recommandations

Style : professionnel, clair, synthétique, en français.
"""

        st.divider()
        st.subheader("📝 Rapport d'Analyse")
        
        with st.spinner("🧠 Rédaction du rapport en cours..."):
            try:
                response = completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Voici des extraits de documents de projet. Utilise uniquement ces informations pour produire ton analyse.\n\n{context_str}"
                        },
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
                st.caption(c["content"][:600] + " [...]")
                st.divider()
