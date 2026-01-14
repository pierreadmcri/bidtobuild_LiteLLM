import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Imports locaux
import config
from folder_selector import folder_selector
from utils import (
    validate_file_path,
    validate_file_size,
    safe_completion,
    safe_embedding,
    estimate_tokens,
    calculate_cost,
    format_cost,
    load_prompt,
    rate_limiter,
    ValidationError,
    FileTooLargeError,
    logger
)

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

# Configuration du niveau de log LiteLLM
os.environ['LITELLM_LOG'] = config.LITELLM_LOG_LEVEL

# Configuration Azure
for var in config.REQUIRED_ENV_VARS:
    value = os.getenv(var)
    if not value:
        st.error(f"❌ Variable d'environnement manquante : {var}")
        st.info("Vérifiez votre fichier `.env`.")
        st.stop()
    os.environ[var] = value

# --- CONSTANTES & CONFIG ---
CACHE_FILE = config.CACHE_FILE
NB_WORKERS = config.NB_WORKERS
BATCH_SIZE = config.BATCH_SIZE

model_name = config.MODEL_NAME
embedding_model_name = config.EMBEDDING_MODEL_NAME

# ==========================================
# 1. FONCTIONS UTILITAIRES
# ==========================================

def read_file_content(filepath):
    """
    Lecture robuste PDF/DOCX/TXT avec validation de taille.

    Args:
        filepath: Chemin du fichier à lire

    Returns:
        Contenu du fichier ou message d'erreur
    """
    ext = os.path.splitext(filepath)[1].lower()
    content = ""

    # Validation de la taille du fichier
    try:
        file_path = Path(filepath)
        validate_file_size(file_path)
    except FileTooLargeError as e:
        logger.warning(f"Fichier trop volumineux ignoré : {filepath}")
        return f"[Alerte : {str(e)}]"
    except Exception as e:
        logger.error(f"Erreur de validation : {e}")
        return f"[Erreur de validation : {str(e)}]"

    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
            if not content.strip():
                return "[Alerte : Aucun texte lisible extrait du PDF.]"
        elif ext == ".docx":
            doc = Document(filepath)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            logger.warning(f"Format non supporté : {ext}")
            return f"[Format {ext} non supporté]"

        logger.info(f"Fichier lu avec succès : {filepath}")
        return content

    except FileNotFoundError:
        logger.error(f"Fichier introuvable : {filepath}")
        return f"[Erreur : Fichier introuvable]"
    except PermissionError:
        logger.error(f"Permission refusée : {filepath}")
        return f"[Erreur : Permission refusée]"
    except Exception as e:
        logger.error(f"Erreur de lecture {filepath}: {str(e)}")
        return f"[Erreur lecture: {str(e)}]"

def scan_directory(directory_path):
    """
    Scan récursif des fichiers avec validation.

    Args:
        directory_path: Chemin du dossier à scanner

    Returns:
        Liste de dictionnaires contenant les métadonnées des fichiers

    Raises:
        ValidationError: Si le chemin est invalide
    """
    files_data = []
    allowed_ext = config.ALLOWED_EXTENSIONS

    # Validation du chemin
    try:
        validated_path = validate_file_path(directory_path)
    except ValidationError as e:
        logger.error(f"Chemin invalide : {e}")
        raise

    if not validated_path.is_dir():
        logger.warning(f"Le chemin n'est pas un dossier : {directory_path}")
        return []

    for root, _, files in os.walk(validated_path):
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
                except Exception as e:
                    logger.warning(f"Impossible de lire {filepath}: {e}")
                    continue

    logger.info(f"Scan terminé : {len(files_data)} fichiers trouvés")
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
    """
    Wrapper pour appel API Embedding avec gestion d'erreur et retry.

    Args:
        texts: Liste de textes à embedder

    Returns:
        Liste d'embeddings ou vecteurs zéro en cas d'échec
    """
    try:
        # Utilisation de safe_embedding qui gère le retry et rate limiting
        embeddings = safe_embedding(
            texts=texts,
            model=embedding_model_name,
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION")
        )
        return embeddings

    except ConnectionError as e:
        logger.error(f"Erreur de connexion lors de l'embedding : {e}")
        return [[0.0] * config.EMBEDDING_DIMENSION for _ in texts]
    except Exception as e:
        logger.error(f"Erreur lors de l'embedding : {e}")
        return [[0.0] * config.EMBEDDING_DIMENSION for _ in texts]

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
        rate_limiter.wait()  # Rate limiting centralisé
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

# Zone de sélection du dossier avec le nouveau composant
default_path = os.path.join(os.getcwd(), "documents_types")
folder_path = folder_selector(default_path=default_path, key="rag_folder")

st.markdown("---")

# Bouton d'action
run_btn = st.button("🚀 Lancer l'analyse RAG", type="primary", use_container_width=True)

if run_btn:
    if not folder_path:
        st.error("⚠️ Veuillez entrer un chemin de dossier.")
    else:
        # Validation du chemin avant traitement
        try:
            validate_file_path(folder_path)
        except ValidationError as e:
            st.error(f"❌ Chemin invalide : {e}")
            logger.error(f"Validation du chemin échouée : {e}")
            st.stop()
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

        # Calcul des tokens des chunks
        total_chunks_tokens = sum(estimate_tokens(c['content']) for c in chunks)

        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)

        with col_info1:
            st.metric(
                label="📚 Documents indexés",
                value=len([s for s in stats if s['État'] == '✅ Indexé']),
                help="Nombre de documents analysés"
            )

        with col_info2:
            st.metric(
                label="✂️ Segments créés",
                value=len(chunks),
                help="Nombre total de chunks générés"
            )

        with col_info3:
            st.metric(
                label="🎫 Tokens indexés",
                value=f"{total_chunks_tokens:,}",
                help="Total de tokens dans tous les segments"
            )

        # --- PHASE 2 : RECHERCHE (RAG) ---
        # Ici on n'injecte PAS de requête utilisateur explicite :
        # le LLM sera guidé uniquement par le prompt système.
        try:
            # Embedding d'une "pseudo-question" neutre pour structurer la recherche
            neutral_query = "Analyse globale de ce projet IT (contexte, périmètre, finances, risques, recommandations)."

            # Utilisation de safe_embedding avec retry automatique
            q_vec_list = safe_embedding(
                texts=[neutral_query],
                model=embedding_model_name,
                api_key=os.getenv("AZURE_API_KEY"),
                api_base=os.getenv("AZURE_API_BASE"),
                api_version=os.getenv("AZURE_API_VERSION")
            )

            q_emb = np.array(q_vec_list[0], dtype=float)

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

            # Calcul des tokens sélectionnés pour le contexte
            selected_tokens = sum(estimate_tokens(c['content']) for c in selected_chunks)

            st.markdown("---")
            st.subheader("🔍 Résultats de la recherche")

            col_search1, col_search2, col_search3 = st.columns(3)

            with col_search1:
                st.metric(
                    label="📄 Segments sélectionnés",
                    value=top_k_chunks,
                    help="Nombre de chunks les plus pertinents"
                )

            with col_search2:
                st.metric(
                    label="🎫 Tokens contexte",
                    value=f"{selected_tokens:,}",
                    help="Tokens qui seront envoyés au LLM"
                )

            with col_search3:
                context_percent = (selected_tokens / total_chunks_tokens) * 100 if total_chunks_tokens > 0 else 0
                st.metric(
                    label="📊 Utilisation",
                    value=f"{context_percent:.1f}%",
                    help="Pourcentage du contenu indexé utilisé"
                )

        except ConnectionError as e:
            st.error(f"❌ Erreur de connexion lors de la recherche vectorielle : {e}")
            logger.error(f"Erreur de connexion : {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur Recherche Vectorielle : {e}")
            logger.error(f"Erreur recherche vectorielle : {e}")
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

        # Chargement du prompt système depuis le fichier
        try:
            system_prompt = load_prompt("rag_system_prompt.txt")
        except Exception as e:
            st.error(f"❌ Impossible de charger le prompt système : {e}")
            logger.error(f"Erreur chargement prompt : {e}")
            system_prompt = "Tu es un expert en analyse de documents de projet IT."

        st.divider()
        st.subheader("📝 Rapport d'Analyse")

        with st.spinner("🧠 Rédaction du rapport en cours..."):
            try:
                # Utilisation de safe_completion avec retry automatique
                response = safe_completion(
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
                output_tokens = response.usage.completion_tokens
                input_tokens_used = response.usage.prompt_tokens

                st.markdown(ai_text)

                # Récapitulatif final des tokens et coûts
                st.markdown("---")
                st.subheader("📊 Récapitulatif de la session")

                # Calcul du coût
                cost_info = calculate_cost(input_tokens_used, output_tokens, model_name)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="📥 Tokens entrée",
                        value=f"{input_tokens_used:,}",
                        help="Tokens envoyés (prompt + contexte)"
                    )

                with col2:
                    st.metric(
                        label="📤 Tokens sortie",
                        value=f"{output_tokens:,}",
                        help="Tokens générés par le LLM"
                    )

                with col3:
                    st.metric(
                        label="🎫 Total tokens",
                        value=f"{cost_info['total_tokens']:,}",
                        help="Total de la génération"
                    )

                with col4:
                    st.metric(
                        label="💰 Coût génération",
                        value=format_cost(cost_info['total_cost']),
                        help=f"Modèle: {model_name}"
                    )

                # Calcul du coût total (embeddings + génération)
                embedding_cost_info = calculate_cost(total_chunks_tokens, 0, embedding_model_name)
                total_session_cost = embedding_cost_info['total_cost'] + cost_info['total_cost']

                # Détail des coûts avec embeddings
                with st.expander("💵 Détail complet des coûts"):
                    st.write("**Coûts de génération (LLM)**")
                    st.write(f"- Modèle: `{model_name}`")
                    st.write(f"- Coût entrée: {format_cost(cost_info['input_cost'])} ({input_tokens_used:,} tokens)")
                    st.write(f"- Coût sortie: {format_cost(cost_info['output_cost'])} ({output_tokens:,} tokens)")
                    st.write(f"- Sous-total LLM: {format_cost(cost_info['total_cost'])}")

                    st.write("")
                    st.write("**Coûts d'indexation (Embeddings)**")
                    st.write(f"- Modèle: `{embedding_model_name}`")
                    st.write(f"- Tokens indexés: {total_chunks_tokens:,}")
                    st.write(f"- Coût embeddings: {format_cost(embedding_cost_info['total_cost'])}")

                    st.write("")
                    st.write(f"**💰 Coût total de la session: {format_cost(total_session_cost)}**")

                    st.info("💡 Les embeddings sont mis en cache. Les prochaines exécutions ne paieront que la génération LLM !")

            except ConnectionError as e:
                st.error("❌ Impossible de se connecter à l'API. Vérifiez votre connexion internet.")
                logger.error(f"Erreur de connexion API : {e}")
                st.code(str(e))
            except PermissionError as e:
                st.error("❌ Erreur d'authentification. Vérifiez votre clé API Azure.")
                logger.error(f"Erreur d'authentification : {e}")
                st.code(str(e))
            except Exception as e:
                st.error(f"❌ Erreur LLM : {e}")
                logger.error(f"Erreur inattendue lors de l'appel LLM : {e}")
                st.code(str(e))

        # --- SOURCES ---
        st.markdown("---")
        with st.expander("🔎 Consulter les extraits sources utilisés"):
            for i, c in enumerate(selected_chunks):
                st.markdown(f"**Source {i+1} : {c['doc_type']}** - {c['doc_name']}")
                st.caption(c["content"][:600] + " [...]")
                st.divider()
