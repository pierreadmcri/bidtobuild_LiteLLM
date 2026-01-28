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
from utils import (
    validate_file_path, validate_file_size, safe_completion, safe_embedding,
    estimate_tokens, calculate_cost, format_cost, load_prompt,
    rate_limiter, ValidationError, FileTooLargeError, logger,
    extract_text_from_image, extract_text_from_excel
)

# Imports lecture
from pypdf import PdfReader
from docx import Document
import fitz  # PyMuPDF pour extraction d'images des PDFs

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================

st.set_page_config(
    page_title="RAG Analyse Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Log & Env
os.environ['LITELLM_LOG'] = config.LITELLM_LOG_LEVEL
for var in config.REQUIRED_ENV_VARS:
    value = os.getenv(var)
    if not value:
        st.error(f"❌ Variable d'environnement manquante : {var}")
        st.info("Vérifiez votre fichier `.env`.")
        st.stop()
    os.environ[var] = value

# Constantes
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
    Lecture robuste PDF/DOCX/TXT/EXCEL/IMAGES avec validation de taille.
    Pour les PDFs, extrait aussi les images et utilise OCR (Tesseract) pour extraire le texte.

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
        # === IMAGES STANDALONE ===
        if ext in [".jpg", ".jpeg", ".png"]:
            logger.info(f"Extraction de texte de l'image : {filepath}")
            text = extract_text_from_image(filepath)
            if text and not text.startswith("["):
                content = f"[IMAGE: {os.path.basename(filepath)}]\n{text}\n"
            else:
                content = f"[IMAGE: {os.path.basename(filepath)}]\n{text}\n"

        # === PDFs AVEC EXTRACTION D'IMAGES ===
        elif ext == ".pdf":
            # Extraction du texte avec pypdf
            reader = PdfReader(filepath)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"

            # Extraction des images avec PyMuPDF
            if config.ENABLE_IMAGE_OCR:
                try:
                    doc = fitz.open(filepath)
                    image_count = 0

                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        image_list = page.get_images(full=True)

                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]

                                # Sauvegarder temporairement l'image dans le dossier cache
                                temp_image_path = os.path.join(config.CACHE_DIR, f"temp_pdf_image_{page_num}_{img_index}.{image_ext}")
                                with open(temp_image_path, "wb") as img_file:
                                    img_file.write(image_bytes)

                                # Extraire le texte de l'image
                                text = extract_text_from_image(temp_image_path)

                                # Ajouter le texte extrait seulement s'il y en a
                                if text and not text.startswith("[") and len(text.strip()) > 0:
                                    content += f"\n[IMAGE PAGE {page_num + 1}]\n{text}\n"
                                    image_count += 1

                                # Nettoyer le fichier temporaire
                                os.remove(temp_image_path)

                            except Exception as img_err:
                                logger.warning(f"Erreur extraction image PDF page {page_num}: {img_err}")
                                continue

                    doc.close()
                    if image_count > 0:
                        logger.info(f"{image_count} image(s) extraite(s) et texte extrait par OCR du PDF")

                except Exception as pdf_img_err:
                    logger.warning(f"Erreur lors de l'extraction d'images du PDF : {pdf_img_err}")

            if not content.strip():
                return "[Alerte : Aucun texte lisible extrait du PDF.]"

        # === EXCEL ===
        elif ext in [".xlsx", ".xlsm"]:
            logger.info(f"Extraction Excel : {filepath}")
            content = extract_text_from_excel(filepath)

        # === DOCX ===
        elif ext == ".docx":
            doc = Document(filepath)
            content = "\n".join([para.text for para in doc.paragraphs])

        # === TXT ===
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
            model=embedding_model_name
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
        "RPO": r".*RPO.*", "PTC": r".*PTC.*",
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
        return None, None, logs, "Aucun document RPO/PTC/BCO/BDC détecté.", []

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

# =======================
# 5. INTERFACE STREAMLIT 
# =======================

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .stMetric {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d4654;
    }
    .stMetric label {
        color: #b8bcc5 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #a0a6b0 !important;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: bold;
    }
    h1 { color: #2c3e50; }
    h2, h3 { color: #34495e; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR OPTIMISÉE ---
with st.sidebar:
    st.title("🎛️ Contrôle")

    st.info("Ce module analyse vos documents RPO, PTC, BCO et BDC pour générer une synthèse structurée.")

    # Section Reset bien visible
    if st.button("🗑️ Vider le Cache", type="secondary", help="Force le rechargement complet des fichiers"):
        st.cache_resource.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.toast("Cache vidé avec succès !", icon="🗑️")
        st.rerun()

    st.markdown("---")

    # Masquer la complexité technique
    with st.expander("🔧 Configuration Avancée", expanded=False):
        st.caption("Paramètres du découpage (Chunking)")
        max_chunk_tokens = st.slider("Taille segments", 200, 1500, 600, 50)
        overlap_tokens = st.slider("Overlap", 0, 400, 120, 20)

        st.caption("Paramètres de recherche (Retrieval)")
        top_k_chunks = st.slider("Top-K segments", 3, 20, 6)
        sim_threshold = st.slider("Seuil similarité", 0.0, 1.0, 0.15, 0.01)

        use_mmr = st.checkbox("Activer MMR (Diversité)", value=True)
        if use_mmr:
            lambda_mmr = st.slider("MMR λ", 0.1, 0.9, 0.7)
        else:
            lambda_mmr = 0.7 # Valeur par défaut si désactivé

# --- MAIN HEADER ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# ⚡")
with col_title:
    st.title("Analyseur de Projets IT")
    st.markdown("RAG Intelligent • RPO / PTC / BCO / BDC")

st.markdown("---")

# --- ZONE DE SÉLECTION DU DOSSIER ---
col_input, col_btn = st.columns([3, 1])
with col_input:
    default_path = os.path.join(os.getcwd(), "documents_types")
    folder_path = st.text_input(
        "📂 Chemin du dossier à analyser",
        value=default_path,
        placeholder="/chemin/absolu/vers/vos/documents"
    )
with col_btn:
    st.write("") # Spacer pour aligner le bouton
    st.write("")
    run_btn = st.button("🚀 Lancer l'analyse", type="primary", width='stretch')

# --- LOGIQUE D'EXÉCUTION ---
if run_btn:
    if not folder_path:
        st.error("⚠️ Veuillez entrer un chemin de dossier valide.")
        st.stop()

    # 1. INDEXATION (Status Container)
    with st.status("🔍 Analyse et indexation des documents...", expanded=True) as status:
        try:
            validate_file_path(folder_path)
            chunks, embeddings, logs, error, stats = load_and_process_data_optimized(
                folder_path, max_chunk_tokens, overlap_tokens
            )

            # Affichage des logs importants en temps réel
            for log in logs:
                st.text(log.replace("**", "").replace("✅", "  >").replace("⚠️", "  !"))

            if error:
                status.update(label="❌ Erreur critique", state="error")
                st.error(error)
                st.stop()
            else:
                status.update(label="✅ Documents indexés et prêts !", state="complete", expanded=False)

        except ValidationError as e:
            status.update(label="❌ Chemin invalide", state="error")
            st.error(f"Erreur de validation : {e}")
            st.stop()
        except Exception as e:
            status.update(label="❌ Erreur technique", state="error")
            st.error(f"Erreur : {e}")
            st.stop()

    # 2. RECHERCHE VECTORIELLE
    try:
        # Embedding query neutre
        neutral_query = """
                        Réunion de Lancement Interne projet: contexte et périmètre BUILD, 
                        charges BUILD et RUN détaillées en jours-homme avec CCJM, budget et marge brute, 
                        macro-planning avec jalons et dates, équipe Orange Business et rôles, 
                        risques projet et actions, méthodologie et livrables, organisation transition RUN, 
                        prérequis et fournitures client, validation contractuelle intégration CRM
                        """

        with st.spinner("🧠 Recherche des passages pertinents..."):
            q_vec_list = safe_embedding(
                texts=[neutral_query],
                model=embedding_model_name
            )
            q_emb = np.array(q_vec_list[0], dtype=float)

            # Similarité Cosinus
            norm_q = np.linalg.norm(q_emb) or 1.0
            norm_docs = np.linalg.norm(embeddings, axis=1)
            norm_docs[norm_docs == 0] = 1.0
            similarities = (embeddings @ q_emb) / (norm_docs * norm_q)

            candidate_indices = np.where(similarities >= sim_threshold)[0]
            if len(candidate_indices) == 0:
                st.warning("⚠️ Seuil de pertinence non atteint. Utilisation des meilleurs segments disponibles.")
                candidate_indices = np.argsort(-similarities)[: max(top_k_chunks * 3, top_k_chunks)]

            # Sélection (MMR ou Standard)
            if use_mmr:
                sub_emb = embeddings[candidate_indices]
                mmr_indices_sub = mmr(sub_emb, q_emb, top_k_chunks, lambda_mmr)
                top_indices = [int(candidate_indices[i]) for i in mmr_indices_sub]
            else:
                sorted_candidates = candidate_indices[np.argsort(-similarities[candidate_indices])]
                top_indices = sorted_candidates[:top_k_chunks].tolist()

            selected_chunks = [chunks[i] for i in top_indices]

            # Prep context
            context_str = ""
            for i, c in enumerate(selected_chunks):
                date_str = c["date"].strftime("%Y-%m-%d") if isinstance(c["date"], datetime) else str(c["date"])
                context_str += f"\n### SEGMENT {i+1}\n[SOURCE: {c['doc_type']} - {c['doc_name']} - {date_str}]\n{c['content']}\n"

    except ConnectionError as e:
        st.error(f"❌ Erreur de connexion lors de la recherche vectorielle : {e}")
        logger.error(f"Erreur de connexion : {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors de la recherche vectorielle : {e}")
        logger.error(f"Erreur recherche : {e}")
        st.stop()

    # 3. GÉNÉRATION & AFFICHAGE (Système d'onglets)
    st.divider()

    # Création des onglets pour organiser l'information
    tab_report, tab_sources, tab_tech = st.tabs(["📝 Rapport d'Analyse", "📂 Sources Utilisées", "📊 Données Techniques"])

    # --- ONGLET 1 : RAPPORT ---
    with tab_report:
        st.subheader("Synthèse IA")

        try:
            system_prompt = load_prompt("rag_system_prompt.txt")
        except Exception:
            system_prompt = "Tu es un expert en analyse de documents de projet IT."

        with st.spinner("✍️ Rédaction du rapport en cours..."):
            try:
                response = safe_completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Voici les extraits:\n\n{context_str}"}
                    ]
                )
                ai_text = response.choices[0].message.content

                # Affichage joli type "Chat"
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(ai_text)

                st.toast("Analyse terminée avec succès !", icon="✅")

                # Bouton de téléchargement
                st.download_button(
                    label="📥 Télécharger le rapport (.txt)",
                    data=ai_text,
                    file_name=f"Rapport_Analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

                # Métriques de coût pour ce run
                output_tokens = response.usage.completion_tokens
                input_tokens_used = response.usage.prompt_tokens
                cost_info = calculate_cost(input_tokens_used, output_tokens, model_name)

            except ConnectionError as e:
                st.error("❌ Impossible de se connecter à l'API. Vérifiez votre connexion internet.")
                logger.error(f"Erreur de connexion API : {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération : {e}")
                logger.error(f"Erreur génération : {e}")
                st.stop()

    # --- ONGLET 2 : SOURCES ---
    with tab_sources:
        st.info(f"💡 L'IA a basé son analyse sur **{len(selected_chunks)} segments** extraits de vos documents.")

        for i, chunk in enumerate(selected_chunks):
            with st.expander(f"📄 Source {i+1}: {chunk['doc_name']} ({chunk['doc_type']})"):
                st.caption(f"Date: {chunk['date']}")
                st.markdown(f"```text\n{chunk['content']}\n```")

    # --- ONGLET 3 : TECHNIQUE & COÛTS ---
    with tab_tech:
        st.subheader("Métriques de la session")

        # Métriques Globales
        total_chunks_tokens = sum(estimate_tokens(c['content']) for c in chunks)
        embedding_cost = calculate_cost(total_chunks_tokens, 0, embedding_model_name)['total_cost']
        total_cost = cost_info['total_cost'] + embedding_cost

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segments Indexés", len(chunks))
        col2.metric("Tokens Indexés", f"{total_chunks_tokens:,}")
        col3.metric("Tokens Générés", f"{output_tokens:,}")
        col4.metric("Coût Total ($)", format_cost(total_cost))

        st.markdown("### 📋 État des fichiers")
        st.dataframe(pd.DataFrame(stats), width='stretch')

        st.markdown("### 💰 Détail des coûts")
        st.json({
            "Modele LLM": model_name,
            "Cout Entree LLM": format_cost(cost_info['input_cost']),
            "Cout Sortie LLM": format_cost(cost_info['output_cost']),
            "Cout Indexation (Embedding)": format_cost(embedding_cost),
            "Total": format_cost(total_cost)
        })
