import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime
from pathlib import Path

# Imports locaux
import config
from utils import (
    validate_file_path,
    validate_file_size,
    safe_completion,
    estimate_tokens,
    load_prompt,
    ValidationError,
    FileTooLargeError,
    logger
)

# Imports pour lire les vrais fichiers
from pypdf import PdfReader
from docx import Document

# ==========================================
# 0. CONFIGURATION
# ==========================================

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

model_name = config.MODEL_NAME
max_input_tokens = config.MAX_INPUT_TOKENS
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS

# Configuration de la page
st.set_page_config(page_title="Scanner Local Documents", page_icon="📂", layout="wide")

# ==========================================
# 1. FONCTIONS UTILITAIRES (Lecture Fichiers)
# ==========================================

def read_file_content(filepath):
    """
    Lit le contenu texte d'un fichier selon son extension.

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
                page_text = page.extract_text() or ""
                content += page_text + "\n"
            if not content.strip():
                return "[Alerte : Aucun texte lisible extrait du PDF. Le document est peut-être scanné ou protégé.]"
        elif ext == ".docx":
            doc = Document(filepath)
            for para in doc.paragraphs:
                content += para.text + "\n"
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            logger.warning(f"Format non supporté : {ext}")
            return f"[Format {ext} non supporté]"

        logger.info(f"Fichier lu avec succès : {filepath}")

    except FileNotFoundError:
        logger.error(f"Fichier introuvable : {filepath}")
        return f"[Erreur : Fichier introuvable]"
    except PermissionError:
        logger.error(f"Permission refusée : {filepath}")
        return f"[Erreur : Permission refusée]"
    except Exception as e:
        logger.error(f"Erreur de lecture {filepath}: {str(e)}")
        return f"[Erreur de lecture : {str(e)}]"

    return content


def scan_directory(directory_path, allowed_extensions=ALLOWED_EXTENSIONS):
    """
    Scanne récursivement un dossier pour lister les fichiers autorisés.

    Args:
        directory_path: Chemin du dossier à scanner
        allowed_extensions: Set des extensions autorisées

    Returns:
        Liste de dictionnaires contenant les métadonnées des fichiers

    Raises:
        ValidationError: Si le chemin est invalide
    """
    files_data = []

    # Validation du chemin
    try:
        validated_path = validate_file_path(directory_path)
    except ValidationError as e:
        logger.error(f"Chemin invalide : {e}")
        raise

    if not validated_path.is_dir():
        logger.warning(f"Le chemin n'est pas un dossier : {directory_path}")
        return []

    for root_dir, _, files in os.walk(validated_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_extensions:
                continue

            filepath = os.path.join(root_dir, filename)

            if not os.path.isfile(filepath):
                continue

            try:
                stats = os.stat(filepath)
                mod_time = datetime.fromtimestamp(stats.st_mtime)

                files_data.append({
                    "name": filename,
                    "path": filepath,
                    "date": mod_time,
                    "size": stats.st_size
                })
            except Exception as e:
                logger.warning(f"Impossible de lire les métadonnées de {filepath}: {e}")
                continue

    logger.info(f"Scan terminé : {len(files_data)} fichiers trouvés")
    return files_data


def truncate_text_by_tokens(text, max_tokens):
    """Tronque un texte pour respecter une limite de tokens approximative."""
    if estimate_tokens(text) <= max_tokens:
        return text, False

    words = text.split()
    low, high = 0, len(words)
    best_text = ""

    while low < high:
        mid = (low + high) // 2
        candidate = " ".join(words[:mid])
        if estimate_tokens(candidate) <= max_tokens:
            best_text = candidate
            low = mid + 1
        else:
            high = mid

    return best_text, True

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
        st.error("⚠️ Veuillez entrer un chemin de dossier.")
    else:
        # Validation du chemin avant traitement
        try:
            validate_file_path(folder_path)
        except ValidationError as e:
            st.error(f"❌ Chemin invalide : {e}")
            logger.error(f"Validation du chemin échouée : {e}")
            st.stop()

        # A. BARRE DE PROGRESSION
        progress_bar = st.progress(0, text="Initialisation...")

        # Etape 1 : Scan et Filtrage
        progress_bar.progress(20, text="Scan du répertoire et filtrage des dates...")

        try:
            final_docs, logs, error = process_files(folder_path)
        except ValidationError as e:
            st.error(f"❌ Erreur de validation : {e}")
            logger.error(f"Erreur lors du traitement : {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")
            logger.error(f"Erreur inattendue lors du traitement : {e}")
            st.stop()

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
            token_limit_reached = False

            for doc in final_docs:
                remaining_budget = max_input_tokens - total_input_tokens
                if remaining_budget <= 0:
                    token_limit_reached = True
                    logs.append(f"⛔️ Limite de {max_input_tokens} tokens d'entrée atteinte, les documents suivants sont ignorés.")
                    break

                truncated_content, was_truncated = truncate_text_by_tokens(doc["content"], remaining_budget)
                doc_tokens = estimate_tokens(truncated_content)
                doc['tokens'] = doc_tokens
                doc_context = f"\n--- DOCUMENT: {doc['name']} ---\n{truncated_content}\n"
                full_context += doc_context
                total_input_tokens += doc_tokens

                if was_truncated:
                    token_limit_reached = True
                    logs.append(f"⚠️ Contexte tronqué pour {doc['name']} afin de rester sous {max_input_tokens} tokens.")

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
            if token_limit_reached:
                st.warning("La limite de tokens d'entrée a été atteinte. Certains documents ont pu être tronqués ou ignorés.")

            # Etape 4 : Appel LLM
            progress_bar.progress(85, text="Interrogation de l'IA (Patience)...")

            # Chargement du prompt système depuis le fichier
            try:
                system_prompt = load_prompt("app_system_prompt.txt")
            except Exception as e:
                st.error(f"Impossible de charger le prompt système : {e}")
                logger.error(f"Erreur chargement prompt : {e}")
                system_prompt = "Tu es un expert en analyse de documents de projet IT."

            try:
                # Utilisation de safe_completion avec retry automatique
                response = safe_completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_context}
                    ],
                    api_key=os.getenv("AZURE_API_KEY"),
                    api_base=os.getenv("AZURE_API_BASE"),
                    api_version=os.getenv("AZURE_API_VERSION")
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

            except ConnectionError as e:
                progress_bar.progress(100, text="Erreur de connexion")
                st.error("❌ Impossible de se connecter à l'API. Vérifiez votre connexion internet.")
                logger.error(f"Erreur de connexion API : {e}")
                st.code(str(e))
            except PermissionError as e:
                progress_bar.progress(100, text="Erreur d'authentification")
                st.error("❌ Erreur d'authentification. Vérifiez votre clé API Azure.")
                logger.error(f"Erreur d'authentification : {e}")
                st.code(str(e))
            except Exception as e:
                progress_bar.progress(100, text="Erreur")
                st.error("❌ Erreur lors de l'appel IA. Détails techniques ci-dessous :")
                logger.error(f"Erreur inattendue lors de l'appel LLM : {e}")
                st.code(str(e))
