"""
Composant Streamlit réutilisable pour la sélection de dossiers
"""
import streamlit as st
import os
from pathlib import Path
import json
from typing import Optional

# Fichier pour sauvegarder les dossiers favoris
FAVORITES_FILE = ".folder_favorites.json"


def load_favorites() -> list:
    """
    Charge la liste des dossiers favoris depuis le fichier

    Returns:
        Liste des chemins de dossiers favoris
    """
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
                favorites = json.load(f)
                # Filtrer les dossiers qui n'existent plus
                return [f for f in favorites if os.path.isdir(f)]
        except Exception:
            return []
    return []


def save_favorites(favorites: list) -> None:
    """
    Sauvegarde la liste des dossiers favoris

    Args:
        favorites: Liste des chemins de dossiers
    """
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favorites, f, indent=2)
    except Exception:
        pass


def add_to_favorites(folder_path: str) -> None:
    """
    Ajoute un dossier aux favoris (max 10)

    Args:
        folder_path: Chemin du dossier à ajouter
    """
    if not os.path.isdir(folder_path):
        return

    favorites = load_favorites()

    # Normaliser le chemin
    normalized_path = str(Path(folder_path).resolve())

    # Retirer si déjà présent (pour le remettre en premier)
    if normalized_path in favorites:
        favorites.remove(normalized_path)

    # Ajouter en première position
    favorites.insert(0, normalized_path)

    # Limiter à 10 favoris
    favorites = favorites[:10]

    save_favorites(favorites)


def try_import_tkinter() -> bool:
    """
    Vérifie si tkinter est disponible

    Returns:
        True si tkinter est disponible, False sinon
    """
    try:
        import tkinter
        return True
    except ImportError:
        return False


def open_folder_dialog() -> Optional[str]:
    """
    Ouvre un dialogue de sélection de dossier (nécessite tkinter)

    Returns:
        Chemin du dossier sélectionné ou None
    """
    try:
        from tkinter import Tk
        from tkinter.filedialog import askdirectory

        # Créer une fenêtre Tk cachée
        root = Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)

        # Ouvrir le dialogue
        folder_path = askdirectory(
            title="Sélectionner un dossier à analyser",
            mustexist=True
        )

        root.destroy()

        return folder_path if folder_path else None

    except Exception:
        return None


def folder_selector(default_path: str = None, key: str = "folder_selector") -> str:
    """
    Composant Streamlit pour sélectionner un dossier

    Args:
        default_path: Chemin par défaut (optionnel)
        key: Clé unique pour le composant Streamlit

    Returns:
        Chemin du dossier sélectionné
    """
    # Chemins suggérés par défaut
    default_suggestions = [
        ("📂 Dossier courant", os.getcwd()),
        ("📁 documents_types (dossier par défaut)", os.path.join(os.getcwd(), "documents_types")),
        ("🏠 Dossier utilisateur", str(Path.home())),
    ]

    # Charger les favoris
    favorites = load_favorites()

    # Construire la liste complète des options
    options = []

    # Ajouter les favoris
    if favorites:
        options.append(("─── ⭐ Favoris ───", ""))
        for i, fav in enumerate(favorites):
            folder_name = Path(fav).name
            options.append((f"⭐ {folder_name}", fav))

    # Ajouter les suggestions par défaut
    options.append(("─── 📋 Suggestions ───", ""))
    options.extend(default_suggestions)

    # Option pour chemin personnalisé
    options.append(("─── ✏️ Personnalisé ───", ""))
    options.append(("✏️ Entrer un chemin personnalisé", "custom"))

    # Créer le selectbox
    st.markdown("### 📂 Sélection du dossier")

    col1, col2 = st.columns([4, 1])

    with col1:
        # Trouver l'index par défaut
        default_index = 0
        if default_path:
            for i, (label, path) in enumerate(options):
                if path == default_path:
                    default_index = i
                    break

        selected = st.selectbox(
            "Choisir un dossier :",
            options=options,
            format_func=lambda x: x[0],
            index=default_index,
            key=f"{key}_selectbox"
        )

        selected_label, selected_path = selected

    # Bouton pour ouvrir le dialogue de fichiers (si tkinter disponible)
    with col2:
        st.write("")
        st.write("")
        if try_import_tkinter():
            if st.button("🗂️ Parcourir", key=f"{key}_browse", help="Ouvrir un dialogue de sélection"):
                dialog_path = open_folder_dialog()
                if dialog_path:
                    st.session_state[f"{key}_custom_path"] = dialog_path
                    st.rerun()

    # Si "custom" ou séparateur sélectionné, afficher le champ de saisie
    if selected_path == "custom" or selected_path == "":
        folder_path = st.text_input(
            "Entrer le chemin complet du dossier :",
            value=st.session_state.get(f"{key}_custom_path", default_path or ""),
            placeholder="/chemin/vers/votre/dossier",
            key=f"{key}_textinput"
        )
    else:
        folder_path = selected_path

    # Afficher des informations sur le dossier sélectionné
    if folder_path and folder_path.strip():
        folder_path = folder_path.strip()

        col_info1, col_info2, col_info3 = st.columns([2, 2, 1])

        with col_info1:
            if os.path.exists(folder_path):
                if os.path.isdir(folder_path):
                    st.success(f"✅ Dossier valide : `{folder_path}`")
                else:
                    st.error("❌ Ce chemin n'est pas un dossier")
            else:
                st.warning("⚠️ Ce dossier n'existe pas")

        with col_info2:
            if os.path.isdir(folder_path):
                # Compter les fichiers supportés
                try:
                    from config import ALLOWED_EXTENSIONS
                    file_count = sum(
                        1 for root, _, files in os.walk(folder_path)
                        for f in files
                        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
                    )
                    st.info(f"📄 {file_count} fichier(s) compatible(s)")
                except Exception:
                    pass

        with col_info3:
            # Bouton pour ajouter aux favoris
            if os.path.isdir(folder_path):
                if st.button("⭐ Favori", key=f"{key}_add_fav", help="Ajouter aux favoris"):
                    add_to_favorites(folder_path)
                    st.success("Ajouté !")
                    st.rerun()

    return folder_path if folder_path else ""
