"""
Composant Streamlit réutilisable pour la sélection de dossiers
"""
import streamlit as st
import os
from pathlib import Path
from typing import Optional


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
    Composant Streamlit simple pour sélectionner un dossier

    Args:
        default_path: Chemin par défaut (optionnel)
        key: Clé unique pour le composant Streamlit

    Returns:
        Chemin du dossier sélectionné
    """
    st.markdown("### 📂 Sélection du dossier")

    col1, col2 = st.columns([5, 1])

    with col1:
        # Utiliser la valeur de la session si elle existe, sinon le default_path
        current_value = st.session_state.get(f"{key}_path", default_path or "")

        folder_path = st.text_input(
            "Chemin du dossier à analyser :",
            value=current_value,
            placeholder="/chemin/vers/votre/dossier",
            key=f"{key}_textinput",
            label_visibility="collapsed"
        )

    # Bouton pour ouvrir le dialogue de fichiers
    with col2:
        if try_import_tkinter():
            if st.button("📁 Parcourir", key=f"{key}_browse", help="Ouvrir un dialogue de sélection", use_container_width=True):
                dialog_path = open_folder_dialog()
                if dialog_path:
                    st.session_state[f"{key}_path"] = dialog_path
                    st.rerun()
        else:
            st.info("⚠️", icon="⚠️")
            st.caption("Installer tkinter pour activer le bouton Parcourir")

    # Afficher des informations sur le dossier sélectionné
    if folder_path and folder_path.strip():
        folder_path = folder_path.strip()

        col_info1, col_info2 = st.columns([1, 1])

        with col_info1:
            if os.path.exists(folder_path):
                if os.path.isdir(folder_path):
                    st.success(f"✅ Dossier valide")
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

    return folder_path if folder_path else ""
