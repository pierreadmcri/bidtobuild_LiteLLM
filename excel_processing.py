"""Module d'extraction spécialisée des données Excel Build/Run.

Extrait et assainit les données des onglets Build et Run des fichiers BCO
pour une ingestion RAG optimisée avec des libellés sémantiques.

Adapté depuis Python_Excel_Tools pour intégration dans le pipeline RAG
de bidtobuild_LiteLLM. Les arguments CLI (input/output/bco-dir) ont été
retirés car la gestion des chemins est assurée par rag_analysis.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTES EXCEL BUILD/RUN
# ==========================================

REQUIRED_SHEETS = {"Build", "Run"}

BUILD_HEADER_ROW_INDEX = 32
BUILD_DATA_START_ROW_INDEX = 35
BUILD_MARGE_CELL = (5, 5)
RUN_HEADER_ROW_INDEX = 32
RUN_TOTAL_VALUE_ROW_INDEX = 33
RUN_DURATION_CELL = (1, 3)

BUILD_TOTAL_HEADER_TEXT = "Nb. Jours (à produire) TOTAL"
BUILD_CCJM_HEADER_TEXT = "CCJM"
RUN_TOTAL_HEADER_TEXT = "nb. jours (à produire) Total"


# ==========================================
# VALIDATION
# ==========================================

def validate_sheets(excel_path: Path) -> None:
    """Vérifie que les onglets Build et Run sont présents."""
    workbook = pd.ExcelFile(excel_path)
    missing = REQUIRED_SHEETS - set(workbook.sheet_names)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing required sheet(s): {missing_list}. "
            f"Available sheets: {', '.join(workbook.sheet_names)}"
        )


def load_required_sheets(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les onglets Build et Run en DataFrames."""
    return (
        pd.read_excel(excel_path, sheet_name="Build", header=None, engine="openpyxl"),
        pd.read_excel(excel_path, sheet_name="Run", header=None, engine="openpyxl"),
    )


# ==========================================
# FONCTIONS UTILITAIRES DE FORMATAGE
# ==========================================

def round_numeric_value(value: object, decimals: int = 2) -> object:
    """Arrondit une valeur numérique."""
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return value
    return round(float(numeric_value), decimals)


def format_percentage(value: object) -> str | None:
    """Formate une valeur en pourcentage."""
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return None
    formatted_value = round(float(numeric_value), 2) * 100
    return f"{formatted_value:.2f}%"


def normalize_value(value: object) -> object:
    """Normalise une valeur pandas pour la sérialisation."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


# ==========================================
# RECHERCHE DE COLONNES
# ==========================================

def find_column_index_on_row(
    dataframe: pd.DataFrame,
    *,
    row_index: int,
    target_text: str,
    sheet_name: str,
) -> int:
    """Trouve l'index de colonne contenant le texte cible sur une ligne donnée."""
    if dataframe.shape[0] <= row_index:
        raise ValueError(
            f"{sheet_name} sheet is too short to locate the target header on row {row_index + 1}."
        )

    normalized_row = dataframe.iloc[row_index].astype("string").str.strip().str.casefold()
    matching_cols = normalized_row[normalized_row == target_text.casefold()].index.tolist()
    if not matching_cols:
        raise ValueError(
            f"Could not find '{target_text}' on {sheet_name} row {row_index + 1}."
        )
    return int(matching_cols[0])


# ==========================================
# EXTRACTION DES DONNÉES BUILD
# ==========================================

def extract_build_rows(build_df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les lignes de profils Build avec filtrage sur les valeurs non nulles."""
    total_col_index = find_column_index_on_row(
        build_df,
        row_index=BUILD_HEADER_ROW_INDEX,
        target_text=BUILD_TOTAL_HEADER_TEXT,
        sheet_name="Build",
    )
    ccjm_col_index = find_column_index_on_row(
        build_df,
        row_index=BUILD_HEADER_ROW_INDEX,
        target_text=BUILD_CCJM_HEADER_TEXT,
        sheet_name="Build",
    )

    selected_df = build_df.iloc[:, [1, 2, total_col_index, ccjm_col_index]].copy()
    selected_df = selected_df.iloc[BUILD_DATA_START_ROW_INDEX:].copy()

    dynamic_total_column = pd.to_numeric(selected_df.iloc[:, 2], errors="coerce")
    column_c_not_empty = (
        selected_df.iloc[:, 1].notna() & selected_df.iloc[:, 1].astype("string").str.strip().ne("")
    )
    filtered_df = selected_df[
        (dynamic_total_column.notna()) & (dynamic_total_column != 0) & column_c_not_empty
    ].copy()

    filtered_df.iloc[:, 2] = filtered_df.iloc[:, 2].apply(round_numeric_value)
    filtered_df.iloc[:, 3] = filtered_df.iloc[:, 3].apply(round_numeric_value)
    filtered_df.columns = ["Profils internes", "Type", "Valeurs", "CCJM"]
    return filtered_df


def extract_build_tag_values(build_df: pd.DataFrame) -> dict[str, object]:
    """Extrait les indicateurs clés du Build (marge, charge totale)."""
    total_col_index = find_column_index_on_row(
        build_df,
        row_index=BUILD_HEADER_ROW_INDEX,
        target_text=BUILD_TOTAL_HEADER_TEXT,
        sheet_name="Build",
    )

    marge_build = build_df.iat[BUILD_MARGE_CELL[0], BUILD_MARGE_CELL[1]]
    charge_total = None
    if build_df.shape[0] > BUILD_DATA_START_ROW_INDEX:
        charge_total = round_numeric_value(build_df.iat[BUILD_DATA_START_ROW_INDEX, total_col_index])

    return {
        "Marge Build": format_percentage(marge_build),
        "Charge totale BUILD JH": charge_total,
    }


# ==========================================
# EXTRACTION DES DONNÉES RUN
# ==========================================

def extract_run_tag_values(run_df: pd.DataFrame) -> dict[str, object]:
    """Extrait les indicateurs clés du Run (charge totale, durée)."""
    charge_totale_run = None
    if run_df.shape[0] > RUN_TOTAL_VALUE_ROW_INDEX:
        total_col_index = find_column_index_on_row(
            run_df,
            row_index=RUN_HEADER_ROW_INDEX,
            target_text=RUN_TOTAL_HEADER_TEXT,
            sheet_name="Run",
        )
        charge_totale_run = round_numeric_value(run_df.iat[RUN_TOTAL_VALUE_ROW_INDEX, total_col_index])

    duree_run = None
    if run_df.shape[0] >= 2 and run_df.shape[1] >= 4:
        duree_run = round_numeric_value(run_df.iat[RUN_DURATION_CELL[0], RUN_DURATION_CELL[1]])

    return {
        "Charge totale RUN JH": charge_totale_run,
        "Durée du RUN en mois": duree_run,
    }


# ==========================================
# CONVERSION DATAFRAME → DICTS
# ==========================================

def dataframe_to_rows(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Convertit un DataFrame de profils Build en liste de dicts normalisés."""
    return [
        {
            "Profils internes": normalize_value(row["Profils internes"]),
            "Type": normalize_value(row["Type"]),
            "Valeurs": normalize_value(row["Valeurs"]),
            "CCJM": normalize_value(row["CCJM"]),
        }
        for _, row in dataframe.iterrows()
    ]


# ==========================================
# CONSTRUCTION DU RECORD JSONL POUR LE RAG
# ==========================================

def build_file_chunk_record(
    excel_path: Path,
    build_rows: list[dict[str, object]],
    build_tags: dict[str, object],
    run_tags: dict[str, object],
) -> dict[str, Any]:
    """Construit un enregistrement JSONL structuré avec labels sémantiques pour le RAG."""
    normalized_build_tags = {key: normalize_value(value) for key, value in build_tags.items()}
    normalized_run_tags = {key: normalize_value(value) for key, value in run_tags.items()}

    preview = " ; ".join(
        (
            f"{row.get('Profils internes')} / {row.get('Type')} / "
            f"{row.get('Valeurs')} / CCJM: {row.get('CCJM')}"
        )
        for row in build_rows #[:5]
    )
    chunk_text = (
        f"Fichier: {excel_path.name} | Profils Build: {len(build_rows)} lignes | "
        f"Build tags: {normalized_build_tags} | Run tags: {normalized_run_tags} | "
        f"Aperçu: {preview}"
    )

    return {
        "id": f"{excel_path.stem}:build_run_summary",
        "text": chunk_text,
        "labels": {
            "source_file": excel_path.name,
            "source_path": str(excel_path),
            "sheet": "Build/Run",
            "section": "build_run_summary",
            "keywords": ["Build", "Run", "Profils internes", "Type", "Valeurs", "CCJM"],
            "chunk_type": "excel_summary",
            "language": "fr",
        },
        "data": {
            "build_rows": build_rows,
            "tags": {
                "build": normalized_build_tags,
                "run": normalized_run_tags,
            },
        },
    }


# ==========================================
# POINT D'ENTRÉE PRINCIPAL
# ==========================================

def process_excel_file(file_path: Path) -> list[dict[str, Any]]:
    """
    Traite un fichier Excel et retourne les enregistrements JSONL pour le RAG.

    Valide la présence des onglets Build/Run, extrait les données pertinentes
    (profils, indicateurs clés) et construit des records structurés avec labels
    sémantiques optimisés pour la recherche vectorielle.

    Args:
        file_path: Chemin vers le fichier Excel (.xlsx ou .xlsm)

    Returns:
        Liste d'enregistrements JSONL structurés

    Raises:
        ValueError: Si les onglets requis sont manquants ou les données invalides
    """
    file_path = Path(file_path)
    logger.info(f"Extraction spécialisée Build/Run : {file_path.name}")

    validate_sheets(file_path)
    build_df, run_df = load_required_sheets(file_path)
    build_rows = dataframe_to_rows(extract_build_rows(build_df).reset_index(drop=True))
    build_tags = extract_build_tag_values(build_df)
    run_tags = extract_run_tag_values(run_df)

    logger.info(
        f"Extraction terminée : {len(build_rows)} profils Build, "
        f"tags Build={build_tags}, tags Run={run_tags}"
    )

    return [
        build_file_chunk_record(
            excel_path=file_path,
            build_rows=build_rows,
            build_tags=build_tags,
            run_tags=run_tags,
        )
    ]
