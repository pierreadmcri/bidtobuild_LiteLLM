# 🏗️ AI Document Scanner

Une application **Streamlit** intelligente qui automatise l'analyse et la synthèse de documents de construction (BCO, RBO, PTC, BDC).

L'outil scanne un répertoire local, identifie les fichiers pertinents grâce à des motifs (Regex), sélectionne automatiquement la version la plus récente en cas de doublon, et génère une synthèse structurée via un LLM (Azure OpenAI / GPT-4) grâce à **LiteLLM**.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LiteLLM](https://img.shields.io/badge/Backend-LiteLLM-orange.svg)

## 🚀 Fonctionnalités

- **📂 Scan Intelligent :** Analyse automatique d'un dossier local.
- **🔍 Filtrage Regex :** Détection automatique des types de documents :
  - **RBO** (Run / Build)
  - **PTC** (Proposition / Technique)
  - **BCO** (Budget / Mandays)
  - **BDC** (Bon de Commande)
- **📅 Gestion des Versions :** En cas de fichiers multiples pour un même type, seule la version la plus récente (date de modification) est conservée.
- **📄 Support Multi-formats :** Lecture native des fichiers `.pdf`, `.docx` et `.txt`.
- **💰 Estimation des Tokens :** Calcul du coût en tokens avant envoi au LLM.
- **🧠 Synthèse IA :** Génération d'un résumé financier et technique via Azure OpenAI (ou tout autre modèle supporté par LiteLLM).

## 🛠️ Prérequis technique

- **Python 3.11** (Recommandé)
- Accès à une clé API (Azure OpenAI, OpenAI, etc.)

## 📦 Installation

1. **Cloner le projet**
   ```bash
   git clone [https://github.com/votre-user/votre-repo.git](https://github.com/votre-user/votre-repo.git)
   cd votre-repo