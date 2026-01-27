# Architecture du Projet

Ce document présente les différentes options d'architecture envisagées pour le système d'analyse de documents.

## Diagramme des Solutions

```mermaid
flowchart LR

  subgraph OPT1[Option 1]
    O1A[Home page] --> O1B[Sélection techno] --> O1C[Choix du client] --> O1D[Lancement du script]
  end

  subgraph OPT2[Option 2]
    O2A[Home page] --> O2B[Choix du client] --> O2C[Détection techno] --> O2D[Lancement du script]
  end

  subgraph BRAVO[A tester : solution Bravo]
    B1[Dossier : documents (tous clients)] --> B2[RAG : Vectorisation] --> B3[IHM : Choix du client]
    B3 --> B4[Lancement du script (.py)]
    NOTE1[[+ Script qui ajoute dynamiquement\nles nouveaux docs/clients dans le RAG]]
  end

  subgraph ALPHA[Solution Alpha]
    A1[IHM : Choix du répertoire client] --> A2[RAG : Vectorisation] --> A3[Lancement du script]
    Q1[[Un RAG est créé pour chaque client ?]]
  end
```

## Description des Solutions

### Option 1
Flux séquentiel avec sélection de la technologie en premier :
1. Page d'accueil
2. Sélection de la technologie
3. Choix du client
4. Lancement du script d'analyse

### Option 2
Flux séquentiel avec choix du client en premier :
1. Page d'accueil
2. Choix du client
3. Détection automatique de la technologie
4. Lancement du script d'analyse

### Solution Bravo (À tester)
Approche centralisée avec RAG unique :
- Tous les documents clients dans un seul dossier
- Vectorisation globale (RAG centralisé)
- Interface pour sélectionner le client
- Script pour ajouter dynamiquement de nouveaux documents/clients

**Avantage** : Un seul RAG pour tous les clients, mise à jour dynamique

### Solution Alpha (Actuelle)
Approche décentralisée par client :
- Interface pour choisir le répertoire du client
- Vectorisation spécifique au client
- Lancement du script d'analyse

**Question ouverte** : Un RAG est-il créé pour chaque client ?

## État Actuel

L'implémentation actuelle suit la **Solution Alpha** avec :
- Sélection manuelle du dossier client
- RAG créé à la demande avec cache
- Support des documents RPO, PTC, BCO, BDC
- Support multi-formats (PDF, DOCX, TXT, Excel, Images)
