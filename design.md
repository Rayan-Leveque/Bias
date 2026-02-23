# Design du pipeline — Point d'ancrage

## Hypothèse centrale

Lorsqu'un LLM évalue des candidats **séparément** (condition single), il tend à favoriser les minorités — probablement par sur-correction RLHF (social desirability). Lorsqu'il est placé en situation de **comparaison directe**, il tend à favoriser la majorité — les stéréotypes implicites s'activent.

Ce phénomène est à rapprocher de la littérature sur l'évaluation séparée vs. conjointe (Hsee, 1996) et des audit studies (Bertrand & Mullainathan, 2004).

**L'hypothèse est empirique et non confirmée — c'est précisément ce que ce pipeline cherche à tester.**

---

## Domaine

**Embauche / évaluation de CV** — contexte français.

Littérature de référence : études de testing ISM Corum, données DARES, et l'article [arXiv:2602.10117](https://arxiv.org/pdf/2602.10117) sur la détection automatique de biais non verbalisés.

---

## Structure expérimentale

### Condition A — Single

- Présenter **un seul CV** au modèle
- Demander une décision binaire : **Accepter / Rejeter**
- Répéter pour N itérations par groupe (ex: N=100)
- Calculer le taux d'acceptation par groupe

### Condition B — Comparatif

- Présenter **deux CV côte à côte** (candidat A vs candidat B)
- Les CV sont identiques sauf les marqueurs d'identité
- Demander un **choix forcé** : "Lequel retenezvous ?"
- Randomiser l'ordre de présentation (A/B vs B/A)
- Calculer P(majorité choisie)

---

## Stimuli

- **Plusieurs templates de CV** (pour éviter que les résultats soient liés à un seul profil)
- **Marqueurs d'identité** : prénom + adresse (photo en option pour les VLMs)
- Paires de prénoms à définir : ex. "Kévin Martin" ↔ "Moussa Diallo"

### Points ouverts sur les stimuli
- Nombre de templates (à décider)
- Liste définitive des paires de prénoms
- Adresses : arrondissements/communes à utiliser
- Condition de contrôle possible (prénom neutre / ambigu) — non décidée

---

## Analyse statistique

**Tests d'indépendance** (chi-carré ou Fisher exact) :

- Condition single : tester si le taux d'acceptation est indépendant du groupe ethnique
- Condition comparative : tester si le choix est indépendant de l'identité du candidat A/B
- Comparer les deux conditions pour mesurer le "shifting bias"

---

## Modèles

Plusieurs modèles à tester (liste non encore définie). L'objectif à terme est de comparer si le phénomène est universel ou propre à certaines architectures/entraînements.

---

## Ce qui reste à décider

| Question | Statut |
|---|---|
| Paires de prénoms définitives | Ouvert |
| Nombre de templates de CV | Ouvert |
| Adresses à utiliser | Ouvert |
| Condition de contrôle (prénom neutre) | Non décidée |
| Modèles à tester en premier | Ouvert |
| Formulation exacte des prompts | Ouvert |
