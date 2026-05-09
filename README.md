# Classification de Tweets de Crise

Système de deep learning pour la classification automatique de tweets postés lors de catastrophes naturelles. Combine l'analyse du texte et des images pour identifier les messages urgents nécessitant une intervention rapide.

Développé par Bensalah Brahim et Rafik Asbai.

## Architecture
BERTweet + LoRA (texte) | ResNet-50 gelé (image) | Cross-Attention 8 têtes + Gate Network (fusion) | 3 classes : Urgence, Information, Non pertinent. Seulement 2.2M paramètres entraînables sur 160M.

## Performances
Accuracy : 83.69% | F1 Macro : 81.77% | F1 Urgence : 88.7% (Rappel : 90.3%)

## Données
60 000 tweets issus de CrisisMMD, Multimodal et CrisisLexT26 — 33 catastrophes couvertes.

