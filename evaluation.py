"""
evaluer_modele.py
─────────────────
Évaluation complète du modèle sur Validation ET Test.
Génère un rapport de métriques global et par classe.

Usage :
    python evaluer_modele.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from sources.donnees.chargeur_donnees import creer_data_loaders
from sources.entrainement.integration_pipeline import assembler_pipeline
from sources.utilitaires.configuration import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
NOMS_CLASSES = cfg.classes.noms
DOSSIER_RESULTATS = Path("experiences/resultats")
DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
HORODATAGE = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT = Path("experiences/points_de_sauvegarde/meilleur_modele.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 65)
print("📊 ÉVALUATION COMPLÈTE DU MODÈLE")
print("=" * 65)
print(f"   Device     : {DEVICE}")
print(f"   Checkpoint : {CHECKPOINT}")

# ══════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════
print("\n📦 Chargement des données...")
_, val_loader, test_loader = creer_data_loaders(batch_size=16)
print(f"   Validation : {len(val_loader.dataset)} échantillons")
print(f"   Test       : {len(test_loader.dataset)} échantillons")

print("\n🧠 Chargement du modèle...")
pipeline = assembler_pipeline(device=DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
pipeline.load_state_dict(checkpoint["model_state_dict"])
pipeline.to(DEVICE)
pipeline.eval()
print("   ✅ Modèle chargé")

# ══════════════════════════════════════════
# 2. FONCTION D'ÉVALUATION
# ══════════════════════════════════════════

@torch.no_grad()
def evaluer_loader(loader, nom="Test"):
    """Évalue le modèle sur un DataLoader."""
    toutes_preds = []
    tous_labels = []
    
    for batch in tqdm(loader, desc=f"   {nom}"):
        textes = batch["texte"]
        images = batch["image"]
        cibles = batch["classe"]
        
        logits = pipeline(textes, images)
        probas = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probas, dim=-1)
        
        toutes_preds.extend(preds.cpu().numpy())
        tous_labels.extend(cibles.cpu().numpy())
    
    preds = np.array(toutes_preds)
    labels = np.array(tous_labels)
    
    # Métriques globales
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
    
    # Par classe
    f1_classe = f1_score(labels, preds, average=None, zero_division=0)
    prec_classe = precision_score(labels, preds, average=None, zero_division=0)
    rec_classe = recall_score(labels, preds, average=None, zero_division=0)
    
    # Matrice de confusion
    cm = confusion_matrix(labels, preds)
    rapport = classification_report(labels, preds, target_names=NOMS_CLASSES, zero_division=0)
    
    return {
        "accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "precision_macro": prec_macro, "recall_macro": rec_macro,
        "f1_par_classe": dict(zip(NOMS_CLASSES, f1_classe)),
        "precision_par_classe": dict(zip(NOMS_CLASSES, prec_classe)),
        "recall_par_classe": dict(zip(NOMS_CLASSES, rec_classe)),
        "confusion_matrix": cm, "classification_report": rapport,
    }

# ══════════════════════════════════════════
# 3. ÉVALUATION
# ══════════════════════════════════════════

print("\n🔄 Évaluation Validation...")
res_val = evaluer_loader(val_loader, "Validation")

print("\n🔄 Évaluation Test...")
res_test = evaluer_loader(test_loader, "Test")

# ══════════════════════════════════════════
# 4. AFFICHAGE CONSOLE
# ══════════════════════════════════════════

for nom, res in [("VALIDATION", res_val), ("TEST", res_test)]:
    print(f"\n{'=' * 65}")
    print(f"📊 RÉSULTATS — {nom}")
    print(f"{'=' * 65}")
    print(f"\n   Accuracy        : {res['accuracy']:.4f}")
    print(f"   F1 Macro        : {res['f1_macro']:.4f}")
    print(f"   F1 Weighted     : {res['f1_weighted']:.4f}")
    print(f"   Précision Macro : {res['precision_macro']:.4f}")
    print(f"   Rappel Macro    : {res['recall_macro']:.4f}")
    print(f"\n   📈 Par classe :")
    for nom_classe in NOMS_CLASSES:
        p = res['precision_par_classe'][nom_classe]
        r = res['recall_par_classe'][nom_classe]
        f1 = res['f1_par_classe'][nom_classe]
        print(f"      {nom_classe:<25} : P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"\n{res['classification_report']}")

# ══════════════════════════════════════════
# 5. GRAPHIQUES
# ══════════════════════════════════════════

print("📊 Génération des graphiques...")
cm = res_test["confusion_matrix"]

# Matrice de confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Matrice de Confusion — Test", fontsize=14, fontweight="bold")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=NOMS_CLASSES, yticklabels=NOMS_CLASSES,
            ax=axes[0], linewidths=0.5)
axes[0].set_title("Valeurs absolues")
axes[0].set_xlabel("Prédit"); axes[0].set_ylabel("Réel")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="YlOrRd",
            xticklabels=NOMS_CLASSES, yticklabels=NOMS_CLASSES,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title("Normalisée (Rappel)")
axes[1].set_xlabel("Prédit"); axes[1].set_ylabel("Réel")

plt.tight_layout()
cm_path = DOSSIER_RESULTATS / f"confusion_matrix_{HORODATAGE}.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Matrice de confusion → {cm_path}")

# Barres F1/Précision/Rappel
f1_vals = [res_test['f1_par_classe'][n] for n in NOMS_CLASSES]
prec_vals = [res_test['precision_par_classe'][n] for n in NOMS_CLASSES]
rec_vals = [res_test['recall_par_classe'][n] for n in NOMS_CLASSES]

x = np.arange(len(NOMS_CLASSES))
largeur = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - largeur, f1_vals, largeur, label="F1-Score", color="#4C72B0")
bars2 = ax.bar(x, prec_vals, largeur, label="Précision", color="#DD8452")
bars3 = ax.bar(x + largeur, rec_vals, largeur, label="Rappel", color="#55A868")

for bar in [*bars1, *bars2, *bars3]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(NOMS_CLASSES, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score"); ax.set_title("Métriques par classe — Test", fontweight="bold")
ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)

barres_path = DOSSIER_RESULTATS / f"metriques_barres_{HORODATAGE}.png"
plt.savefig(barres_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Barres métriques → {barres_path}")

# ══════════════════════════════════════════
# 6. SAUVEGARDE JSON
# ══════════════════════════════════════════

resultats_json = {
    "validation": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                   for k, v in res_val.items() if k != "classification_report"},
    "test": {k: v.tolist() if isinstance(v, np.ndarray) else v 
             for k, v in res_test.items() if k != "classification_report"},
}

json_path = DOSSIER_RESULTATS / f"evaluation_complete_{HORODATAGE}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resultats_json, f, ensure_ascii=False, indent=2)
print(f"   ✅ JSON → {json_path}")

# ══════════════════════════════════════════
# 7. RÉSUMÉ FINAL
# ══════════════════════════════════════════

print(f"\n{'=' * 65}")
print(f"📊 RÉSUMÉ FINAL")
print(f"{'=' * 65}")
print(f"   Validation — Acc: {res_val['accuracy']:.4f} | F1: {res_val['f1_macro']:.4f}")
print(f"   Test       — Acc: {res_test['accuracy']:.4f} | F1: {res_test['f1_macro']:.4f}")
print(f"\n   ✅ Test Accuracy > Validation : {'OUI (bonne généralisation)' if res_test['accuracy'] > res_val['accuracy'] else 'ATTENTION (overfitting possible)'}")
print(f"📁 Résultats : {DOSSIER_RESULTATS}/")
print(f"{'=' * 65}\n")