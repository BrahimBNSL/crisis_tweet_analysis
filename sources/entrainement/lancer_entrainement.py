"""
lancer_entrainement.py
───────────────────────
Script principal pour lancer l'entraînement complet du modèle multimodal
de classification de tweets de crise.

Utilisation :
    python -m sources.entrainement.lancer_entrainement
    python -m sources.entrainement.lancer_entrainement --epochs 30 --batch_size 8 --device cuda
"""

import sys
import logging
import argparse
from pathlib import Path

import torch

# Ajouter la racine au path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sources.utilitaires.configuration import cfg
from sources.utilitaires.reproductibilite import fixer_reproductibilite
from sources.donnees.chargeur_donnees import creer_data_loaders
from sources.entrainement.integration_pipeline import lancer_entrainement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('entrainement.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def parser_arguments():
    parser = argparse.ArgumentParser(description="Entraînement multimodal de crise")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=cfg.entrainement.nb_epochs)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--graine", type=int, default=cfg.entrainement.graine)
    parser.add_argument("--dossier_checkpoint", type=str, default=str(cfg.chemins.dossier_checkpoints))
    return parser.parse_args()


def main():
    print("\n" + "=" * 80)
    print("🚨 CLASSIFICATION MULTIMODALE DE TWEETS DE CRISE")
    print("=" * 80)
    
    args = parser_arguments()
    
    print(f"\n⚙️  Configuration :")
    print(f"   • Device        : {args.device}")
    print(f"   • Epochs        : {args.epochs}")
    print(f"   • Batch size    : {args.batch_size}")
    print(f"   • Graine        : {args.graine}")
    
    # Reproductibilité
    fixer_reproductibilite(graine=args.graine)
    
    # Vérifier GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA non disponible → basculement sur CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        logger.info(f"🖥️  GPU détecté : {torch.cuda.get_device_name(0)}")
        logger.info(f"   Mémoire totale : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go")
    
    # DataLoaders
    logger.info("📦 Création des DataLoaders...")
    train_loader, val_loader, test_loader = creer_data_loaders(
        batch_size=args.batch_size,
        nb_workers=0,
    )
    
    # Entraînement
    logger.info(f"\n{'='*80}")
    logger.info("🚀 LANCEMENT DE L'ENTRAÎNEMENT")
    logger.info(f"{'='*80}")
    
    historique = lancer_entrainement(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=args.device,
        dossier_checkpoint=Path(args.dossier_checkpoint),
        freeze_bert=True,
        freeze_resnet=True,
    )
    
    # Résumé
    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("=" * 80)
    print(f"   📁 Checkpoints : {args.dossier_checkpoint}")
    print(f"   📊 Historique  : {args.dossier_checkpoint}/historique_entrainement.json")
    print(f"   📝 Log         : entrainement.log")
    
    if historique["val_loss"]:
        print(f"\n   Meilleure val_loss : {min(historique['val_loss']):.4f}")
        print(f"   Meilleure val_acc  : {max(historique['val_acc']):.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())