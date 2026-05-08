"""
chargeur_donnees.py
────────────────────
DataLoader multimodal unifié — charge tout en une seule ligne.

Combine :
    • Chargement des CSV (train/val/test)
    • Création des Datasets PyTorch
    • Application du NettoyeurTweet
    • DataLoaders avec collate_fn_multimodal

Utilisation :
    from sources.donnees.chargeur_donnees import creer_data_loaders
    train_loader, val_loader, test_loader = creer_data_loaders(batch_size=16)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

from torch.utils.data import DataLoader

from sources.donnees.jeu_de_donnees import (
    charger_splits_csv,
    preparer_donnees,
    JeuDeDonneesCrise,
    NOMS_CLASSES,
)
from sources.donnees.traitement_texte import NettoyeurTweet
from sources.entrainement.boucle_entrainement import collate_fn_multimodal
from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  FONCTION PRINCIPALE
# ══════════════════════════════════════════

def creer_data_loaders(
    batch_size: int = 32,
    nb_workers: int = 4,
    dossier_csv: Optional[Path] = None,
    proportions: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    graine: int = 42,
    utiliser_cache: bool = True,
    mettre_minuscules: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les 3 DataLoaders (train, val, test) en une seule ligne.
    
    Étapes :
        1. Vérifie si train/val/test.csv existent
        2. Si non, exécute preparer_donnees() pour les créer
        3. Charge les CSV
        4. Crée le NettoyeurTweet
        5. Crée les 3 JeuDeDonneesCrise
        6. Crée les 3 DataLoaders avec collate_fn_multimodal
    
    Args:
        batch_size: Taille des batches
        nb_workers: Nombre de workers pour DataLoader
        dossier_csv: Dossier contenant les CSV
        proportions: (train, val, test) si création nécessaire
        graine: Graine aléatoire pour la reproductibilité
        utiliser_cache: Utiliser le cache CrisisMMD
        mettre_minuscules: Mettre le texte en minuscules
    
    Returns:
        Tuple (train_loader, val_loader, test_loader)
    
    Example:
        train_loader, val_loader, test_loader = creer_data_loaders(
            batch_size=16,
            nb_workers=2,
        )
        
        for batch in train_loader:
            textes = batch["texte"]
            images = batch["image"]
            classes = batch["classe"]
    """
    logger.info("=" * 60)
    logger.info("📦 CRÉATION DES DATA LOADERS")
    logger.info("=" * 60)
    
    # ── 1. Vérifier/créer les CSV ──
    if dossier_csv is None:
        dossier_csv = cfg.chemins.dossier_traitees
    
    chemin_train = dossier_csv / "train.csv"
    chemin_val   = dossier_csv / "val.csv"
    chemin_test  = dossier_csv / "test.csv"
    
    if not (chemin_train.exists() and chemin_val.exists() and chemin_test.exists()):
        logger.info("📂 CSV non trouvés → Création des splits...")
        preparer_donnees(
            proportions=proportions,
            graine=graine,
            sauvegarder_csv=True,
            dossier_csv=dossier_csv,
        )
    else:
        logger.info("📂 CSV trouvés → Chargement direct")
    
    # ── 2. Charger les CSV ──
    df_train, df_val, df_test = charger_splits_csv(dossier_csv)
    
    # ── 3. Créer le NettoyeurTweet ──
    nettoyeur = NettoyeurTweet(
        supprimer_urls=cfg.texte.supprimer_urls,
        remplacer_mentions=cfg.texte.remplacer_mentions,
        normaliser_hashtags=cfg.texte.normaliser_hashtags,
        expandre_abreviations=cfg.texte.expandre_abreviations,
        remplacer_nombres=cfg.texte.remplacer_nombres,
        reduire_repetitions=cfg.texte.reduire_repetitions,
        mettre_minuscules=mettre_minuscules,
    )
    
    # ── 4. Créer les Datasets ──
    logger.info("📊 Création des Datasets PyTorch...")
    dataset_train = JeuDeDonneesCrise(df_train, nettoyeur, utiliser_texte_nettoye=True)
    dataset_val   = JeuDeDonneesCrise(df_val, nettoyeur, utiliser_texte_nettoye=True)
    dataset_test  = JeuDeDonneesCrise(df_test, nettoyeur, utiliser_texte_nettoye=True)
    
    # ── 5. Créer les DataLoaders ──
    logger.info(f"📦 Création des DataLoaders (batch_size={batch_size}, workers={nb_workers})...")
    
    # Ajuster les workers selon l'OS
    if nb_workers > 0:
        import os
        if os.name == 'nt':  # Windows
            nb_workers = min(nb_workers, 0)  # Windows a des problèmes avec >0
            if nb_workers == 0:
                logger.info("   ℹ️  Windows détecté → nb_workers=0 (évite les erreurs)")
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_multimodal,
        num_workers=nb_workers,
        pin_memory=True if nb_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_multimodal,
        num_workers=nb_workers,
        pin_memory=True if nb_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_multimodal,
        num_workers=nb_workers,
        pin_memory=True if nb_workers > 0 else False,
    )
    
    # ── 6. Résumé ──
    logger.info("✅ DataLoaders créés avec succès !")
    logger.info(f"   • Train : {len(train_loader)} batches × {batch_size} = ~{len(train_loader) * batch_size} ex.")
    logger.info(f"   • Val   : {len(val_loader)} batches × {batch_size} = ~{len(val_loader) * batch_size} ex.")
    logger.info(f"   • Test  : {len(test_loader)} batches × {batch_size} = ~{len(test_loader) * batch_size} ex.")
    
    # Distribution
    for nom, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        dataset = loader.dataset
        dist = dataset.get_stats()["distribution"]
        logger.info(f"   {nom} — Distribution :")
        for cl, nb in sorted(dist.items()):
            logger.info(f"      Classe {cl} ({NOMS_CLASSES[cl]}): {nb}")
    
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════
# 2.  FONCTIONS SPÉCIFIQUES
# ══════════════════════════════════════════

def creer_train_loader(
    batch_size: int = 32,
    nb_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Crée uniquement le DataLoader d'entraînement."""
    train_loader, _, _ = creer_data_loaders(
        batch_size=batch_size,
        nb_workers=nb_workers,
        **kwargs,
    )
    return train_loader


def creer_val_loader(
    batch_size: int = 32,
    nb_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Crée uniquement le DataLoader de validation."""
    _, val_loader, _ = creer_data_loaders(
        batch_size=batch_size,
        nb_workers=nb_workers,
        **kwargs,
    )
    return val_loader


def creer_test_loader(
    batch_size: int = 32,
    nb_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Crée uniquement le DataLoader de test."""
    _, _, test_loader = creer_data_loaders(
        batch_size=batch_size,
        nb_workers=nb_workers,
        **kwargs,
    )
    return test_loader


# ══════════════════════════════════════════
# 3.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🧪 TEST — Chargeur de Données Unifié")
    print("=" * 70)
    
    # ── Test 1 : Création des DataLoaders ──
    print("\n📌 Test 1 — Création des DataLoaders")
    try:
        train_loader, val_loader, test_loader = creer_data_loaders(
            batch_size=16,
            nb_workers=0,  # 0 pour Windows
        )
    except Exception as e:
        print(f"   ⚠️  Erreur : {e}")
        print("   → Test avec données factices...")
        
        # Fallback : dataset factice pour test
        class DatasetFactice(torch.utils.data.Dataset):
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return {
                    "texte": f"test {idx}",
                    "image": None,
                    "classe": idx % 4,
                    "source": "test",
                    "catastrophe": "test",
                    "idx": idx,
                }
        
        import torch
        dataset = DatasetFactice()
        train_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn_multimodal)
        val_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn_multimodal)
        test_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn_multimodal)
    
    # ── Test 2 : Vérification d'un batch ──
    print("\n📌 Test 2 — Inspection d'un batch")
    batch = next(iter(train_loader))
    print(f"   Clés du batch : {list(batch.keys())}")
    print(f"   Texte  : {len(batch['texte'])} éléments, type={type(batch['texte'][0])}")
    print(f"   Image  : {len(batch['image'])} éléments")
    print(f"   Classe : {batch['classe'].shape}, valeurs={batch['classe'].tolist()}")
    print(f"   Source : {batch['source'][:3]}")
    
    # ── Test 3 : Itération complète ──
    print("\n📌 Test 3 — Itération sur 3 batches")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"   Batch {i+1} : {len(batch['texte'])} textes, {batch['classe'].shape[0]} classes")
    
    print(f"\n✅ chargeur_donnees.py — Tout OK !")
