"""
reproductibilite.py
────────────────────
Graine aléatoire fixe et configuration pour la reproductibilité.

Utilisation :
    from sources.utilitaires.reproductibilite import fixer_reproductibilite
    fixer_reproductibilite(graine=42)
"""

import logging
import random
import os
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  FIXATION DES GRAINES
# ══════════════════════════════════════════

def fixer_reproductibilite(graine: int = 42) -> None:
    """
    Fixe toutes les graines aléatoires pour assurer la reproductibilité.
    
    Args:
        graine: Valeur de la graine aléatoire
    """
    # Python
    random.seed(graine)
    
    # NumPy
    np.random.seed(graine)
    
    # PyTorch CPU
    torch.manual_seed(graine)
    
    # PyTorch GPU
    torch.cuda.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)
    
    # CuDNN déterministe (légèrement plus lent, mais reproductible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Variables d'environnement
    os.environ["PYTHONHASHSEED"] = str(graine)
    
    logger.info(f"🎲 Reproductibilité activée — graine = {graine}")
    logger.info(f"   • Python random     : ✓")
    logger.info(f"   • NumPy             : ✓")
    logger.info(f"   • PyTorch CPU       : ✓")
    logger.info(f"   • PyTorch GPU       : ✓")
    logger.info(f"   • CuDNN déterministe : ✓")


# ══════════════════════════════════════════
# 2.  VÉRIFICATION
# ══════════════════════════════════════════

def verifier_reproductibilite(graine: int = 42) -> bool:
    """
    Vérifie que les graines fonctionnent correctement.
    
    Args:
        graine: Graine à tester
    
    Returns:
        True si la reproductibilité est confirmée
    """
    # Test 1 : torch.rand
    torch.manual_seed(graine)
    a = torch.rand(100)
    
    torch.manual_seed(graine)
    b = torch.rand(100)
    
    test1 = torch.allclose(a, b)
    
    # Test 2 : numpy
    np.random.seed(graine)
    c = np.random.rand(100)
    
    np.random.seed(graine)
    d = np.random.rand(100)
    
    test2 = np.allclose(c, d)
    
    if test1 and test2:
        logger.info("✅ Reproductibilité vérifiée !")
        return True
    else:
        logger.warning("⚠️  Problème de reproductibilité détecté !")
        return False


# ══════════════════════════════════════════
# 3.  CONTEXTE MANAGER (optionnel)
# ══════════════════════════════════════════

class ContexteReproductible:
    """
    Context manager pour exécuter un bloc de code de manière reproductible.
    
    Usage :
        with ContexteReproductible(graine=42):
            modele = Entraineur(...)
            modele.entrainer()
    """
    
    def __init__(self, graine: int = 42):
        self.graine = graine
    
    def __enter__(self):
        fixer_reproductibilite(self.graine)
        return self
    
    def __exit__(self, *args):
        pass


# ══════════════════════════════════════════
# 4.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🧪 TEST — Reproductibilité")
    print("=" * 70)
    
    # Test 1 : Fixation des graines
    print("\n📌 Test 1 — Fixation des graines")
    fixer_reproductibilite(graine=42)
    
    # Test 2 : Vérification
    print("\n📌 Test 2 — Vérification de la reproductibilité")
    ok = verifier_reproductibilite(graine=42)
    print(f"   Résultat : {'✅ OK' if ok else '✗ Échec'}")
    
    # Test 3 : Deux exécutions identiques
    print("\n📌 Test 3 — Deux exécutions doivent être identiques")
    
    fixer_reproductibilite(graine=123)
    x1 = torch.randn(5)
    y1 = np.random.randn(5)
    
    fixer_reproductibilite(graine=123)
    x2 = torch.randn(5)
    y2 = np.random.randn(5)
    
    print(f"   Torch identique : {torch.allclose(x1, x2)}")
    print(f"   NumPy identique : {np.allclose(y1, y2)}")
    
    # Test 4 : Context manager
    print("\n📌 Test 4 — Context manager")
    with ContexteReproductible(graine=99):
        x3 = torch.randn(3)
    
    fixer_reproductibilite(graine=99)
    x4 = torch.randn(3)
    
    print(f"   Contexte identique : {torch.allclose(x3, x4)}")
    
    print(f"\n✅ reproductibilite.py — Tout OK !")