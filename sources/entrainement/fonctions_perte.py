"""
fonctions_perte.py
───────────────────
Fonctions de perte spécialisées pour la classification de tweets de crise.

Stratégies :
    1. Focal Loss        — Réduit l'impact des exemples faciles (classes majoritaires)
    2. Weighted CE       — CrossEntropy pondérée par les poids de classes
    3. Focal + Weighted  — Combine les deux pour un déséquilibre extrême
    4. Label Smoothing   — Régularisation contre l'overfitting

Utilisation :
    loss_fn = creer_fonction_perte(type_perte="focal_weighted")
    loss = loss_fn(logits, cibles)
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  FOCAL LOSS
# ══════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss pour le déséquilibre de classes.
    
    FL(p) = -α * (1 - p)^γ * log(p)
    
    Où :
        p   = probabilité prédite pour la vraie classe
        α   = facteur de pondération des classes (alpha)
        γ   = facteur de focalisation (gamma)
    
    Plus γ est grand, plus on réduit la contribution des exemples faciles.
    
    Args:
        alpha: Poids par classe [nb_classes] ou float
        gamma: Facteur de focalisation (défaut : 2.0)
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha          # Poids par classe
        self.gamma = gamma          # Focalisation
        self.reduction = reduction
        
        logger.info(f"🎯 FocalLoss initialisée : γ={gamma}, α={'oui' if alpha is not None else 'non'}")
    
    def forward(
        self,
        logits: torch.Tensor,
        cibles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la Focal Loss.
        
        Args:
            logits: [B, nb_classes] logits bruts (avant softmax)
            cibles: [B] indices des classes
        
        Returns:
            Perte scalaire (ou par élément si reduction='none')
        """
        # Calculer les probabilités
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Probabilité de la vraie classe
        probs_cibles = probs.gather(1, cibles.unsqueeze(1)).squeeze(1)  # [B]
        log_probs_cibles = log_probs.gather(1, cibles.unsqueeze(1)).squeeze(1)  # [B]
        
        # Facteur de focalisation : (1 - p)^gamma
        focal_weight = (1 - probs_cibles) ** self.gamma
        
        # Appliquer alpha si défini
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_cibles = self.alpha.gather(0, cibles)  # [B]
            focal_weight = alpha_cibles * focal_weight
        
        # Calculer la loss
        loss = -focal_weight * log_probs_cibles
        
        # Réduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ══════════════════════════════════════════
# 2.  FOCAL LOSS PONDÉRÉE (FOCAL + WEIGHTS)
# ══════════════════════════════════════════

class FocalLossWeighted(nn.Module):
    """
    Focal Loss avec pondération automatique des classes.
    
    Combine :
        • Focal Loss (gamma)
        • Poids inverses à la fréquence des classes
        • Option de label smoothing
    
    Args:
        nb_classes: Nombre de classes
        gamma: Facteur de focalisation
        poids_classes: Poids par classe (si None, calculé automatiquement)
        label_smoothing: Taux de label smoothing
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(
        self,
        nb_classes: int = 3,
        gamma: float = 2.0,
        poids_classes: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.nb_classes = nb_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        # Poids des classes
        if poids_classes is not None:
            self.poids_classes = torch.tensor(poids_classes)
        else:
            self.poids_classes = torch.tensor(cfg.classes.poids)
        
        logger.info(f"🎯 FocalLossWeighted : γ={gamma}, smoothing={label_smoothing}")
        logger.info(f"   Poids : {self.poids_classes.tolist()}")
    
    def forward(
        self,
        logits: torch.Tensor,
        cibles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, nb_classes]
            cibles: [B]
        
        Returns:
            Perte scalaire
        """
        B = logits.size(0)
        
        # Déplacer les poids sur le bon device
        if self.poids_classes.device != logits.device:
            self.poids_classes = self.poids_classes.to(logits.device)
        
        # ── Label Smoothing ──
        if self.label_smoothing > 0:
            # One-hot lissé : (1 - ε) * one_hot + ε / nb_classes
            cibles_one_hot = torch.zeros(B, self.nb_classes, device=logits.device)
            cibles_one_hot.scatter_(1, cibles.unsqueeze(1), 1.0)
            cibles_one_hot = cibles_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / self.nb_classes
        else:
            cibles_one_hot = F.one_hot(cibles, num_classes=self.nb_classes).float()
        
        # ── Probabilités ──
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # ── Focal Loss par classe ──
        focal_weight = (1 - probs) ** self.gamma  # [B, nb_classes]
        
        # ── CrossEntropy pondérée ──
        ce = -cibles_one_hot * log_probs  # [B, nb_classes]
        
        # Appliquer les poids de classes et focal
        poids = self.poids_classes.unsqueeze(0).expand(B, -1)  # [B, nb_classes]
        loss = poids * focal_weight * ce  # [B, nb_classes]
        
        # Sommer sur les classes
        loss = loss.sum(dim=-1)  # [B]
        
        # Réduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ══════════════════════════════════════════
# 3.  CROSS ENTROPY PONDÉRÉE
# ══════════════════════════════════════════

class WeightedCrossEntropy(nn.Module):
    """
    CrossEntropy simple avec poids de classes.
    Utile comme baseline.
    """
    
    def __init__(
        self,
        poids_classes: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        if poids_classes is not None:
            self.poids = torch.tensor(poids_classes)
        else:
            self.poids = torch.tensor(cfg.classes.poids)
        
        self.label_smoothing = label_smoothing
        
        logger.info(f"📊 WeightedCrossEntropy : smoothing={label_smoothing}")
        logger.info(f"   Poids : {self.poids.tolist()}")
    
    def forward(
        self,
        logits: torch.Tensor,
        cibles: torch.Tensor,
    ) -> torch.Tensor:
        if self.poids.device != logits.device:
            self.poids = self.poids.to(logits.device)
        
        return F.cross_entropy(
            logits,
            cibles,
            weight=self.poids,
            label_smoothing=self.label_smoothing,
        )


# ══════════════════════════════════════════
# 4.  FONCTIONS DE PERTE COMBINÉES
# ══════════════════════════════════════════

class PerteMixte(nn.Module):
    """
    Combine Focal Loss + une perte auxiliaire.
    
    Utile pour l'apprentissage multi-tâche ou la régularisation.
    
    loss = λ1 * focal_loss + λ2 * perte_auxiliaire
    """
    
    def __init__(
        self,
        perte_principale: nn.Module,
        lambda_principale: float = 1.0,
        perte_auxiliaire: Optional[nn.Module] = None,
        lambda_auxiliaire: float = 0.1,
    ):
        super().__init__()
        self.perte_principale = perte_principale
        self.lambda_principale = lambda_principale
        self.perte_auxiliaire = perte_auxiliaire
        self.lambda_auxiliaire = lambda_auxiliaire
    
    def forward(
        self,
        logits: torch.Tensor,
        cibles: torch.Tensor,
        logits_aux: Optional[torch.Tensor] = None,
        cibles_aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.lambda_principale * self.perte_principale(logits, cibles)
        
        if self.perte_auxiliaire is not None and logits_aux is not None:
            loss += self.lambda_auxiliaire * self.perte_auxiliaire(logits_aux, cibles_aux)
        
        return loss


# ══════════════════════════════════════════
# 5.  FACTORY DE FONCTIONS DE PERTE
# ══════════════════════════════════════════

def creer_fonction_perte(
    type_perte: str = "focal_weighted",
    nb_classes: int = 3,
    gamma: Optional[float] = None,
    poids_classes: Optional[List[float]] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> nn.Module:
    """
    Fabrique la fonction de perte appropriée.
    
    Args:
        type_perte: 'focal', 'focal_weighted', 'weighted_ce', 'ce'
        nb_classes: Nombre de classes
        gamma: Facteur de focalisation (défaut : cfg.entrainement.focal_gamma)
        poids_classes: Poids par classe (défaut : cfg.classes.poids)
        label_smoothing: Taux de label smoothing
        reduction: 'mean' | 'sum' | 'none'
    
    Returns:
        Module de perte
    """
    if gamma is None:
        gamma = cfg.entrainement.focal_gamma
    
    if poids_classes is None:
        poids_classes = cfg.classes.poids
    
    mapping = {
        "focal": lambda: FocalLoss(
            alpha=torch.tensor(poids_classes),
            gamma=gamma,
            reduction=reduction,
        ),
        "focal_weighted": lambda: FocalLossWeighted(
            nb_classes=nb_classes,
            gamma=gamma,
            poids_classes=poids_classes,
            label_smoothing=label_smoothing,
            reduction=reduction,
        ),
        "weighted_ce": lambda: WeightedCrossEntropy(
            poids_classes=poids_classes,
            label_smoothing=label_smoothing,
        ),
        "ce": lambda: nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction,
        ),
    }
    
    if type_perte not in mapping:
        logger.warning(f"⚠️  Type de perte '{type_perte}' inconnu. Utilisation de 'focal_weighted'.")
        type_perte = "focal_weighted"
    
    perte = mapping[type_perte]()
    logger.info(f"✅ Fonction de perte créée : {type_perte}")
    return perte


# ══════════════════════════════════════════
# 6.  UTILITAIRES DE MONITORING
# ══════════════════════════════════════════

class MoniteurPerte:
    """
    Suivi des pertes par classe pendant l'entraînement.
    Utile pour détecter le surapprentissage ou l'oubli catastrophique.
    """
    
    def __init__(self, nb_classes: int = 4):
        self.nb_classes = nb_classes
        self.reset()
    
    def reset(self):
        self.perte_par_classe = {i: [] for i in range(self.nb_classes)}
        self.perte_globale = []
    
    def update(
        self,
        logits: torch.Tensor,
        cibles: torch.Tensor,
        perte_fn: nn.Module,
    ):
        """Enregistre la perte globale et par classe."""
        with torch.no_grad():
            # Perte globale
            perte = perte_fn(logits, cibles)
            self.perte_globale.append(perte.item())
            
            # Perte par classe
            for classe in range(self.nb_classes):
                mask = (cibles == classe)
                if mask.any():
                    perte_classe = perte_fn(logits[mask], cibles[mask])
                    self.perte_par_classe[classe].append(perte_classe.item())
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de perte."""
        import numpy as np
        
        stats = {
            "globale": {
                "mean": np.mean(self.perte_globale),
                "std": np.std(self.perte_globale),
                "min": np.min(self.perte_globale),
            }
        }
        
        for classe in range(self.nb_classes):
            if self.perte_par_classe[classe]:
                stats[f"classe_{classe}"] = {
                    "mean": np.mean(self.perte_par_classe[classe]),
                    "nb_echantillons": len(self.perte_par_classe[classe]),
                }
        
        return stats
    
    def afficher(self):
        """Affiche un résumé des pertes."""
        stats = self.get_stats()
        print(f"\n📊 Moniteur de perte :")
        print(f"   Globale : mean={stats['globale']['mean']:.4f} ± {stats['globale']['std']:.4f}")
        for classe in range(self.nb_classes):
            if f"classe_{classe}" in stats:
                s = stats[f"classe_{classe}"]
                print(f"   Classe {classe} : mean={s['mean']:.4f} ({s['nb_echantillons']} ex.)")


# ══════════════════════════════════════════
# 7.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🧪 TEST — Fonctions de Perte")
    print("=" * 70)
    
    # Créer des données factices déséquilibrées
    B = 32
    nb_classes = 4
    
    # Simuler un batch déséquilibré (beaucoup de classe 2, peu de classe 3)
    cibles = torch.tensor([0]*3 + [1]*8 + [2]*18 + [3]*3)  # Total 32
    
    # Simuler des logits (légèrement corrélés aux cibles)
    torch.manual_seed(42)
    logits = torch.randn(B, nb_classes)
    logits[range(B), cibles] += 2.0  # Boost pour la vraie classe
    
    print(f"\n📊 Distribution du batch :")
    for c in range(nb_classes):
        print(f"   Classe {c} : {(cibles == c).sum().item()} exemples")
    
    # ── Test 1 : CrossEntropy simple ──
    print("\n📌 Test 1 — CrossEntropy simple")
    ce = nn.CrossEntropyLoss()
    loss_ce = ce(logits, cibles)
    print(f"   Loss CE simple : {loss_ce.item():.4f}")
    
    # ── Test 2 : Focal Loss ──
    print("\n📌 Test 2 — Focal Loss")
    focal = creer_fonction_perte("focal", gamma=2.0)
    loss_focal = focal(logits, cibles)
    print(f"   Loss Focal : {loss_focal.item():.4f}")
    
    # ── Test 3 : Focal Loss Weighted ──
    print("\n📌 Test 3 — Focal Loss Weighted")
    focal_w = creer_fonction_perte("focal_weighted", gamma=2.0)
    loss_focal_w = focal_w(logits, cibles)
    print(f"   Loss Focal Weighted : {loss_focal_w.item():.4f}")
    
    # ── Test 4 : Weighted CrossEntropy ──
    print("\n📌 Test 4 — Weighted CrossEntropy")
    wce = creer_fonction_perte("weighted_ce")
    loss_wce = wce(logits, cibles)
    print(f"   Loss WCE : {loss_wce.item():.4f}")
    
    # ── Test 5 : Comparaison ──
    print("\n📌 Test 5 — Comparaison des pertes")
    pertes = {
        "CE simple": loss_ce.item(),
        "Focal": loss_focal.item(),
        "Focal Weighted": loss_focal_w.item(),
        "Weighted CE": loss_wce.item(),
    }
    for nom, val in sorted(pertes.items(), key=lambda x: x[1]):
        print(f"   {nom:<20} : {val:.4f}")
    
    # ── Test 6 : Label Smoothing ──
    print("\n📌 Test 6 — Avec Label Smoothing")
    focal_smooth = creer_fonction_perte("focal_weighted", label_smoothing=0.1)
    loss_smooth = focal_smooth(logits, cibles)
    print(f"   Loss avec smoothing : {loss_smooth.item():.4f}")
    
    # ── Test 7 : Moniteur de perte ──
    print("\n📌 Test 7 — Moniteur de perte")
    moniteur = MoniteurPerte(nb_classes=3)
    for _ in range(5):
        moniteur.update(logits, cibles, focal_w)
    moniteur.afficher()
    
    # ── Test 8 : Vérification gradients ──
    print("\n📌 Test 8 — Vérification backpropagation")
    logits_grad = torch.randn(8, 4, requires_grad=True)
    cibles_grad = torch.randint(0, 4, (8,))
    loss_grad = focal_w(logits_grad, cibles_grad)
    loss_grad.backward()
    print(f"   Gradient : {'✓' if logits_grad.grad is not None else '✗'}")
    print(f"   Norm grad : {logits_grad.grad.norm().item():.4f}")
    
    # ── Test 9 : Cohérence des pertes ──
    print("\n📌 Test 9 — Test de cohérence")
    # Si logits = [100, 0, 0, 0] et cible = 0, la perte doit être ~0
    logits_confiants = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    cible_confiante = torch.tensor([0])
    loss_confiante = focal_w(logits_confiants, cible_confiante)
    print(f"   Perte exemple facile (p≈1) : {loss_confiante.item():.6f}")
    
    # Si logits = [0, 0, 0, 0] et cible = 0, la perte doit être > 0
    logits_incertains = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    cible_incertaine = torch.tensor([0])
    loss_incertaine = focal_w(logits_incertains, cible_incertaine)
    print(f"   Perte exemple difficile (p≈0.25) : {loss_incertaine.item():.4f}")
    print(f"   Ratio difficile/facile : {loss_incertaine.item() / max(loss_confiante.item(), 1e-8):.1f}x")
    
    print(f"\n✅ fonctions_perte.py — Tout OK !")