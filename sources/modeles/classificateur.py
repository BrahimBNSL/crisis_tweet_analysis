"""
classificateur.py
─────────────────
Tête de classification finale pour les tweets de crise.

Architecture :
    Fusion [256] → Dense(256) → BatchNorm → GELU → Dropout(0.3)
                 → Dense(128) → BatchNorm → GELU → Dropout(0.3)
                 → Dense(4)   → Softmax

Utilisation :
    classificateur = ClassificateurCrise()
    logits = classificateur(embedding_fusion)
    # → torch.Tensor de forme [B, 4]
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  CLASSIFICATEUR
# ══════════════════════════════════════════

class ClassificateurCrise(nn.Module):
    """
    Tête de classification avec couches denses + normalisation + dropout.
    
    Args:
        dim_entree: Dimension d'entrée (défaut : cfg.fusion.dim_sortie = 256)
        nb_classes: Nombre de classes (défaut : cfg.classes.nb_classes = 4)
        couches_cachees: Liste des dimensions cachées (défaut : [256, 128])
        dropout: Taux de dropout
        activation: 'relu', 'gelu', 'silu'
        utiliser_batchnorm: Si True, ajoute BatchNorm1d
    """
    
    def __init__(
        self,
        dim_entree: Optional[int] = None,
        nb_classes: Optional[int] = None,
        couches_cachees: Optional[List[int]] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        utiliser_batchnorm: bool = True,
    ):
        super().__init__()
        
        # Configuration
        self.dim_entree = dim_entree or cfg.fusion.dim_sortie  # 256
        self.nb_classes = nb_classes or cfg.classes.nb_classes  # 4
        self.couches_cachees = couches_cachees or cfg.classificateur.couches_denses  # [256, 128]
        self.dropout = dropout if dropout is not None else 0.5  # 0.3
        self.activation_str = activation or cfg.classificateur.activation  # 'gelu'
        self.utiliser_batchnorm = utiliser_batchnorm
        
        # Choisir la fonction d'activation
        self.activation = self._get_activation(self.activation_str)
        
        # ── Construire les couches ──
        couches = []
        dim_entree_couche = self.dim_entree
        
        for i, dim_sortie_couche in enumerate(self.couches_cachees):
            # Dense
            couches.append(nn.Linear(dim_entree_couche, dim_sortie_couche))
            
            # BatchNorm
            if self.utiliser_batchnorm:
                couches.append(nn.BatchNorm1d(dim_sortie_couche))
            
            # Activation
            couches.append(self.activation)
            
            # Dropout
            couches.append(nn.Dropout(self.dropout))
            
            dim_entree_couche = dim_sortie_couche
        
        # Couche de sortie (logits)
        couches.append(nn.Linear(dim_entree_couche, self.nb_classes))
        
        self.reseau = nn.Sequential(*couches)
        
        # ── Poids des classes pour la loss ──
        self.poids_classes = torch.tensor(cfg.classes.poids)
        
        # Stats
        nb_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✅ ClassificateurCrise initialisé :")
        logger.info(f"   • Entrée     : {self.dim_entree}")
        logger.info(f"   • Cachées    : {self.couches_cachees}")
        logger.info(f"   • Sortie     : {self.nb_classes}")
        logger.info(f"   • Activation : {self.activation_str}")
        logger.info(f"   • Dropout    : {self.dropout}")
        logger.info(f"   • BatchNorm  : {self.utiliser_batchnorm}")
        logger.info(f"   • Params     : {nb_params:,}")
    
    # ──────────────────────────────────────
    @staticmethod
    def _get_activation(nom: str) -> nn.Module:
        """Retourne le module d'activation correspondant."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "elu": nn.ELU(inplace=True),
        }
        if nom.lower() not in activations:
            logger.warning(f"⚠️  Activation '{nom}' inconnue. Utilisation de GELU.")
            return nn.GELU()
        return activations[nom.lower()]
    
    # ──────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Passe avant : embedding → logits ou probabilités.
        
        Args:
            x: Tenseur [B, dim_entree]
            return_probs: Si True, applique softmax
        
        Returns:
            logits [B, nb_classes] ou probas [B, nb_classes]
        """
        logits = self.reseau(x)
        
        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits
    
    # ─────────────────────────────────────
        
    def predire(self, x, seuil_urgence: float = 0.35):
     """Prédit avec seuil ajustable pour Urgence."""
     with torch.no_grad():
        probas = self.forward(x, return_probs=True)
        B = probas.size(0)
        
        # Seuil pour Urgence, puis argmax entre Info/Non pertinent
        p_urgence = probas[:, 0]
        
        classes = torch.where(
            p_urgence > seuil_urgence,
            torch.zeros(B, dtype=torch.long, device=probas.device),  # Urgence
            torch.argmax(probas[:, 1:], dim=-1) + 1  # Info ou Non pertinent
        )
        return classes, probas    

    # ──────────────────────────────────────
    def get_poids_classes(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Retourne les poids des classes pour la loss.
        Utile pour Focal Loss ou CrossEntropy pondérée.
        """
        return self.poids_classes.to(device)


# ══════════════════════════════════════════
# 2.  MODÈLE COMPLET (fusion + classification)
# ══════════════════════════════════════════

class ModeleComplet(nn.Module):
    """
    Modèle complet : Fusion + Classification.
    
    Combine FusionCrossModule et ClassificateurCrise en un seul module.
    Pratique pour l'entraînement de bout en bout.
    
    Args:
        fusion: Module de fusion cross-modale
        classificateur: Tête de classification
    """
    
    def __init__(
        self,
        fusion: nn.Module,
        classificateur: ClassificateurCrise,
    ):
        super().__init__()
        self.fusion = fusion
        self.classificateur = classificateur
    
    def forward(
        self,
        emb_texte: torch.Tensor,
        emb_image: torch.Tensor,
        return_probs: bool = False,
        return_poids: bool = False,
    ) -> torch.Tensor:
        """
        Embeddings texte + image → logits/probas.
        
        Args:
            emb_texte: [B, 768]
            emb_image: [B, 2048]
            return_probs: Appliquer softmax
            return_poids: Retourner les poids du gate
        
        Returns:
            logits [B, 4] ou (logits, poids_gate)
        """
        if return_poids:
            fusion_out, poids_gate = self.fusion(emb_texte, emb_image, return_poids=True)
            logits = self.classificateur(fusion_out, return_probs=return_probs)
            return logits, poids_gate
        
        fusion_out = self.fusion(emb_texte, emb_image)
        logits = self.classificateur(fusion_out, return_probs=return_probs)
        return logits
    
    def predire(
        self,
        emb_texte: torch.Tensor,
        emb_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit la classe pour un batch.
        
        Returns:
            classes: [B]
            probas: [B, 4]
        """
        fusion_out = self.fusion(emb_texte, emb_image)
        return self.classificateur.predire(fusion_out)


# ══════════════════════════════════════════
# 3.  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def creer_classificateur(
    dim_entree: Optional[int] = None,
    nb_classes: Optional[int] = None,
) -> ClassificateurCrise:
    """Fabrique un classificateur avec la configuration centrale."""
    return ClassificateurCrise(
        dim_entree=dim_entree,
        nb_classes=nb_classes,
    )


def creer_modele_complet(
    fusion: Optional[nn.Module] = None,
    classificateur: Optional[ClassificateurCrise] = None,
) -> ModeleComplet:
    """
    Crée le modèle complet (fusion + classification).
    
    Si fusion ou classificateur ne sont pas fournis, ils sont créés
    avec la configuration par défaut.
    """
    from sources.modeles.fusion_crossmodale import FusionCrossModule
    
    if fusion is None:
        fusion = FusionCrossModule()
    if classificateur is None:
        classificateur = ClassificateurCrise()
    
    return ModeleComplet(fusion, classificateur)


def resume_modele(modele: ModeleComplet) -> None:
    """Affiche un résumé du modèle complet."""
    nb_params_total = sum(p.numel() for p in modele.parameters())
    nb_params_train = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    
    print(f"\n📋 Résumé du modèle complet :")
    print(f"   ═══════════════════════════════════")
    print(f"   Fusion       : {sum(p.numel() for p in modele.fusion.parameters()):,} params")
    print(f"   Classifieur  : {sum(p.numel() for p in modele.classificateur.parameters()):,} params")
    print(f"   ───────────────────────────────────")
    print(f"   Total        : {nb_params_total:,} params")
    print(f"   Entraînables : {nb_params_train:,} params")


# ══════════════════════════════════════════
# 4.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🧪 TEST — Classificateur + Modèle Complet")
    print("=" * 70)
    
    # ── Test 1 : Création du classificateur ──
    print("\n📌 Test 1 — Création du classificateur")
    classificateur = creer_classificateur()
    
    # ── Test 2 : Forward ──
    print("\n📌 Test 2 — Forward (logits)")
    B = 4
    x = torch.randn(B, 256)  # Simule la sortie de fusion
    logits = classificateur(x)
    print(f"   Entrée  : {x.shape}")
    print(f"   Logits  : {logits.shape}")
    print(f"   Logits  :\n{logits}")
    
    # ── Test 3 : Probabilités ──
    print("\n📌 Test 3 — Probabilités (softmax)")
    probas = classificateur(x, return_probs=True)
    print(f"   Probas  : {probas.shape}")
    print(f"   Probas  :\n{probas}")
    print(f"   Somme par ligne : {probas.sum(dim=1)}")
    
    # ── Test 4 : Prédiction ──
    print("\n📌 Test 4 — Prédiction")
    classes, probas = classificateur.predire(x)
    print(f"   Classes prédites : {classes}")
    print(f"   Confiance max    : {probas.max(dim=1)[0]}")
    
    # ── Test 5 : Poids des classes ──
    print("\n📌 Test 5 — Poids des classes")
    poids = classificateur.get_poids_classes()
    for i, nom in enumerate(cfg.classes.noms):
        print(f"   Classe {i} ({nom}): poids = {poids[i]:.1f}")
    
    # ── Test 6 : Modèle complet ──
    print("\n📌 Test 6 — Modèle complet (Fusion + Classification)")
    modele_complet = creer_modele_complet()
    emb_texte = torch.randn(B, 768)
    emb_image = torch.randn(B, 2048)
    
    logits_complet = modele_complet(emb_texte, emb_image)
    print(f"   Texte entrée : {emb_texte.shape}")
    print(f"   Image entrée : {emb_image.shape}")
    print(f"   Logits       : {logits_complet.shape}")
    
    # ── Test 7 : Prédiction complète ──
    print("\n📌 Test 7 — Prédiction bout en bout")
    classes_complet, probas_complet = modele_complet.predire(emb_texte, emb_image)
    print(f"   Classes prédites : {classes_complet}")
    for i in range(B):
        top2 = torch.topk(probas_complet[i], 2)
        print(f"   [{i}] Top-2 : classes={top2.indices.tolist()}, probs={top2.values.tolist()}")
    
    # ── Test 8 : Résumé du modèle ──
    print("\n📌 Test 8 — Résumé du modèle complet")
    resume_modele(modele_complet)
    
    # ── Test 9 : Backpropagation ──
    print("\n📌 Test 9 — Vérification backpropagation")
    x_grad = torch.randn(2, 256, requires_grad=True)
    logits_grad = classificateur(x_grad)
    loss = logits_grad.sum()
    loss.backward()
    print(f"   Gradient entrée : {'✓' if x_grad.grad is not None else '✗'}")
    print(f"   Norm gradient   : {x_grad.grad.norm().item():.4f}")
    
    # ── Test 10 : GPU ──
    print("\n📌 Test 10 — Test GPU")
    if torch.cuda.is_available():
        classificateur_gpu = classificateur.to("cuda")
        x_gpu = torch.randn(8, 256).to("cuda")
        logits_gpu = classificateur_gpu(x_gpu, return_probs=True)
        classes_gpu = torch.argmax(logits_gpu, dim=1)
        print(f"   GPU logits  : {logits_gpu.shape}")
        print(f"   GPU classes : {classes_gpu}")
        print(f"   Mémoire GPU : {torch.cuda.memory_allocated(0) / 1024**2:.1f} Mo")
    else:
        print("   GPU non disponible")
    
    print(f"\n✅ classificateur.py — Tout OK !")
