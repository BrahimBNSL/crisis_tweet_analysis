

import logging
from pathlib import Path
from typing import List, Optional, Union
from sources.donnees.traitement_images import TraiteurImage
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)



class EncodeurImage(nn.Module):
    
    
    
    DIMS_SORTIE = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "resnext50_32x4d": 2048,
        "resnext101_32x8d": 2048,
        "wide_resnet50_2": 2048,
        "wide_resnet101_2": 2048,
        "efficientnet_b0": 1280,
        "efficientnet_b4": 1792,
        "efficientnet_b7": 2560,
        "vit_b_16": 768,
        "vit_b_32": 768,
        "vit_l_16": 1024,
        "mobilenet_v3_large": 960,
        "densenet121": 1024,
    }
    
    def __init__(
        self,
        nom_modele: Optional[str] = None,
        freeze_backbone: Optional[bool] = None,
        dim_projection: Optional[int] = None,
        dropout: float = 0.2,
        utiliser_pretrained: bool = True,
    ):
        super().__init__()
        
        # Configuration
        self.nom_modele = nom_modele or cfg.image.nom_modele
        self.freeze_backbone = freeze_backbone if freeze_backbone is not None else cfg.image.freeze_resnet
        self.dim_sortie_backbone = self.DIMS_SORTIE.get(self.nom_modele, 2048)
        self.utiliser_pretrained = utiliser_pretrained
        
        # ── Charger le backbone ──
        logger.info(f"  Chargement du backbone : {self.nom_modele}")
        self.backbone = self._charger_backbone()
        
        # Geler les poids si demandé
        if self.freeze_backbone:
            logger.info("  Backbone gelé (freeze)")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ── Projection optionnelle ──
        self.dim_projection = dim_projection
        if dim_projection and dim_projection > 0:
            self.projection = nn.Sequential(
                nn.Linear(self.dim_sortie_backbone, dim_projection),
                nn.BatchNorm1d(dim_projection),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.dim_sortie = dim_projection
            logger.info(f" Projection ajoutée : {self.dim_sortie_backbone} → {dim_projection}")
        else:
            self.projection = None
            self.dim_sortie = self.dim_sortie_backbone
        
        # Stats
        nb_params_total = sum(p.numel() for p in self.parameters())
        nb_params_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f" EncodeurImage initialisé :")
        logger.info(f"   • Modèle       : {self.nom_modele}")
        logger.info(f"   • Dim backbone : {self.dim_sortie_backbone}")
        logger.info(f"   • Dim sortie   : {self.dim_sortie}")
        logger.info(f"   • Freeze       : {self.freeze_backbone}")
        logger.info(f"   • Pretrained   : {self.utiliser_pretrained}")
        logger.info(f"   • Params totaux       : {nb_params_total:,}")
        logger.info(f"   • Params entraînables : {nb_params_train:,}")
    
    # ──────────────────────────────────────
    def _charger_backbone(self) -> nn.Module:
        
        # Récupérer la fonction de chargement
        if self.nom_modele.startswith("resnet") or self.nom_modele.startswith("resnext") or self.nom_modele.startswith("wide_resnet"):
            return self._charger_resnet()
        elif self.nom_modele.startswith("efficientnet"):
            return self._charger_efficientnet()
        elif self.nom_modele.startswith("vit"):
            return self._charger_vit()
        elif self.nom_modele.startswith("mobilenet"):
            return self._charger_mobilenet()
        elif self.nom_modele.startswith("densenet"):
            return self._charger_densenet()
        else:
            logger.warning(f"  Modèle inconnu : {self.nom_modele}. Utilisation de ResNet-50 par défaut.")
            self.nom_modele = "resnet50"
            return self._charger_resnet()
    
    def _charger_resnet(self) -> nn.Module:
        
        model_fn = getattr(models, self.nom_modele)
        model = model_fn(weights="IMAGENET1K_V2" if self.utiliser_pretrained else None)
        # Garder tout sauf la couche FC
        modules = list(model.children())[:-1]  # Retire AdaptiveAvgPool + FC
        return nn.Sequential(
            *modules,
            nn.Flatten(),  # [B, 2048]
        )
    
    def _charger_efficientnet(self) -> nn.Module:
        
        model_fn = getattr(models, self.nom_modele)
        model = model_fn(weights="IMAGENET1K_V1" if self.utiliser_pretrained else None)
        # Remplacer le classifier par Flatten
        model.classifier = nn.Flatten()
        return model
    
    def _charger_vit(self) -> nn.Module:
       
        model_fn = getattr(models, self.nom_modele)
        model = model_fn(weights="IMAGENET1K_V1" if self.utiliser_pretrained else None)
        # Utiliser le token CLS comme embedding
        model.heads = nn.Flatten()
        return model
    
    def _charger_mobilenet(self) -> nn.Module:
       
        model_fn = getattr(models, self.nom_modele)
        model = model_fn(weights="IMAGENET1K_V1" if self.utiliser_pretrained else None)
        model.classifier = nn.Flatten()
        return model
    
    def _charger_densenet(self) -> nn.Module:
       
        model_fn = getattr(models, self.nom_modele)
        model = model_fn(weights="IMAGENET1K_V1" if self.utiliser_pretrained else None)
        model.classifier = nn.Flatten()
        return model
    
    # ──────────────────────────────────────
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        # Passage dans le backbone
        features = self.backbone(images)  # [B, dim_backbone]
        
        # S'aplatir si nécessaire (certains modèles sortent [B, D, 1, 1])
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # Global Average Pooling
        
        # S'assurer que c'est bien 2D
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Projection optionnelle
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    # ──────────────────────────────────────
    def encoder_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        
        from sources.donnees.traitement_images import TraiteurImage
        
        traiteur = TraiteurImage(mode="eval")
        tenseur = traiteur.charger_et_transformer(image)  # [3, 224, 224]
        tenseur = tenseur.unsqueeze(0)  # [1, 3, 224, 224]
        
        with torch.no_grad():
            embedding = self.forward(tenseur)
        
        return embedding.squeeze(0)  # [dim_sortie]
    
    # ──────────────────────────────────────
    def encoder_batch(
        self,
        images: List[Optional[Union[str, Path, Image.Image]]],
    ) -> torch.Tensor:
       
        from sources.donnees.traitement_images import TraiteurImage
        
        traiteur = TraiteurImage(mode="eval")
        tenseurs = traiteur.charger_batch(images)  # [B, 3, 224, 224]
        
        device = next(self.parameters()).device
        tenseurs = tenseurs.to(device)
        
        with torch.no_grad():
            embeddings = self.forward(tenseurs)
        
        return embeddings


# ══════════════════════════════════════════
# 2.  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def creer_encodeur_image(
    nom_modele: Optional[str] = None,
    freeze_backbone: Optional[bool] = None,
    dim_projection: Optional[int] = None,
) -> EncodeurImage:
    
    return EncodeurImage(
        nom_modele=nom_modele or cfg.image.nom_modele,
        freeze_backbone=freeze_backbone if freeze_backbone is not None else cfg.image.freeze_resnet,
        dim_projection=dim_projection,
    )


def lister_modeles_disponibles() -> List[str]:
    
    print("\n📋 Modèles disponibles :")
    print("-" * 50)
    for nom, dim in sorted(EncodeurImage.DIMS_SORTIE.items()):
        print(f"   {nom:<30} → {dim:>6} dims")
    return list(EncodeurImage.DIMS_SORTIE.keys())


def comparer_images(
    encodeur: EncodeurImage,
    images: List[Union[str, Path, Image.Image]],
    noms: Optional[List[str]] = None,
) -> None:
    
    import torch.nn.functional as F
    
    if noms is None:
        noms = [f"Image {i}" for i in range(len(images))]
    
    print("\n Comparaison des embeddings d'images :")
    print("-" * 60)
    
    embeddings = encodeur.encoder_batch(images)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            sim = (embeddings_norm[i] @ embeddings_norm[j]).item()
            print(f"   Similarité [{noms[i]}] vs [{noms[j]}] : {sim:.4f}")


# ══════════════════════════════════════════
# 3.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print(" TEST — EncodeurImage (ResNet-50)")
    print("=" * 70)
    
    # ── Test 1 : Création de l'encodeur ──
    print("\n Test 1 — Création de l'encodeur")
    encodeur = creer_encodeur_image()
    
    # ── Test 2 : Image factice ──
    print("\n Test 2 — Encodage d'une image factice")
    image_factice = Image.new('RGB', (300, 250), color='red')
    embedding = encodeur.encoder_image(image_factice)
    print(f"   Image factice : {image_factice.size}")
    print(f"   Embedding shape : {embedding.shape}")
    print(f"   Mean   : {embedding.mean().item():.4f}")
    print(f"   Std    : {embedding.std().item():.4f}")
    print(f"   Min    : {embedding.min().item():.4f}")
    print(f"   Max    : {embedding.max().item():.4f}")
    
    # ── Test 3 : Batch d'images factices ──
    print("\n Test 3 — Encodage d'un batch d'images")
    batch_images = [
        Image.new('RGB', (300, 250), color='red'),
        Image.new('RGB', (300, 250), color='blue'),
        Image.new('RGB', (300, 250), color='green'),
        Image.new('RGB', (300, 250), color='yellow'),
    ]
    embeddings_batch = encodeur.encoder_batch(batch_images)
    print(f"   Batch shape : {embeddings_batch.shape}")
    
    # ── Test 4 : Similarité entre images ──
    print("\n Test 4 — Similarité entre images")
    comparer_images(
        encodeur,
        batch_images,
        noms=["Rouge", "Bleu", "Vert", "Jaune"]
    )
    
    # ── Test 5 : Image noire (fallback) ──
    print("\n Test 5 — Fallback image noire (None)")
    embedding_noir = encodeur.encoder_image(None)
    print(f"   Embedding shape : {embedding_noir.shape}")
    print(f"   Mean : {embedding_noir.mean().item():.4f}")
    
    # ── Test 6 : Test avec une vraie image si disponible ──
    print("\n Test 6 — Test avec une image réelle (si disponible)")
    chemin_image = Path(r"D:\crisis_tweet_analysis\donnees\brutes\multimodal\damaged_infrastructure\images")
    if chemin_image.exists():
        images_dispo = list(chemin_image.glob("*.jpg"))
        if images_dispo:
            img_reelle = images_dispo[0]
            print(f"   Image : {img_reelle.name}")
            embedding_reelle = encodeur.encoder_image(str(img_reelle))
            print(f"   Embedding shape : {embedding_reelle.shape}")
            print(f"   Mean : {embedding_reelle.mean().item():.4f}, Std : {embedding_reelle.std().item():.4f}")
        else:
            print("   Aucune image trouvée dans le dossier")
    else:
        print(f"   Dossier introuvable : {chemin_image}")
    
    # ── Test 7 : Liste des modèles disponibles ──
    print("\ Test 7 — Modèles supportés")
    lister_modeles_disponibles()
    
    # ── Test 8 : Vérification GPU ──
    print("\n Test 8 — Info device")
    device = next(encodeur.parameters()).device
    print(f"   Device : {device}")
    if torch.cuda.is_available():
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
        # Déplacer sur GPU pour tester
        encodeur_gpu = encodeur.to("cuda")
        traiteur = TraiteurImage(mode="eval")
        tenseur = traiteur.charger_et_transformer(image_factice).unsqueeze(0).to("cuda")
        embedding_gpu = encodeur_gpu(tenseur)
        print(f"   GPU embedding shape : {embedding_gpu.shape}")
        print(f"   Mémoire GPU allouée : {torch.cuda.memory_allocated(0) / 1024**2:.1f} Mo")
    
    print(f"\n encodeur_image.py — Tout OK !")