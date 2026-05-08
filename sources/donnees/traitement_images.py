"""
traitement_images.py
─────────────────────
Chargement, prétraitement et augmentation des images pour le modèle multimodal.

Pipeline :
    image PIL / chemin
        → redimensionnement 224×224
        → normalisation ImageNet
        → augmentations (entraînement uniquement)
            • Flip horizontal
            • Rotation (±15°)
            • ColorJitter (luminosité, contraste, saturation)
        → tenseur [3, 224, 224] prêt pour ResNet-50 / ViT

Intégration avec jeu_de_donnees.py via le DataLoader multimodal.
"""

import logging
from typing import Optional, Tuple, List, Dict, Union

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance

from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  TRANSFORMATIONS DE BASE
# ══════════════════════════════════════════

def transformation_inference() -> transforms.Compose:
    """
    Transformation pour l'inférence / validation / test.
    Aucune augmentation — juste redimensionnement + normalisation.
    
    Returns:
        Compose [Resize → ToTensor → Normalize]
    """
    return transforms.Compose([
        transforms.Resize((cfg.image.taille_image, cfg.image.taille_image)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.image.moyenne_norm,
            std=cfg.image.std_norm,
        ),
    ])


def transformation_entrainement() -> transforms.Compose:
    """
    Transformation pour l'entraînement avec augmentations légères.
    
    Augmentations :
        • Flip horizontal (50% de chance)
        • Rotation aléatoire ±15°
        • ColorJitter (luminosité ±10%, contraste ±10%, saturation ±10%)
        • Resize 224×224
        • ToTensor
        • Normalize ImageNet
    
    Returns:
        Compose avec augmentations
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg.image.rotation_max_deg),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.0,  # pas de changement de teinte
        ),
        transforms.Resize((cfg.image.taille_image, cfg.image.taille_image)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.image.moyenne_norm,
            std=cfg.image.std_norm,
        ),
    ])


# ══════════════════════════════════════════
# 2.  TRANSFORMATIONS SPÉCIALISÉES (optionnel)
# ══════════════════════════════════════════

def transformation_legere() -> transforms.Compose:
    """
    Version allégée pour fine-tuning prudent.
    Seulement flip horizontal + normalisation.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((cfg.image.taille_image, cfg.image.taille_image)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.image.moyenne_norm,
            std=cfg.image.std_norm,
        ),
    ])


def transformation_augmentee() -> transforms.Compose:
    """
    Version plus agressive pour petites datasets.
    Ajoute crop aléatoire et changement de teinte.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg.image.rotation_max_deg),
        transforms.RandomResizedCrop(
            size=cfg.image.taille_image,
            scale=(0.8, 1.0),  # crop entre 80% et 100% de l'image
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.image.moyenne_norm,
            std=cfg.image.std_norm,
        ),
    ])


# ══════════════════════════════════════════
# 3.  CLASSE DE TRAITEMENT D'IMAGE
# ══════════════════════════════════════════

class TraiteurImage:
    """
    Encapsule les transformations d'image pour le pipeline multimodal.
    
    Gère :
        • Chargement depuis un chemin ou un objet PIL
        • Conversion RGB automatique
        • Gestion des images manquantes ou corrompues (fallback : tenseur noir)
        • Application des transformations d'augmentation
    
    Args:
        mode: 'train', 'eval', 'light', 'augmented'
    """
    
    def __init__(self, mode: str = "eval"):
        self.mode = mode.lower()
        
        # Choisir la transformation selon le mode
        if self.mode == "train":
            self.transform = transformation_entrainement()
        elif self.mode == "eval" or self.mode == "val" or self.mode == "test":
            self.transform = transformation_inference()
        elif self.mode == "light":
            self.transform = transformation_legere()
        elif self.mode == "augmented":
            self.transform = transformation_augmentee()
        else:
            raise ValueError(f"Mode inconnu : {mode}. Choisir parmi 'train', 'eval', 'light', 'augmented'.")
        
        logger.info(f"🖼️  TraiteurImage initialisé en mode '{self.mode}'")
    
    def charger_et_transformer(
        self,
        image: Optional[Union[str, Image.Image]],
    ) -> torch.Tensor:
        """
        Charge une image (si chemin) et applique les transformations.
        
        Args:
            image: Chemin vers l'image (str) ou objet PIL.Image, ou None
        
        Returns:
            Tenseur [3, 224, 224] normalisé, ou tenseur noir si image invalide
        """
        # ── Cas : pas d'image ──
        if image is None:
            return self._image_noire()
        
        try:
            # ── Charger si c'est un chemin ──
            if isinstance(image, str):
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            else:
                logger.warning(f"Type d'image non supporté : {type(image)}")
                return self._image_noire()
            
            # ── Appliquer les transformations ──
            tenseur = self.transform(pil_image)
            return tenseur
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors du traitement d'image : {e}")
            return self._image_noire()
    
    def charger_batch(
        self,
        images: List[Optional[Union[str, Image.Image]]],
    ) -> torch.Tensor:
        """
        Traite un batch d'images.
        
        Args:
            images: Liste de chemins, PIL.Image, ou None
        
        Returns:
            Tenseur [B, 3, 224, 224]
        """
        tenseurs = [self.charger_et_transformer(img) for img in images]
        return torch.stack(tenseurs, dim=0)
    
    def _image_noire(self) -> torch.Tensor:
        """
        Crée un tenseur noir (zéros) comme fallback.
        Utile pour les tweets sans image dans un batch multimodal.
        
        Returns:
            Tenseur [3, 224, 224] de zéros normalisé ImageNet
        """
        # Image noire normalisée avec les stats ImageNet
        noir = torch.zeros(3, cfg.image.taille_image, cfg.image.taille_image)
        noir = transforms.Normalize(
            mean=cfg.image.moyenne_norm,
            std=cfg.image.std_norm,
        )(noir)
        return noir
    
    def denormaliser(self, tenseur: torch.Tensor) -> torch.Tensor:
        """
        Inverse la normalisation ImageNet pour visualisation.
        
        Args:
            tenseur: Tenseur normalisé [3, H, W] ou [B, 3, H, W]
        
        Returns:
            Tenseur dénormalisé (valeurs ≈ [0, 1])
        """
        mean = torch.tensor(cfg.image.moyenne_norm).view(-1, 1, 1)
        std = torch.tensor(cfg.image.std_norm).view(-1, 1, 1)
        
        if tenseur.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        return tenseur * std + mean


# ══════════════════════════════════════════
# 4.  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def obtenir_transformation(mode: str = "eval") -> transforms.Compose:
    """
    Retourne la transformation appropriée selon le mode.
    Version fonctionnelle (sans classe).
    
    Args:
        mode: 'train', 'eval', 'light', 'augmented'
    
    Returns:
        Compose PyTorch
    """
    mapping = {
        "train": transformation_entrainement,
        "eval": transformation_inference,
        "val": transformation_inference,
        "test": transformation_inference,
        "light": transformation_legere,
        "augmented": transformation_augmentee,
    }
    func = mapping.get(mode.lower())
    if func is None:
        raise ValueError(f"Mode '{mode}' inconnu. Choisir parmi {list(mapping.keys())}")
    return func()


def afficher_statistiques_batch(tenseur: torch.Tensor) -> Dict[str, float]:
    """
    Affiche les statistiques d'un batch d'images normalisé.
    
    Args:
        tenseur: [B, 3, H, W] normalisé ImageNet
    
    Returns:
        Dictionnaire {min, max, mean, std}
    """
    return {
        "min": tenseur.min().item(),
        "max": tenseur.max().item(),
        "mean": tenseur.mean().item(),
        "std": tenseur.std().item(),
        "shape": list(tenseur.shape),
    }


# ══════════════════════════════════════════
# 5.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🧪 TEST — Traitement Images")
    print("=" * 70)
    
    # Test 1 : TraiteurImage en mode train
    print("\n📌 Test 1 — Mode entraînement")
    traiteur_train = TraiteurImage(mode="train")
    print(f"   Transformations : {traiteur_train.transform}")
    
    # Test 2 : TraiteurImage en mode eval
    print("\n📌 Test 2 — Mode évaluation")
    traiteur_eval = TraiteurImage(mode="eval")
    print(f"   Transformations : {traiteur_eval.transform}")
    
    # Test 3 : Créer une image factice et la transformer
    print("\n📌 Test 3 — Transformation d'une image factice")
    image_factice = Image.new('RGB', (300, 250), color='red')
    print(f"   Image entrée : {image_factice.size}")
    
    tenseur_train = traiteur_train.charger_et_transformer(image_factice)
    print(f"   Sortie train  : {tenseur_train.shape} — {afficher_statistiques_batch(tenseur_train.unsqueeze(0))}")
    
    tenseur_eval = traiteur_eval.charger_et_transformer(image_factice)
    print(f"   Sortie eval   : {tenseur_eval.shape} — {afficher_statistiques_batch(tenseur_eval.unsqueeze(0))}")
    
    # Test 4 : Image noire (fallback)
    print("\n📌 Test 4 — Fallback image noire")
    tenseur_noir = traiteur_eval.charger_et_transformer(None)
    print(f"   Tenseur noir  : {tenseur_noir.shape}, mean={tenseur_noir.mean().item():.4f}")
    
    # Test 5 : Batch d'images
    print("\n📌 Test 5 — Batch de 4 images")
    batch_images = [image_factice, None, image_factice, image_factice]
    batch_tenseur = traiteur_eval.charger_batch(batch_images)
    print(f"   Batch shape   : {batch_tenseur.shape}")
    print(f"   Stats batch   : {afficher_statistiques_batch(batch_tenseur)}")
    
    # Test 6 : Dénormalisation
    print("\n📌 Test 6 — Dénormalisation")
    image_denorm = traiteur_eval.denormaliser(tenseur_eval)
    print(f"   Avant denorm  : min={tenseur_eval.min():.3f}, max={tenseur_eval.max():.3f}")
    print(f"   Après denorm  : min={image_denorm.min():.3f}, max={image_denorm.max():.3f}")
    
    # Test 7 : Vérifier les dimensions pour ResNet
    print("\n📌 Test 7 — Compatibilité ResNet-50")
    print(f"   Taille image   : {cfg.image.taille_image}×{cfg.image.taille_image}")
    print(f"   Normalisation  : mean={cfg.image.moyenne_norm}, std={cfg.image.std_norm}")
    print(f"   Dim sortie     : {cfg.image.dim_sortie}")
    print(f"   Freeze ResNet  : {cfg.image.freeze_resnet}")
    
    print(f"\n✅ traitement_images.py — Tout OK !")