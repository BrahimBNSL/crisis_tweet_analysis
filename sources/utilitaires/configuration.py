
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict



RACINE = Path(__file__).resolve().parents[2]  



@dataclass
class CheminConfig:
    # Données brutes
    dossier_brut_beyrouth:   Path = RACINE / "donnees" / "brutes" / "DMD_BEYROUTH"
    dossier_brut_crisismmd:  Path = RACINE / "donnees" / "brutes" / "CrisisMMD_v2.0"
    dossier_brut_multimodal: Path = RACINE / "donnees" / "brutes" / "multimodal"

    # Sous-dossiers CrisisMMD
    annotations_crisismmd:   Path = RACINE / "donnees" / "brutes" / "CrisisMMD_v2.0" / "annotations"
    images_crisismmd:        Path = RACINE / "donnees" / "brutes" / "CrisisMMD_v2.0" / "data_image"

    # Données traitées
    dossier_traitees:    Path = RACINE / "donnees" / "traitees"
    fichier_train:       Path = RACINE / "donnees" / "traitees" / "entrainement.csv"
    fichier_val:         Path = RACINE / "donnees" / "traitees" / "validation.csv"
    fichier_test:        Path = RACINE / "donnees" / "traitees" / "test.csv"
    cache_crisimmd:      Path = RACINE / "donnees" / "traitees" / "crisimmd_fusionne.csv"

    # Données augmentées
    dossier_augmente:    Path = RACINE / "donnees" / "augmentees"

    # Expériences
    dossier_configs:     Path = RACINE / "experiences" / "configurations"
    dossier_checkpoints: Path = RACINE / "experiences" / "points_de_sauvegarde"

    # Rapports
    dossier_figures:     Path = RACINE / "rapports" / "figures"



@dataclass
class ClassesConfig:
    
    noms: List[str] = field(default_factory=lambda: [
    "urgence",           
    "info_signalement",
    "non_pertinent",
    ])
    poids: List[float] = field(default_factory=lambda: [3, 1.0, 1.5])

    @property
    def nb_classes(self) -> int:
        return len(self.noms)

    @property
    def label_vers_idx(self) -> Dict[str, int]:
        return {nom: i for i, nom in enumerate(self.noms)}

    @property
    def idx_vers_label(self) -> Dict[int, str]:
        return {i: nom for i, nom in enumerate(self.noms)}

@dataclass
class TexteConfig:
    # Modèle pré-entraîné HuggingFace
    nom_modele:         str  = "vinai/bertweet-base"
    longueur_max:       int  = 128          # tokens max par tweet
    freeze_bert:        bool = False        # Fine-tune tout BERTweet
    dim_sortie:         int  = 768          # taille du [CLS] en sortie

    # Nettoyage
    supprimer_urls:         bool = True
    remplacer_mentions:     bool = True
    normaliser_hashtags:    bool = True
    expandre_abreviations:  bool = True
    remplacer_nombres:      bool = True
    reduire_repetitions:    bool = True
    mettre_minuscules:      bool = False    



@dataclass
class ImageConfig:
    # Modèle pré-entraîné torchvision
    nom_modele:         str  = "resnet50"
    taille_image:       int  = 224          # pixels (carré)
    freeze_resnet:      bool = False        # Fine-tune ResNet
    dim_sortie:         int  = 2048         # après avg-pool global

    # Normalisation ImageNet
    moyenne_norm:       List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std_norm:           List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation (train uniquement)
    flip_horizontal:    bool  = True
    rotation_max_deg:   int   = 15
    color_jitter:       bool  = True


# ══════════════════════════════════════════
# 5.  MÉTADONNÉES  (MLP)
# ══════════════════════════════════════════
@dataclass
class MetaConfig:
    # Dimensions des features d'entrée
    dim_lieu:           int  = 32           # embedding géographique
    dim_horodatage:     int  = 16           # encoding temporel (sin/cos)
    dim_source:         int  = 8            # type de compte (officiel, citoyen…)

    # Architecture MLP
    couches_cachees:    List[int] = field(default_factory=lambda: [64, 64])
    dim_sortie:         int  = 64
    dropout:            float = 0.5

    @property
    def dim_entree(self) -> int:
        return self.dim_lieu + self.dim_horodatage + self.dim_source


# ══════════════════════════════════════════
# 6.  FUSION CROSS-MODALE
# ══════════════════════════════════════════
@dataclass
class FusionConfig:
    # Projection commune avant attention
    dim_projection:     int   = 256         # texte(768) + image(2048) → 256
    nb_tetes_attention: int   = 8           # multi-head cross-attention
    dropout_attention:  float = 0.1

    # Gate (gating network)
    utiliser_gate:      bool  = True
    dim_gate_cache:     int   = 128

    # Dimension de sortie de la fusion
    dim_sortie:         int   = 256         # entrée du contexte temporel


# ══════════════════════════════════════════
# 7.  CONTEXTE TEMPOREL  (Bi-LSTM)
# ══════════════════════════════════════════
@dataclass
class TemporelConfig:
    dim_entree:         int   = 256         # == FusionConfig.dim_sortie
    dim_cachee_lstm:    int   = 128         # par direction → sortie 256
    nb_couches_lstm:    int   = 1
    dropout_lstm:       float = 0.1

    # Bloc Transformer au-dessus du Bi-LSTM
    utiliser_transformer: bool = True
    nb_tetes_transfo:     int  = 4
    dim_feedforward:      int  = 512
    dropout_transfo:      float = 0.1

    @property
    def dim_sortie(self) -> int:
        # Bi-LSTM → 2 × dim_cachee_lstm
        return self.dim_cachee_lstm * 2


# ══════════════════════════════════════════
# 8.  CLASSIFICATEUR
# ══════════════════════════════════════════
@dataclass
class ClassificateurConfig:
    # dim_entree = TemporelConfig.dim_sortie + MetaConfig.dim_sortie
    couches_denses:     List[int] = field(default_factory=lambda: [256, 128])
    dropout:            float = 0.3
    activation:         str   = "gelu"      # relu | gelu | silu


# ══════════════════════════════════════════
# 9.  ENTRAÎNEMENT
# ══════════════════════════════════════════
@dataclass
class EntrainementConfig:
    # Taille de batch
    taille_batch_train: int   = 32
    taille_batch_eval:  int   = 64

    # Optimiseur (AdamW)
    lr_bert:            float = 2e-5        # LR faible pour le pré-entraîné
    lr_resnet:          float = 1e-4        # LR pour ResNet
    lr_reste:           float = 1e-3        # LR pour les couches custom
    weight_decay:       float = 0.01
    gradient_clip:      float = 1.0

    # Planificateur cosine + échauffement
    nb_epochs:          int   = 30
    nb_epochs_warmup:   int   = 3           # warmup linéaire
    eta_min:            float = 1e-7        # LR minimale cosine

    # Focal Loss
    focal_gamma:        float = 2.0         # focalisation sur exemples difficiles
    focal_alpha:        float = 0.25        # pondération de base

    # Early stopping
    patience:           int   = 5           # epochs sans amélioration
    delta_min:          float = 1e-4        # amélioration minimale

    # Reproductibilité
    graine:             int   = 42

    # Device
    utiliser_gpu:       bool  = True
    nb_workers:         int   = 4           # DataLoader workers

    # Split
    proportion_train:   float = 0.70
    proportion_val:     float = 0.15
    proportion_test:    float = 0.15


# ══════════════════════════════════════════
# 10. JOURNALISATION
# ══════════════════════════════════════════
@dataclass
class JournalisationConfig:
    outil:              str   = "wandb"     # "wandb" | "tensorboard" | "both"
    projet_wandb:       str   = "crisis-tweets-classification"
    log_chaque_n_steps: int   = 50
    sauvegarder_meilleur: bool = True


# ══════════════════════════════════════════
# 11. CONFIG GLOBALE  (point d'entrée unique)
# ══════════════════════════════════════════
@dataclass
class Config:
    """
    Usage :
        from sources.utilitaires.configuration import cfg
        print(cfg.texte.nom_modele)   # "vinai/bertweet-base"
    """
    chemins:          CheminConfig          = field(default_factory=CheminConfig)
    classes:          ClassesConfig         = field(default_factory=ClassesConfig)
    texte:            TexteConfig           = field(default_factory=TexteConfig)
    image:            ImageConfig           = field(default_factory=ImageConfig)
    meta:             MetaConfig            = field(default_factory=MetaConfig)
    fusion:           FusionConfig          = field(default_factory=FusionConfig)
    temporel:         TemporelConfig        = field(default_factory=TemporelConfig)
    classificateur:   ClassificateurConfig  = field(default_factory=ClassificateurConfig)
    entrainement:     EntrainementConfig    = field(default_factory=EntrainementConfig)
    journalisation:   JournalisationConfig  = field(default_factory=JournalisationConfig)

    def __post_init__(self):
        """Crée les dossiers nécessaires s'ils n'existent pas."""
        dossiers = [
            self.chemins.dossier_traitees,
            self.chemins.dossier_checkpoints,
            self.chemins.dossier_figures,
            self.chemins.dossier_augmente,
        ]
        for d in dossiers:
            d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────
# Instance par défaut exportée
# ──────────────────────────────────────────
cfg = Config()


# ──────────────────────────────────────────
# Test rapide
# ──────────────────────────────────────────
if __name__ == "__main__":
    c = Config()
    print("=== Configuration chargée ===")
    print(f"  Racine projet  : {RACINE}")
    print(f"  Modèle texte   : {c.texte.nom_modele}")
    print(f"  Modèle image   : {c.image.nom_modele}")
    print(f"  Nb classes     : {c.classes.nb_classes}")
    print(f"  Noms classes   : {c.classes.noms}")
    print(f"  Poids classes  : {c.classes.poids}")
    print(f"  Dim fusion out : {c.fusion.dim_sortie}")
    print(f"  Dim LSTM out   : {c.temporel.dim_sortie}")
    print(f"  LR BERTweet    : {c.entrainement.lr_bert}")
    print(f"  Epochs         : {c.entrainement.nb_epochs}")
    print(f"  Patience ES    : {c.entrainement.patience}")
    print(f"  Split          : {c.entrainement.proportion_train}/{c.entrainement.proportion_val}/{c.entrainement.proportion_test}")
    print(f"  Dossier traité : {c.chemins.dossier_traitees}")
    print("✓ Tout OK")