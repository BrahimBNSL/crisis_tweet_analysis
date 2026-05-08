"""
encodeur_texte.py
─────────────────
Encodeur de texte basé sur BERTweet pour la classification de tweets de crise.

Architecture :
    texte nettoyé (str)
        → TokeniseurBERTweet (input_ids + attention_mask)
        → BERTweet pré-entraîné (vinai/bertweet-base)
        → pooled_output [CLS]  →  embedding [768]
        → projection optionnelle → embedding [dim_sortie]

Utilisation :
    encodeur = EncodeurTexte()
    embedding = encodeur("explosion massive HASHTAG Beyrouth USER HTTPURL")
    # → torch.Tensor de forme [768]
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  ENCODEUR BERTWEET
# ══════════════════════════════════════════

class EncodeurTexte(nn.Module):
    """
    Encodeur de texte utilisant BERTweet (ou tout modèle HuggingFace).
    
    Pipeline :
        texte → tokenizer → BERTweet → embedding [CLS]
    
    Args:
        nom_modele: Nom du modèle HuggingFace (défaut : cfg.texte.nom_modele)
        longueur_max: Longueur max des tokens (défaut : cfg.texte.longueur_max)
        freeze_bert: Si True, gèle les poids de BERTweet
        dim_projection: Si > 0, ajoute une couche de projection après [CLS]
        dropout: Taux de dropout après projection
    """
    
    def __init__(
        self,
        nom_modele: Optional[str] = None,
        longueur_max: Optional[int] = None,
        freeze_bert: Optional[bool] = None,
        dim_projection: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Configuration
        self.nom_modele = nom_modele or cfg.texte.nom_modele
        self.longueur_max = longueur_max or cfg.texte.longueur_max
        self.freeze_bert = freeze_bert if freeze_bert is not None else cfg.texte.freeze_bert
        self.dim_sortie = cfg.texte.dim_sortie  # 768 pour BERTweet-base
        
        # ── Tokenizer ──
        logger.info(f"📥 Chargement du tokenizer : {self.nom_modele}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.nom_modele,
            normalization=True,  # Normalisation BERTweet (emojis → texte)
            use_fast=True,
        )
        
        # ── Modèle BERTweet ──
        logger.info(f"🧠 Chargement du modèle : {self.nom_modele}")
        self.bert = AutoModel.from_pretrained(self.nom_modele)
        
        # Geler les poids si demandé
        if self.freeze_bert:
            logger.info("❄️  BERTweet gelé (freeze)")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # ── Projection optionnelle ──
        self.dim_projection = dim_projection
        if dim_projection and dim_projection > 0:
            self.projection = nn.Sequential(
                nn.Linear(self.dim_sortie, dim_projection),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.dim_sortie = dim_projection
            logger.info(f"📐 Projection ajoutée : {cfg.texte.dim_sortie} → {dim_projection}")
        else:
            self.projection = None
        
        # Stats
        nb_params_total = sum(p.numel() for p in self.parameters())
        nb_params_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✅ EncodeurTexte initialisé :")
        logger.info(f"   • Modèle      : {self.nom_modele}")
        logger.info(f"   • Longueur max : {self.longueur_max}")
        logger.info(f"   • Dim sortie  : {self.dim_sortie}")
        logger.info(f"   • Freeze BERT : {self.freeze_bert}")
        logger.info(f"   • Params totaux : {nb_params_total:,}")
        logger.info(f"   • Params entraînables : {nb_params_train:,}")
    
    # ──────────────────────────────────────
    def tokeniser(
        self,
        textes: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenise une liste de textes (déjà nettoyés).
        
        Args:
            textes: Liste de str
        
        Returns:
            Dictionnaire {input_ids, attention_mask}
        """
        encodage = self.tokenizer(
            textes,
            max_length=self.longueur_max,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids": encodage["input_ids"],
            "attention_mask": encodage["attention_mask"],
        }
    
    # ──────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Passe avant : texte tokenisé → embedding.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] masque d'attention
        
        Returns:
            Embedding [B, dim_sortie]
        """
        # Passage dans BERTweet
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Récupérer le token [CLS] (premier token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # Projection optionnelle
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        return pooled_output
    
    # ──────────────────────────────────────
    def encoder_texte(self, texte: str) -> torch.Tensor:
        """
        Encode un seul texte (pratique pour l'inférence).
        
        Args:
            texte: Texte nettoyé (str)
        
        Returns:
            Embedding [dim_sortie]
        """
        tokens = self.tokeniser([texte])
        with torch.no_grad():
            embedding = self.forward(
                tokens["input_ids"],
                tokens["attention_mask"],
            )
        return embedding.squeeze(0)  # [dim_sortie]
    
    # ──────────────────────────────────────
    def encoder_batch(self, textes: list) -> torch.Tensor:
        """
        Encode un batch de textes.
        
        Args:
            textes: Liste de str
        
        Returns:
            Embeddings [B, dim_sortie]
        """
        tokens = self.tokeniser(textes)
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            embeddings = self.forward(
                tokens["input_ids"],
                tokens["attention_mask"],
            )
        return embeddings


# ══════════════════════════════════════════
# 2.  PIPELINE TEXTE COMPLET (nettoyage + encodage)
# ══════════════════════════════════════════

class PipelineTexteComplet(nn.Module):
    """
    Pipeline complet : nettoyage → tokenisation → encodage.
    
    Combine NettoyeurTweet + EncodeurTexte en un seul module.
    Utile pour l'inférence de bout en bout.
    
    Args:
        encodeur: Instance de EncodeurTexte
        nettoyeur: Instance de NettoyeurTweet
    """
    
    def __init__(self, encodeur: EncodeurTexte, nettoyeur):
        super().__init__()
        self.encodeur = encodeur
        self.nettoyeur = nettoyeur
    
    def forward(self, textes_bruts: list) -> torch.Tensor:
        """
        Textes bruts → embeddings.
        
        Args:
            textes_bruts: Liste de textes bruts (non nettoyés)
        
        Returns:
            Embeddings [B, dim_sortie]
        """
        # Nettoyer les textes
        textes_nettoyes = self.nettoyeur.nettoyer_batch(textes_bruts)
        
        # Tokeniser
        tokens = self.encodeur.tokeniser(textes_nettoyes)
        device = next(self.encodeur.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Encoder
        embeddings = self.encodeur.forward(
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        return embeddings
    
    def encoder_texte(self, texte_brut: str) -> torch.Tensor:
        """Un seul texte brut → embedding."""
        texte_nettoye = self.nettoyeur.nettoyer(texte_brut)
        return self.encodeur.encoder_texte(texte_nettoye)


# ══════════════════════════════════════════
# 3.  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def creer_encodeur_texte(
    freeze_bert: Optional[bool] = None,
    dim_projection: Optional[int] = None,
) -> EncodeurTexte:
    """
    Fabrique un EncodeurTexte avec la configuration centrale.
    
    Args:
        freeze_bert: Geler BERTweet (défaut : cfg.texte.freeze_bert)
        dim_projection: Dimension de projection (None = pas de projection)
    
    Returns:
        EncodeurTexte initialisé
    """
    return EncodeurTexte(
        nom_modele=cfg.texte.nom_modele,
        longueur_max=cfg.texte.longueur_max,
        freeze_bert=freeze_bert if freeze_bert is not None else cfg.texte.freeze_bert,
        dim_projection=dim_projection,
    )


def comparer_embeddings(
    encodeur: EncodeurTexte,
    textes: list,
) -> None:
    """
    Affiche la similarité cosinus entre les embeddings de plusieurs textes.
    Utile pour vérifier que l'encodeur fonctionne correctement.
    """
    import torch.nn.functional as F
    
    print("\n📊 Comparaison des embeddings :")
    print("-" * 50)
    
    embeddings = encodeur.encoder_batch(textes)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    for i in range(len(textes)):
        for j in range(i + 1, len(textes)):
            sim = (embeddings_norm[i] @ embeddings_norm[j]).item()
            print(f"   Similarité [{i}] vs [{j}] : {sim:.4f}")
            print(f"      Texte {i} : {textes[i][:60]}...")
            print(f"      Texte {j} : {textes[j][:60]}...")
            print()


# ══════════════════════════════════════════
# 4.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🧪 TEST — EncodeurTexte (BERTweet)")
    print("=" * 70)
    
    # ── Test 1 : Création de l'encodeur ──
    print("\n📌 Test 1 — Création de l'encodeur")
    encodeur = creer_encodeur_texte()
    
    # ── Test 2 : Encodage d'un seul texte ──
    print("\n📌 Test 2 — Encodage d'un seul texte")
    texte_test = "explosion massive HASHTAG Beyrouth USER HTTPURL injured people need help"
    embedding = encodeur.encoder_texte(texte_test)
    print(f"   Texte  : {texte_test}")
    print(f"   Shape  : {embedding.shape}")
    print(f"   Mean   : {embedding.mean().item():.4f}")
    print(f"   Std    : {embedding.std().item():.4f}")
    print(f"   Min    : {embedding.min().item():.4f}")
    print(f"   Max    : {embedding.max().item():.4f}")
    
    # ── Test 3 : Encodage d'un batch ──
    print("\n📌 Test 3 — Encodage d'un batch")
    textes_batch = [
        "HASHTAG BREAKING earthquake in Iran HTTPURL USER",
        "RT USER: Trump pledges NUMBER million for HASHTAG Harvey relief HTTPURL",
        "Buy this amazing product HASHTAG ad HASHTAG sponsored HTTPURL",
    ]
    embeddings_batch = encodeur.encoder_batch(textes_batch)
    print(f"   Batch shape : {embeddings_batch.shape}")
    print(f"   Nombre de textes : {len(textes_batch)}")
    
    # ── Test 4 : Comparaison de similarité ──
    print("\n📌 Test 4 — Similarité cosinus entre textes")
    comparer_embeddings(encodeur, textes_batch)
    
    # ── Test 5 : Tokenisation seule ──
    print("\n📌 Test 5 — Tokenisation")
    tokens = encodeur.tokeniser(textes_batch)
    print(f"   input_ids shape      : {tokens['input_ids'].shape}")
    print(f"   attention_mask shape : {tokens['attention_mask'].shape}")
    print(f"   Tokens non-PAD (moy) : {tokens['attention_mask'].sum(dim=1).float().mean().item():.0f}")
    
    # ── Test 6 : Pipeline complet (avec nettoyage) ──
    print("\n📌 Test 6 — Pipeline complet (nettoyage + encodage)")
    from sources.donnees.traitement_texte import NettoyeurTweet
    
    nettoyeur = NettoyeurTweet()
    pipeline = PipelineTexteComplet(encodeur, nettoyeur)
    
    textes_bruts = [
        "RT @CNN: BREAKING: Huge explosion in #Beirut !! https://t.co/abc123 2750 dead ???",
        "Please donate to help victims of #LebanonCrisis https://t.co/xyz999",
    ]
    
    embeddings_bruts = pipeline(textes_bruts)
    print(f"   Textes bruts → embeddings : {embeddings_bruts.shape}")
    print(f"   Texte 1 brut : {textes_bruts[0][:80]}...")
    print(f"   Embedding 1  : mean={embeddings_bruts[0].mean():.4f}, std={embeddings_bruts[0].std():.4f}")
    
    # ── Test 7 : Vérification mémoire GPU ──
    print("\n📌 Test 7 — Info device")
    device = next(encodeur.parameters()).device
    print(f"   Device : {device}")
    if torch.cuda.is_available():
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire allouée : {torch.cuda.memory_allocated(0) / 1024**2:.1f} Mo")
    
    print(f"\n✅ encodeur_texte.py — Tout OK !")