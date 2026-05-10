

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from sources.utilitaires.configuration import cfg
from sources.entrainement.boucle_entrainement import EntraineurPipeline

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 1.  PIPELINE D'INFÉRENCE / ENTRAÎNEMENT
# ══════════════════════════════════════════

class PipelineCrise(nn.Module):
    
    
    def __init__(self, modele_complet, encodeur_texte, encodeur_image, nettoyeur, traiteur_image, device=None):
        super().__init__()
        self.modele_complet = modele_complet
        self.encodeur_texte = encodeur_texte
        self.encodeur_image = encodeur_image
        self.nettoyeur = nettoyeur
        self.traiteur_image = traiteur_image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encodeur_texte = self.encodeur_texte.to(self.device)
        self.encodeur_image = self.encodeur_image.to(self.device)
        self.modele_complet = self.modele_complet.to(self.device)
        logger.info(f" PipelineCrise initialisé sur {self.device}")
    
    def _encoder_texte(self, textes):
        textes_nettoyes = self.nettoyeur.nettoyer_batch(textes)
        tokens = self.encodeur_texte.tokeniser(textes_nettoyes)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.set_grad_enabled(self.encodeur_texte.training):
            return self.encodeur_texte(tokens["input_ids"], tokens["attention_mask"])
    
    def _encoder_image(self, images):
        tenseurs = self.traiteur_image.charger_batch(images).to(self.device)
        with torch.set_grad_enabled(self.encodeur_image.training):
            return self.encodeur_image(tenseurs)
    
    def forward(self, textes, images, return_poids=False):
        emb_texte = self._encoder_texte(textes)
        emb_image = self._encoder_image(images)
        if return_poids:
            logits, poids = self.modele_complet(emb_texte, emb_image, return_poids=True)
            return logits, poids
        return self.modele_complet(emb_texte, emb_image)
    
    @torch.no_grad()
    def predire(self, textes, images):
        self.eval()
        logits = self.forward(textes, images)
        probas = torch.softmax(logits, dim=-1)
        return torch.argmax(probas, dim=-1), probas



def assembler_pipeline(
    device=None,
    utiliser_gate=True,
    freeze_bert=True,
    unfreeze_2_couches=True, 
    freeze_resnet=True,
    utiliser_lora=True,
) -> PipelineCrise:
   
    from sources.donnees.traitement_texte import NettoyeurTweet
    from sources.donnees.traitement_images import TraiteurImage
    from sources.modeles.encodeur_texte import EncodeurTexte
    from sources.modeles.encodeur_image import EncodeurImage
    from sources.modeles.fusion_crossmodale import FusionCrossModule
    from sources.modeles.classificateur import ClassificateurCrise, ModeleComplet
    
    logger.info("🏗️  Assemblage du pipeline (Cross-Attention + LoRA + 2 couches BERT)...")
    
    # 1. Nettoyeur
    nettoyeur = NettoyeurTweet(
        supprimer_urls=True, remplacer_mentions=True,
        normaliser_hashtags=True, expandre_abreviations=True,
        remplacer_nombres=True, reduire_repetitions=True,
        mettre_minuscules=False,
    )
    
    # 2. Traiteur image
    traiteur_image = TraiteurImage(mode="train")
    
    # 3. Encodeur texte (BERTweet GELÉ)
    encodeur_texte = EncodeurTexte(freeze_bert=True)
    
    # 4. Débloquer les 2 dernières couches de BERTweet
    if unfreeze_2_couches:
        logger.info(" Déblocage des 2 dernières couches de BERTweet...")
        nb_debloque = 0
        for name, param in encodeur_texte.bert.named_parameters():
            if any(k in name for k in ['encoder.layer.10', 'encoder.layer.11', 'pooler']):
                param.requires_grad = True
                nb_debloque += 1
            else:
                param.requires_grad = False
        logger.info(f"    {nb_debloque} paramètres débloqués (+~25M params)")
    
    # 5. Appliquer LoRA à BERTweet
    if utiliser_lora:
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1)
            encodeur_texte.bert = get_peft_model(encodeur_texte.bert, lora_config)
            nb_lora = sum(p.numel() for p in encodeur_texte.bert.parameters() if p.requires_grad)
            logger.info(f" LoRA appliqué (+{nb_lora:,} params)")
        except ImportError:
            logger.warning("  peft non installé")
    
    # 6. Encodeur image (ResNet GELÉ)
    encodeur_image = EncodeurImage(freeze_backbone=freeze_resnet)
    
    # 7. Fusion Cross-Attention + Gate
    fusion = FusionCrossModule(
        dim_texte=768, dim_image=2048, dim_projection=256,
        nb_tetes=8, utiliser_gate=utiliser_gate, dropout=0.3,
    )
    
    # 8. Classificateur
    classificateur = ClassificateurCrise()
    modele_complet = ModeleComplet(fusion, classificateur)
    
    # 9. Pipeline final
    pipeline = PipelineCrise(modele_complet, encodeur_texte, encodeur_image, nettoyeur, traiteur_image, device)
    
    nb_total = sum(p.numel() for p in pipeline.parameters())
    nb_train = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    logger.info(f" Pipeline assemblé : {nb_total:,} total, {nb_train:,} entraînables ({(nb_train/nb_total)*100:.1f}%)")
    
    return pipeline



def lancer_entrainement(
    train_loader, val_loader, test_loader=None,
    epochs=30, device=None, dossier_checkpoint=None,
    freeze_bert=True, unfreeze_2_couches=True,
    freeze_resnet=True, utiliser_lora=True,
) -> Dict:
    """Lance l'entraînement complet."""
    from sources.entrainement.fonctions_perte import creer_fonction_perte
    
    pipeline = assembler_pipeline(
        device=device, freeze_bert=freeze_bert,
        unfreeze_2_couches=unfreeze_2_couches,
        freeze_resnet=freeze_resnet, utiliser_lora=utiliser_lora,
    )
    
    perte_fn = creer_fonction_perte("focal_weighted", label_smoothing=0.15)
    
    trainer = EntraineurPipeline(
        pipeline=pipeline, modele=pipeline,
        train_loader=train_loader, val_loader=val_loader,
        perte_fn=perte_fn, epochs=epochs, device=device,
        dossier_checkpoint=dossier_checkpoint,
    )
    
    historique = trainer.entrainer()
    
    if test_loader is not None:
        resultats = trainer.evaluer_test(test_loader)
        logger.info(f" Test Acc: {resultats['test_acc']:.4f} | Test F1: {resultats['test_f1']:.4f}")
    
    return historique


# ══════════════════════════════════════════
# 4.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 70)
    print(" TEST — Pipeline (Cross-Attention + LoRA + 2 couches BERT)")
    print("=" * 70)
    
    pipeline = assembler_pipeline(device="cpu", utiliser_lora=True, unfreeze_2_couches=True)
    
    textes = ["explosion massive Beyrouth HTTPURL", "info trafic HASHTAG info", "publicité HASHTAG ad"]
    images = [None, None, None]
    
    pipeline.eval()
    with torch.no_grad():
        logits = pipeline(textes, images)
        classes = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        for i in range(3):
            print(f"   [{i}] Classe {classes[i].item()} ({cfg.classes.noms[classes[i].item()]})")
    
    pipeline.train()
    loss = pipeline(textes, images).sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in pipeline.parameters() if p.requires_grad)
    print(f"   Gradients : {'✓ OK' if has_grad else '✗ Échec'}")
    
    print(f"\n integration_pipeline.py — Tout OK !")