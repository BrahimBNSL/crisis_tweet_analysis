

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, classification_report

from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# 0.  COLLATE FUNCTION
# ══════════════════════════════════════════

def collate_fn_multimodal(batch: List[Dict]) -> Dict:
    
    texte       = [item["texte"] for item in batch]
    image       = [item["image"] for item in batch]
    classe      = torch.tensor([item["classe"] for item in batch], dtype=torch.long)
    has_image   = torch.tensor(
        [item["image"] is not None for item in batch], dtype=torch.bool
    )
    source      = [item.get("source", "") for item in batch]
    catastrophe = [item.get("catastrophe", "") for item in batch]
    idx         = torch.tensor([item.get("idx", 0) for item in batch], dtype=torch.long)

    return {
        "texte"      : texte,
        "image"      : image,
        "classe"     : classe,
        "has_image"  : has_image,
        "source"     : source,
        "catastrophe": catastrophe,
        "idx"        : idx,
    }


# ══════════════════════════════════════════
# 1.  PLANIFICATEUR DE LR  (Cosine + Warmup)
# ══════════════════════════════════════════

class WarmupCosineScheduler:
    

    def __init__(
        self,
        optimizer      : torch.optim.Optimizer,
        warmup_epochs  : int,
        total_epochs   : int,
        steps_per_epoch: int,
        lr_max         : float = 1e-3,
        lr_min         : float = 1e-7,
    ):
        self.warmup_steps = warmup_epochs  * steps_per_epoch
        self.total_steps  = total_epochs   * steps_per_epoch
        self.lr_max       = lr_max
        self.lr_min       = lr_min

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)).item())
            return self.lr_min / self.lr_max + (1 - self.lr_min / self.lr_max) * cosine

        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def step(self):
        self.scheduler.step()

    def get_lr(self) -> List[float]:
        return self.scheduler.get_last_lr()


# ══════════════════════════════════════════
# 2.  EARLY STOPPING  (sur F1-macro)
# ══════════════════════════════════════════

class EarlyStopping:
    

    def __init__(
        self,
        patience             : int   = 5,
        delta_min            : float = 1e-4,
        mode                 : str   = "max",
        sauvegarder_meilleur : bool  = True,
    ):
        self.patience              = patience
        self.delta_min             = delta_min
        self.mode                  = mode
        self.sauvegarder_meilleur  = sauvegarder_meilleur

        self.compteur         = 0
        self.meilleure_valeur = float("-inf") if mode == "max" else float("inf")
        self.early_stop       = False
        self.meilleur_etat    = None

        logger.info(
            f" EarlyStopping : patience={patience}, "
            f"delta={delta_min}, mode={mode}"
        )

    def __call__(self, valeur: float, modele: nn.Module) -> bool:
        amelioration = (
            valeur - self.meilleure_valeur > self.delta_min
            if self.mode == "max"
            else self.meilleure_valeur - valeur > self.delta_min
        )

        if amelioration:
            self.meilleure_valeur = valeur
            self.compteur         = 0
            if self.sauvegarder_meilleur:
                self.meilleur_etat = {
                    k: v.cpu().clone() for k, v in modele.state_dict().items()
                }
        else:
            self.compteur += 1
            if self.compteur >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restaurer_meilleur(self, modele: nn.Module) -> None:
        
        if self.meilleur_etat is not None:
            modele.load_state_dict(self.meilleur_etat)
            logger.info(" Meilleur modèle restauré !")


# ══════════════════════════════════════════
# 3.  ENTRAÎNEUR PRINCIPAL
# ══════════════════════════════════════════

class Entraineur:
   

    def __init__(
        self,
        modele                  : nn.Module,
        train_loader            : DataLoader,
        val_loader              : DataLoader,
        perte_fn                : nn.Module,
        epochs                  : Optional[int]   = None,
        lr_bert                 : Optional[float]  = None,
        lr_resnet               : Optional[float]  = None,
        lr_reste                : Optional[float]  = None,
        warmup_epochs           : Optional[int]   = None,
        gradient_clip           : Optional[float]  = None,
        patience_early_stopping : Optional[int]   = None,
        dossier_checkpoint      : Optional[Path]  = None,
        device                  : Optional[str]   = None,
        utiliser_wandb          : bool             = False,
        log_interval            : int              = 50,
    ):
        self.modele       = modele
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.perte_fn     = perte_fn

        self.epochs        = epochs        or cfg.entrainement.nb_epochs
        self.warmup_epochs = warmup_epochs or cfg.entrainement.nb_epochs_warmup
        self.gradient_clip = gradient_clip or cfg.entrainement.gradient_clip
        self.log_interval  = log_interval

        self.device = device or (
            "cuda" if torch.cuda.is_available() and cfg.entrainement.utiliser_gpu
            else "cpu"
        )
        self.modele = self.modele.to(self.device)

        # ── Optimiseur ──
        self.optimizer = self._creer_optimiseur(
            lr_bert   or cfg.entrainement.lr_bert,
            lr_resnet or cfg.entrainement.lr_resnet,
            lr_reste  or cfg.entrainement.lr_reste,
        )

        # ── Scheduler ──
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs   = self.warmup_epochs,
            total_epochs    = self.epochs,
            steps_per_epoch = len(train_loader),
            lr_max          = lr_reste or cfg.entrainement.lr_reste,
            lr_min          = cfg.entrainement.eta_min,
        )

        # ── Early stopping sur F1-macro (mode=max) ──
        self.early_stopping = EarlyStopping(
            patience             = patience_early_stopping or cfg.entrainement.patience,
            delta_min            = cfg.entrainement.delta_min,
            mode                 = "max",
        )

        # ── Checkpoint ──
        self.dossier_checkpoint = dossier_checkpoint or cfg.chemins.dossier_checkpoints
        self.dossier_checkpoint.mkdir(parents=True, exist_ok=True)

        # ── WandB ──
        self.utiliser_wandb = utiliser_wandb
        if self.utiliser_wandb:
            try:
                # pyrefly: ignore [missing-import]
                import wandb
                wandb.init(project=cfg.journalisation.projet_wandb)
                self.wandb = wandb
            except ImportError:
                logger.warning("  WandB non installé → logging désactivé.")
                self.utiliser_wandb = False

        # ── Historique ──
        self.historique = {
            "train_loss": [],
            "val_loss"  : [],
            "val_acc"   : [],
            "val_f1"    : [],
            "lr"        : [],
        }

        nb_params = sum(p.numel() for p in modele.parameters() if p.requires_grad)
        logger.info(" Entraineur initialisé :")
        logger.info(f"   • Device      : {self.device}")
        logger.info(f"   • Epochs      : {self.epochs}")
        logger.info(f"   • Warmup      : {self.warmup_epochs} epochs")
        logger.info(f"   • LR bert     : {cfg.entrainement.lr_bert}")
        logger.info(f"   • LR resnet   : {cfg.entrainement.lr_resnet}")
        logger.info(f"   • LR reste    : {cfg.entrainement.lr_reste}")
        logger.info(f"   • Grad clip   : {self.gradient_clip}")
        logger.info(f"   • Early stop  : F1-macro (mode=max)")
        logger.info(f"   • Params      : {nb_params:,}")

    # ──────────────────────────────────────
    def _creer_optimiseur(
        self,
        lr_bert  : float,
        lr_resnet: float,
        lr_reste : float,
    ) -> AdamW:
        
        params_bert   = []
        params_resnet = []
        params_reste  = []

        for name, param in self.modele.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if "encodeur_texte" in name_lower or "bert" in name_lower:
                params_bert.append(param)
            elif any(k in name_lower for k in ("encodeur_image", "resnet", "backbone")):
                params_resnet.append(param)
            else:
                params_reste.append(param)

        groupes = []
        if params_bert:
            groupes.append({"params": params_bert,   "lr": lr_bert,   "name": "BERTweet"})
        if params_resnet:
            groupes.append({"params": params_resnet, "lr": lr_resnet, "name": "ResNet"})
        if params_reste:
            groupes.append({"params": params_reste,  "lr": lr_reste,  "name": "Custom"})

        optimizer = AdamW(groupes, weight_decay=cfg.entrainement.weight_decay)

        logger.info("   Groupes d'optimiseur :")
        for g in groupes:
            logger.info(
                f"      {g['name']:<12} : {len(g['params'])} tensors, LR={g['lr']}"
            )

        return optimizer

    # ──────────────────────────────────────
    # Les trois méthodes suivantes sont des stubs.
    # EntraineurPipeline les surcharge avec les vrais forwards.
    # ──────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        raise NotImplementedError(
            "Utilisez EntraineurPipeline, pas Entraineur directement."
        )

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        raise NotImplementedError(
            "Utilisez EntraineurPipeline, pas Entraineur directement."
        )

    @torch.no_grad()
    def evaluer_test(self, test_loader: DataLoader) -> Dict:
        raise NotImplementedError(
            "Utilisez EntraineurPipeline, pas Entraineur directement."
        )

    # ──────────────────────────────────────
    def entrainer(self) -> Dict:
        
        logger.info(f"\n{'='*60}")
        logger.info(" DÉBUT DE L'ENTRAÎNEMENT")
        logger.info(f"{'='*60}")

        temps_debut = time.time()

        for epoch in range(1, self.epochs + 1):
            temps_epoch_debut = time.time()

            # ── Train ──
            train_loss = self._train_epoch(epoch)

            # ── Validation ──
            val_loss, val_acc, val_f1 = self._eval_epoch(self.val_loader)

            # ── Historique ──
            self.historique["train_loss"].append(train_loss)
            self.historique["val_loss"].append(val_loss)
            self.historique["val_acc"].append(val_acc)
            self.historique["val_f1"].append(val_f1)
            self.historique["lr"].append(self.scheduler.get_lr()[0])

            temps_epoch = time.time() - temps_epoch_debut
            logger.info(
                f" Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1 : {val_f1:.4f} | "
                f"Temps: {temps_epoch:.1f}s"
            )

            if self.utiliser_wandb:
                self.wandb.log({
                    "epoch"     : epoch,
                    "train_loss": train_loss,
                    "val_loss"  : val_loss,
                    "val_acc"   : val_acc,
                    "val_f1"    : val_f1,
                    "lr"        : self.scheduler.get_lr()[0],
                })

            # ── Early stopping sur F1-macro ──
            if self.early_stopping(val_f1, self.modele):
                logger.info(f" Early stopping déclenché à l'epoch {epoch}")
                break

        # ── Restaurer le meilleur checkpoint ──
        self.early_stopping.restaurer_meilleur(self.modele)

        self._sauvegarder_modele()
        self._sauvegarder_historique()

        temps_total = time.time() - temps_debut
        logger.info(f"\n Entraînement terminé en {temps_total / 60:.1f} minutes")
        logger.info(f"   Meilleur F1-macro : {self.early_stopping.meilleure_valeur:.4f}")

        return self.historique

    # ──────────────────────────────────────
    def _sauvegarder_modele(self) -> None:
        """Sauvegarde le modèle + optimiseur + historique."""
        chemin = self.dossier_checkpoint / "meilleur_modele.pt"
        torch.save(
            {
                "model_state_dict"    : self.modele.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "historique"          : self.historique,
            },
            chemin,
        )
        logger.info(f" Modèle sauvegardé : {chemin}")

    def _sauvegarder_historique(self) -> None:
        """Sauvegarde l'historique en JSON."""
        chemin = self.dossier_checkpoint / "historique_entrainement.json"
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self.historique, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Historique sauvegardé : {chemin}")


# ══════════════════════════════════════════
# 4.  ENTRAÎNEUR PIPELINE  (sous-classe opérationnelle)
# ══════════════════════════════════════════

class EntraineurPipeline(Entraineur):
   

    def __init__(self, pipeline, *args, **kwargs):
        kwargs.pop("pipeline", None)        
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline
        self.pipeline.train()

    # ──────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
       
        self.pipeline.train()
        perte_totale = 0.0
        nb_batches   = 0

        for batch_idx, batch in enumerate(self.train_loader):
            textes  = batch["texte"]
            images  = batch["image"]
            cibles  = batch["classe"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.pipeline(textes, images)      
            loss   = self.perte_fn(logits, cibles)

            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.pipeline.parameters(), self.gradient_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            perte_totale += loss.item()
            nb_batches   += 1

            if batch_idx % self.log_interval == 0:
                lr_actuelle = self.scheduler.get_lr()[0]
                logger.info(
                    f"   Epoch {epoch:3d} | "
                    f"Batch {batch_idx:4d}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | LR: {lr_actuelle:.2e}"
                )

        return perte_totale / nb_batches

    # ──────────────────────────────────────
    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        
        self.pipeline.eval()
        perte_totale = 0.0
        correct      = 0
        total        = 0
        all_preds    = []
        all_cibles   = []

        for batch in loader:
            textes  = batch["texte"]
            images  = batch["image"]
            cibles  = batch["classe"].to(self.device)

            logits  = self.pipeline(textes, images)  
            loss    = self.perte_fn(logits, cibles)

            perte_totale += loss.item()
            predits = torch.argmax(logits, dim=1)
            correct += (predits == cibles).sum().item()
            total   += cibles.size(0)

            all_preds.extend(predits.cpu().tolist())
            all_cibles.extend(cibles.cpu().tolist())

        perte_moyenne = perte_totale / len(loader)
        accuracy      = correct / total if total > 0 else 0.0
        f1_macro      = f1_score(
            all_cibles, all_preds, average="macro", zero_division=0
        )

        return perte_moyenne, accuracy, f1_macro

    # ──────────────────────────────────────
    @torch.no_grad()
    def evaluer_test(self, test_loader: DataLoader) -> Dict:
       
        self.pipeline.eval()
        perte_totale = 0.0
        correct      = 0
        total        = 0
        all_preds    = []
        all_cibles   = []

        for batch in test_loader:
            textes  = batch["texte"]
            images  = batch["image"]
            cibles  = batch["classe"].to(self.device)

            logits  = self.pipeline(textes, images)   
            loss    = self.perte_fn(logits, cibles)

            perte_totale += loss.item()
            predits = torch.argmax(logits, dim=1)
            correct += (predits == cibles).sum().item()
            total   += cibles.size(0)

            all_preds.extend(predits.cpu().tolist())
            all_cibles.extend(cibles.cpu().tolist())

        test_loss = perte_totale / len(test_loader)
        test_acc  = correct / total if total > 0 else 0.0
        test_f1   = f1_score(
            all_cibles, all_preds, average="macro", zero_division=0
        )

        rapport = classification_report(
            all_cibles,
            all_preds,
            target_names=cfg.classes.noms,
            zero_division=0,
        )

        logger.info(
            f" Test — Loss: {test_loss:.4f} | "
            f"Acc: {test_acc:.4f} | F1: {test_f1:.4f}"
        )
        logger.info(f"\n{rapport}")

        return {
            "test_loss": test_loss,
            "test_acc" : test_acc,
            "test_f1"  : test_f1,
            "rapport"  : rapport,
            "preds"    : all_preds,
            "cibles"   : all_cibles,
        }


# ══════════════════════════════════════════
# 5.  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def creer_entraineur(
    modele      : nn.Module,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    perte_fn    : nn.Module,
    **kwargs,
) -> Entraineur:
    
    return Entraineur(
        modele       = modele,
        train_loader = train_loader,
        val_loader   = val_loader,
        perte_fn     = perte_fn,
        **kwargs,
    )


def charger_checkpoint(
    chemin   : Path,
    modele   : nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    
    checkpoint = torch.load(chemin, map_location="cpu", weights_only=False)
    modele.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"📦 Checkpoint chargé : {chemin}")
    return checkpoint.get("historique", {})


# ══════════════════════════════════════════
# 6.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print(" TEST — boucle_entrainement.py")
    print("=" * 70)

    # ── Test via le vrai pipeline (recommandé) ──
    try:
        from sources.entrainement.integration_pipeline import lancer_entrainement
        from sources.donnees.chargeur_donnees import creer_data_loaders

        print("\n Chargement des données...")
        train_loader, val_loader, test_loader = creer_data_loaders(
            batch_size=4,
            nb_workers=0,
        )

        print("\n Lancement de l'entraînement (2 epochs, CPU)...")
        historique = lancer_entrainement(
            train_loader       = train_loader,
            val_loader         = val_loader,
            test_loader        = test_loader,
            epochs             = 2,
            device             = "cpu",
            freeze_bert        = True,
            freeze_resnet      = True,
            utiliser_lora      = False,   
        )

        print(f"\n Historique :")
        for cle, valeurs in historique.items():
            if valeurs:
                print(f"   {cle:<15} : {[f'{v:.4f}' for v in valeurs]}")

        best_f1 = max(historique["val_f1"]) if historique["val_f1"] else 0.0
        print(f"\n Test OK — Meilleur F1-macro : {best_f1:.4f}")

    except Exception as e:
        print(f"\n  Test pipeline complet impossible : {e}")
        print("   → Vérifiez que les données et les modules sont disponibles.")
        print("   → Pour tester uniquement l'infrastructure, lancez integration_pipeline.py")