"""
jeu_de_donnees.py
─────────────────
Dataset PyTorch unifié avec augmentation avancée :
    • Texte : WordNet synonymes, EDA, rétro-traduction (anglais)
    • Image : rotation, flip, color jitter (pour éviter overfitting)
    • Classe Urgence ×3 avec variations
    • CrisisLexT26 : +28K tweets de crise (texte seul)

Sources :
    1. CrisisMMD_v2.0  —  Tweets + images Twitter
    2. Multimodal       —  Paires image/texte par catégorie
    3. CrisisLexT26     —  Tweets de 26 crises (texte seul)

Mapping vers 3 classes :
    0 — Urgence (Affected individuals)
    1 — Info / signalement
    2 — Non pertinent
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps

# Ajouter la racine au path
RACINE_PROJET = Path(__file__).resolve().parents[2]
if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))

from sources.donnees.traitement_texte import NettoyeurTweet
from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION & CONSTANTES
# ════════════════════════════════════════════════════════════════════

RACINE_DONNEES = cfg.chemins.dossier_brut_crisismmd.parent
CRISIMMD_ANNOTATIONS = cfg.chemins.annotations_crisismmd
CRISIMMD_IMAGES = cfg.chemins.images_crisismmd
MULTIMODAL_RACINE = cfg.chemins.dossier_brut_multimodal
DOSSIER_TRAITEES = cfg.chemins.dossier_traitees
DOSSIER_TRAITEES.mkdir(parents=True, exist_ok=True)

# Charger lexiques.json
CHEMIN_SYNONYMES = RACINE_PROJET / "lexiques.json"

def charger_synonymes() -> Dict[str, List[str]]:
    try:
        with open(CHEMIN_SYNONYMES, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

SYNONYMES = charger_synonymes()

# Initialiser NLTK
try:
    import nltk
    from nltk.corpus import wordnet, stopwords
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
    NLTK_OK = True
except ImportError:
    NLTK_OK = False
    STOP_WORDS = set()

# Initialiser deep-translator
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False

# ═══════════════════════════════════════
# MAPPINGS
# ═══════════════════════════════════════

# CrisisLexT26 → 3 classes
MAPPING_CRISISLEX = {
    "Affected individuals": 0,          # Urgence
    "Other Useful Information": 1,      # Info
    "Infrastructure and utilities": 1,  # Info
    "Caution and advice": 1,            # Info
    "Donations and volunteering": 1,    # Info
    "Sympathy and support": 2,          # Non pertinent
    "Not applicable": 2,               # Non pertinent
    # "Not labeled" : EXCLU
}

# CrisisMMD → 3 classes
MAPPING_TEXTE_CRISIMMD: Dict[str, int] = {
    "rescue_volunteering_or_donation_effort": 0,
    "injured_or_dead_people": 0,
    "affected_individuals": 0,
    "other_relevant_information": 1,
    "infrastructure_and_utility_damage": 1,
    "not_humanitarian": 2,
}

# Multimodal → 3 classes
MAPPING_DOSSIER_MULTIMODAL: Dict[str, int] = {
    "human_damage": 0,
    "damaged_infrastructure": 1,
    "damaged_nature": 1,
    "fires": 1,
    "flood": 1,
    "non_damage": 2,
}

NOMS_CLASSES: Dict[int, str] = {
    0: "Urgence",
    1: "Info / signalement",
    2: "Non pertinent",
}


# ════════════════════════════════════════════════════════════════════
# 2.  AUGMENTATION TEXTE (Anglais)
# ════════════════════════════════════════════════════════════════════

def synonymes_wordnet(mot: str) -> List[str]:
    if not NLTK_OK:
        return []
    synonyms = set()
    for syn in wordnet.synsets(mot.lower()):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != mot.lower():
                synonyms.add(name)
    return list(synonyms)


def remplacer_synonymes_en(texte: str, p: float = 0.3) -> str:
    mots = texte.split()
    for i, mot in enumerate(mots):
        mot_clean = mot.lower().strip('.,!?;:"\'()[]')
        candidats = SYNONYMES.get(mot_clean, [])
        if not candidats and NLTK_OK:
            candidats = synonymes_wordnet(mot_clean)
        if candidats and random.random() < p:
            synonyme = random.choice(candidats)
            if mot[0].isupper():
                synonyme = synonyme.capitalize()
            suffix = mot[len(mot_clean):]
            mots[i] = synonyme + suffix
    return ' '.join(mots)


def supprimer_stopwords_en(texte: str, p: float = 0.2) -> str:
    if not STOP_WORDS:
        return texte
    mots = texte.split()
    return ' '.join([mot for mot in mots if mot.lower() not in STOP_WORDS or random.random() > p])


def echanger_mots(texte: str) -> str:
    mots = texte.split()
    if len(mots) < 2:
        return texte
    i = random.randint(0, len(mots) - 2)
    mots[i], mots[i+1] = mots[i+1], mots[i]
    return ' '.join(mots)


def supprimer_mots(texte: str, p: float = 0.1) -> str:
    mots = texte.split()
    if len(mots) < 5:
        return texte
    nb_garder = max(3, int(len(mots) * (1 - p)))
    return ' '.join(random.sample(mots, nb_garder))


def retro_traduire_en(texte: str) -> str:
    if not TRANSLATOR_OK or len(texte) < 10:
        return texte
    try:
        fr = GoogleTranslator(source='en', target='fr').translate(texte[:500])
        en = GoogleTranslator(source='fr', target='en').translate(fr)
        return en
    except Exception:
        return texte


def augmenter_texte_en(texte: str) -> str:
    techniques = [
        lambda t: remplacer_synonymes_en(t, p=0.3),
        lambda t: supprimer_stopwords_en(t, p=0.15),
        lambda t: echanger_mots(t),
        lambda t: supprimer_mots(t, p=0.1),
    ]
    if TRANSLATOR_OK and random.random() < 0.2 and len(texte) > 20:
        return retro_traduire_en(texte)
    return random.choice(techniques)(texte)


# ════════════════════════════════════════════════════════════════════
# 3.  AUGMENTATION IMAGE
# ════════════════════════════════════════════════════════════════════

def augmenter_image(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = image.rotate(random.uniform(-15, 15), expand=False, fillcolor=0)
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
    return image


# ════════════════════════════════════════════════════════════════════
# 4.  CHARGEMENT DES DONNÉES BRUTES
# ════════════════════════════════════════════════════════════════════

def charger_crisimmd(fichiers_tsv=None, utiliser_cache=True) -> pd.DataFrame:
    cache_path = DOSSIER_TRAITEES / "crisimmd_fusionne.csv"
    if utiliser_cache and cache_path.exists():
        logger.info(f"📦 Cache CrisisMMD chargé")
        return pd.read_csv(cache_path)
    
    if fichiers_tsv is None:
        fichiers_tsv = [f for f in os.listdir(CRISIMMD_ANNOTATIONS) if f.endswith('.tsv') and not f.startswith('._')]
    
    logger.info(f"📂 Chargement de {len(fichiers_tsv)} fichiers CrisisMMD...")
    dataframes = []
    for fichier in sorted(fichiers_tsv):
        df = pd.read_csv(CRISIMMD_ANNOTATIONS / fichier, sep='\t')
        df['source'] = 'crisimmd'
        df['catastrophe'] = fichier.replace('_final_data.tsv', '')
        dataframes.append(df)
    
    df_total = pd.concat(dataframes, ignore_index=True)
    df_total['classe'] = df_total.apply(
        lambda row: fusionner_labels_crisimmd(row['text_human'], row['image_human']), axis=1
    )
    df_total['texte'] = df_total['tweet_text']
    
    if utiliser_cache:
        df_total.to_csv(cache_path, index=False)
    logger.info(f"✅ CrisisMMD : {len(df_total):,} tweets")
    return df_total


def charger_multimodal() -> pd.DataFrame:
    logger.info("📂 Chargement Multimodal...")
    lignes = []
    
    for dossier in os.listdir(MULTIMODAL_RACINE):
        chemin = MULTIMODAL_RACINE / dossier
        if not chemin.is_dir():
            continue
        classe = MAPPING_DOSSIER_MULTIMODAL.get(dossier)
        if classe is None:
            continue
        
        img_dir = chemin / "images"
        txt_dir = chemin / "text"
        if not img_dir.exists() or not txt_dir.exists():
            continue
        
        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')) or img_file.startswith('._'):
                continue
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = txt_dir / txt_file
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    texte = f.read().strip()
                if texte:
                    lignes.append({
                        'image_path': str((img_dir / img_file).resolve()),
                        'texte': texte, 'classe': classe,
                        'source': 'multimodal', 'catastrophe': dossier,
                    })
    
    df = pd.DataFrame(lignes)
    logger.info(f"✅ Multimodal : {len(df):,} paires")
    return df


def charger_crisislex_t26(dossier=None) -> pd.DataFrame:
    """Charge les tweets labellisés de CrisisLexT26 (texte seul, sans images)."""
    if dossier is None:
        dossier = RACINE_DONNEES / "CrisisLexT26"
    
    logger.info("📂 Chargement CrisisLexT26...")
    lignes = []
    
    for csv_file in Path(dossier).rglob("*_labeled.csv"):
        df = pd.read_csv(csv_file)
        nom_crise = csv_file.parent.name.replace("_", " ").title()
        
        for _, row in df.iterrows():
            type_info = str(row.get(" Information Type", "")).strip()
            
            # Ignorer les tweets non labellisés
            if type_info == "Not labeled":
                continue
            
            if type_info in MAPPING_CRISISLEX:
                classe = MAPPING_CRISISLEX[type_info]
            else:
                continue  # Ignorer les types inconnus
            
            lignes.append({
                'texte': str(row.get(" Tweet Text", "")).strip(),
                'classe': classe,
                'source': 'crisislex',
                'catastrophe': nom_crise,
                'image_path': None,  # Pas d'image
            })
    
    df = pd.DataFrame(lignes)
    logger.info(f"✅ CrisisLexT26 : {len(df):,} tweets")
    return df


def fusionner_labels_crisimmd(text_human, image_human) -> int:
    ct = MAPPING_TEXTE_CRISIMMD.get(str(text_human)) if pd.notna(text_human) else None
    ci = MAPPING_TEXTE_CRISIMMD.get(str(image_human)) if pd.notna(image_human) else None
    if ct is not None and ci is not None:
        return min(ct, ci)
    return ct if ct is not None else (ci if ci is not None else 2)


# ════════════════════════════════════════════════════════════════════
# 5.  AUGMENTATION DE LA CLASSE URGENCE
# ════════════════════════════════════════════════════════════════════

def augmenter_classe_urgence(df: pd.DataFrame, facteur: int = 3) -> pd.DataFrame:
    logger.info(f"🔧 Augmentation Urgence (x{facteur})...")
    
    df_urgence = df[df['classe'] == 0].copy()
    nb_original = len(df_urgence)
    augmentes = []
    random.seed(42)
    
    for copie in range(facteur - 1):
        logger.info(f"   Copie {copie+1}/{facteur-1}...")
        for _, row in df_urgence.iterrows():
            nouvelle = row.copy()
            nouvelle['texte'] = augmenter_texte_en(str(row['texte']))
            
            if 'image_path' in row and pd.notna(row['image_path']):
                try:
                    img = Image.open(row['image_path']).convert('RGB')
                    img_aug = augmenter_image(img)
                    chemin_orig = Path(row['image_path'])
                    nouveau_nom = f"{chemin_orig.stem}_aug{copie}{chemin_orig.suffix}"
                    nouveau_chemin = chemin_orig.parent / nouveau_nom
                    img_aug.save(nouveau_chemin)
                    nouvelle['image_path'] = str(nouveau_chemin)
                except Exception:
                    pass
            
            augmentes.append(nouvelle)
    
    df_augmente = pd.DataFrame(augmentes)
    df_final = pd.concat([df, df_augmente], ignore_index=True)
    
    nb_final = len(df_final[df_final['classe'] == 0])
    logger.info(f"   ✅ Urgence : {nb_original} → {nb_final}")
    return df_final


# ════════════════════════════════════════════════════════════════════
# 6.  NETTOYAGE
# ════════════════════════════════════════════════════════════════════

def nettoyer_textes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🧹 Nettoyage des textes...")
    nettoyeur = NettoyeurTweet(
        supprimer_urls=True, remplacer_mentions=True,
        normaliser_hashtags=True, expandre_abreviations=True,
        remplacer_nombres=True, reduire_repetitions=True,
        mettre_minuscules=False,
    )
    df['texte_nettoye'] = df['texte'].apply(lambda x: nettoyeur.nettoyer(str(x)) if pd.notna(x) else "")
    return df


# ════════════════════════════════════════════════════════════════════
# 7.  SAUVEGARDE / CHARGEMENT CSV
# ════════════════════════════════════════════════════════════════════

def sauvegarder_splits_csv(df_train, df_val, df_test, dossier_sortie=None):
    if dossier_sortie is None:
        dossier_sortie = DOSSIER_TRAITEES
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    for nom, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        chemin = dossier_sortie / f"{nom}.csv"
        df.to_csv(chemin, index=False)
        logger.info(f"   • {nom}.csv : {len(df):,} lignes")


def charger_splits_csv(dossier_entree=None):
    if dossier_entree is None:
        dossier_entree = DOSSIER_TRAITEES
    dfs = {}
    for nom in ["train", "val", "test"]:
        chemin = dossier_entree / f"{nom}.csv"
        if not chemin.exists():
            raise FileNotFoundError(f"❌ {nom}.csv introuvable")
        dfs[nom] = pd.read_csv(chemin)
    logger.info(f"📂 Splits : {len(dfs['train']):,} / {len(dfs['val']):,} / {len(dfs['test']):,}")
    return dfs["train"], dfs["val"], dfs["test"]


# ════════════════════════════════════════════════════════════════════
# 8.  PRÉPARATION DES DONNÉES
# ════════════════════════════════════════════════════════════════════

def preparer_donnees(
    proportions=(0.70, 0.15, 0.15),
    graine=42,
    charger_crisimmd_flag=True,
    charger_multimodal_flag=True,
    charger_crisislex_flag=True,
    augmenter_urgence=True,
    facteur_augmentation=3,
    sauvegarder_csv=True,
    dossier_csv=None,
):
    logger.info("=" * 60)
    logger.info(f"📦 PRÉPARATION DES DONNÉES (+CrisisLexT26, x{facteur_augmentation})")
    logger.info("=" * 60)
    
    # Charger toutes les sources
    dataframes = []
    if charger_crisimmd_flag:
        dataframes.append(charger_crisimmd())
    if charger_multimodal_flag:
        dataframes.append(charger_multimodal())
    if charger_crisislex_flag:
        dataframes.append(charger_crisislex_t26())
    
    if not dataframes:
        raise ValueError("Aucune source activée !")
    
    df_total = pd.concat(dataframes, ignore_index=True)
    logger.info(f"📦 Total fusionné : {len(df_total):,}")
    
    # Augmenter Urgence
    if augmenter_urgence:
        df_total = augmenter_classe_urgence(df_total, facteur=facteur_augmentation)
    
    # Nettoyer
    df_total = nettoyer_textes_dataframe(df_total)
    
    # Colonnes
    colonnes = ['texte', 'texte_nettoye', 'classe', 'source', 'catastrophe']
    if 'image_path' in df_total.columns:
        colonnes.append('image_path')
    df_total = df_total[[c for c in colonnes if c in df_total.columns]]
    
    # Split
    df_total = df_total.sample(frac=1, random_state=graine).reset_index(drop=True)
    n = len(df_total)
    n_train, n_val = int(n * proportions[0]), int(n * proportions[1])
    
    df_train = df_total.iloc[:n_train].reset_index(drop=True)
    df_val = df_total.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_total.iloc[n_train + n_val:].reset_index(drop=True)
    
    for nom, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        logger.info(f"   {nom:<6} : {len(df):,} lignes")
        for cl in range(3):
            nb = (df['classe'] == cl).sum()
            logger.info(f"      Classe {cl} ({NOMS_CLASSES[cl]}): {nb:>6,} ({nb/len(df)*100:.1f}%)")
    
    if sauvegarder_csv:
        sauvegarder_splits_csv(df_train, df_val, df_test, dossier_csv)
    
    logger.info("✅ Données prêtes !")
    return df_train, df_val, df_test


# ════════════════════════════════════════════════════════════════════
# 9.  DATASET PYTHON
# ════════════════════════════════════════════════════════════════════

class JeuDeDonneesCrise(Dataset):
    def __init__(self, donnees, nettoyeur=None, utiliser_texte_nettoye=True, augmenter=False):
        self.donnees = donnees.reset_index(drop=True)
        self.nettoyeur = nettoyeur
        self.utiliser_texte_nettoye = utiliser_texte_nettoye and ('texte_nettoye' in donnees.columns)
        self.augmenter = augmenter
        self.nb_classes = 3
        self.distribution = self.donnees['classe'].value_counts().to_dict()
        logger.info(f"📊 Dataset : {len(self)} échantillons")
    
    def __len__(self):
        return len(self.donnees)
    
    def __getitem__(self, idx):
        ligne = self.donnees.iloc[idx]
        
        if self.utiliser_texte_nettoye and pd.notna(ligne.get('texte_nettoye')):
            texte = str(ligne['texte_nettoye'])
        else:
            texte = str(ligne.get('texte', ''))
            if self.nettoyeur:
                texte = self.nettoyeur.nettoyer(texte)
        
        image = None
        if 'image_path' in ligne and pd.notna(ligne['image_path']):
            chemin = Path(str(ligne['image_path']))
            if chemin.exists():
                try:
                    image = Image.open(chemin).convert('RGB')
                    if self.augmenter and random.random() < 0.3:
                        image = augmenter_image(image)
                except Exception:
                    pass
        
        return {
            'texte': texte, 'image': image,
            'classe': int(ligne['classe']),
            'source': str(ligne.get('source', '')),
            'catastrophe': str(ligne.get('catastrophe', '')),
            'idx': idx,
        }
    
    def get_stats(self):
        return {'nb_echantillons': len(self), 'distribution': self.distribution, 'classes': NOMS_CLASSES}


# ════════════════════════════════════════════════════════════════════
# 10.  TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 70)
    print("🧪 TEST — Dataset avec CrisisLexT26 + augmentation")
    print("=" * 70)
    
    df_train, df_val, df_test = preparer_donnees(
        charger_crisislex_flag=True,
        augmenter_urgence=True,
        facteur_augmentation=3,
        sauvegarder_csv=True,
    )
    
    print(f"\n✅ Colonnes : {list(df_train.columns)}")
    for cl in range(3):
        nb = (df_train['classe'] == cl).sum()
        print(f"   Classe {cl} ({NOMS_CLASSES[cl]}): {nb}")
    
    print(f"\n✅ jeu_de_donnees.py — Tout OK !")