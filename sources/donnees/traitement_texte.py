

import re
import unicodedata
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from sources.utilitaires.configuration import cfg



ABREVIATIONS: Dict[str, str] = {

    # ── Urgence & secours ──────────────────────────────────────────
    "sos":          "save our souls",
    "asap":         "as soon as possible",
    "pls":          "please",
    "plz":          "please",
    "ppl":          "people",
    "hlp":          "help",
    "hlep":         "help",
    "plssss":       "please",
    "plsss":        "please",
    "plss":         "please",
    "omg":          "oh my god",
    "omfg":         "oh my god",
    "wtf":          "what the hell",
    "smh":          "shaking my head",
    "rn":           "right now",
    "atm":          "at the moment",
    "asf":          "as hell",
    "af":           "very",
    "imo":          "in my opinion",
    "imho":         "in my humble opinion",
    "fyi":          "for your information",
    "tbh":          "to be honest",

    # ── Médias & communication ─────────────────────────────────────
    "rt":           "retweet",
    "dm":           "direct message",
    "lol":          "laughing out loud",
    "lmao":         "laughing",
    "rofl":         "laughing",
    "ikr":          "i know right",
    "irl":          "in real life",
    "idk":          "i do not know",
    "idc":          "i do not care",
    "ngl":          "not going to lie",
    "nvm":          "never mind",
    "btw":          "by the way",
    "bc":           "because",
    "b/c":          "because",
    "w/":           "with",
    "w/o":          "without",
    "vs":           "versus",
    "aka":          "also known as",
    "etc":          "etcetera",

    # ── Institutions & acteurs de crise ───────────────────────────
    "govt":         "government",
    "gov":          "government",
    "gov't":        "government",
    "min":          "ministry",
    "dept":         "department",
    "intl":         "international",
    "natl":         "national",
    "org":          "organization",
    "ngo":          "non-governmental organization",
    "un":           "united nations",
    "wfp":          "world food programme",
    "who":          "world health organization",
    "icrc":         "international committee of the red cross",
    "ifrc":         "international federation of red cross",

    # ── Géographie & localisation ──────────────────────────────────
    "leb":          "Lebanon",
    "lbn":          "Lebanon",
    "bey":          "Beirut",
    "brt":          "Beirut",
    "nr":           "near",
    "b/w":          "between",
    "n":            "north",
    "s":            "south",
    "e":            "east",
    "w":            "west",
    "ave":          "avenue",
    "blvd":         "boulevard",
    "rd":           "road",
    "st":           "street",

    # ── Catastrophes & urgences médicales ─────────────────────────
    "eq":           "earthquake",
    "quake":        "earthquake",
    "hur":          "hurricane",
    "evac":         "evacuation",
    "emer":         "emergency",
    "emerg":        "emergency",
    "hosp":         "hospital",
    "med":          "medical",
    "inj":          "injured",
    "injrd":        "injured",
    "surv":         "survivor",
    "vic":          "victim",
    "resc":         "rescue",
    "vol":          "volunteer",
    "dist":         "distribution",
    "distrib":      "distribution",
    "dmg":          "damage",
    "dest":         "destroyed",
    "destr":        "destroyed",
    "struct":       "structure",
    "infra":        "infrastructure",
    "bldg":         "building",
    "bld":          "building",
    "flr":          "floor",
    "ppl":          "people",
    "approx":       "approximately",
    "est":          "estimated",
    "conf":         "confirmed",
    "unconf":       "unconfirmed",
    "rep":          "reported",
    "rptd":         "reported",
    "upd":          "update",
    "info":         "information",
    "req":          "request",
    "avail":        "available",
    "unavail":      "unavailable",
    "acc":          "access",
    "loc":          "location",
    "coord":        "coordinates",
    "lat":          "latitude",
    "lng":          "longitude",
    "asst":         "assistance",
    "supp":         "support",
    "pkg":          "package",
    "pkg":          "package",
    "amt":          "amount",
    "qty":          "quantity",
    "cnt":          "count",
    "num":          "number",
    "tmp":          "temporary",
    "temp":         "temporary",
    "perm":         "permanent",
    "hrs":          "hours",
    "hr":           "hour",
    "min":          "minutes",
    "sec":          "seconds",
    "wk":           "week",
    "wks":          "weeks",
    "mo":           "month",
    "yr":           "year",
    "yrs":          "years",

    # ── Abréviations françaises ────────────────────────────────────
    "svp":          "s'il vous plaît",
    "stp":          "s'il te plaît",
    "tjrs":         "toujours",
    "tjr":          "toujours",
    "bcp":          "beaucoup",
    "qd":           "quand",
    "pr":           "pour",
    "pk":           "pourquoi",
    "pq":           "pourquoi",
    "mnt":          "maintenant",
    "mm":           "même",
    "dc":           "donc",
    "jsp":          "je ne sais pas",
    "jvp":          "je vais pas",
    "slt":          "salut",
    "bjr":          "bonjour",
    "bsr":          "bonsoir",
    "mrc":          "merci",
    "pb":           "problème",
    "nb":           "nota bene",
    "cf":           "confer",
    "ds":           "dans",
    "ss":           "sans",
    "tt":           "tout",
    "ts":           "tous",
    "tjs":          "toujours",
    "qque":         "quelque",
    "qqch":         "quelque chose",
    "qqn":          "quelqu'un",
    "env":          "environ",
    "infos":        "informations",
    "urgts":        "urgents",
    "urgt":         "urgent",
    "secours":      "secours",
    "dpt":          "département",
    "gouv":         "gouvernement",
    "min":          "ministère",

    
    "lbnan":        "Lebanon",
    "byrt":         "Beirut",
    "inshallah":    "god willing",
    "yalla":        "let us go",
    "hamdella":     "thank god",
}


# ══════════════════════════════════════════
# 1.  NETTOYEUR DE TWEETS
# ══════════════════════════════════════════

class NettoyeurTweet:


    # Tokens spéciaux reconnus par BERTweet
    TOKEN_USER   = "USER"
    TOKEN_URL    = "HTTPURL"
    TOKEN_NUMBER = "NUMBER"

    # Regex de base compilées une seule fois
    _RE_URL     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _RE_MENTION = re.compile(r"@\w{1,50}")
    _RE_HASHTAG = re.compile(r"#(\w+)")
    _RE_NOMBRE  = re.compile(r"\b\d+([.,]\d+)*\b")
    _RE_ESPACES = re.compile(r"\s+")
    _RE_REPETE  = re.compile(r"(.)\1{3,}")

    def __init__(
        self,
        supprimer_urls:        bool = True,
        remplacer_mentions:    bool = True,
        normaliser_hashtags:   bool = True,
        expandre_abreviations: bool = True,   # ← NOUVEAU flag
        remplacer_nombres:     bool = True,
        reduire_repetitions:   bool = True,
        mettre_minuscules:     bool = False,
        abreviations_custom:   Optional[Dict[str, str]] = None,
    ):
        self.supprimer_urls        = supprimer_urls
        self.remplacer_mentions    = remplacer_mentions
        self.normaliser_hashtags   = normaliser_hashtags
        self.expandre_abreviations = expandre_abreviations
        self.remplacer_nombres     = remplacer_nombres
        self.reduire_repetitions   = reduire_repetitions
        self.mettre_minuscules     = mettre_minuscules

        
        self._dict_abrev: Dict[str, str] = {**ABREVIATIONS}
        if abreviations_custom:
            self._dict_abrev.update(abreviations_custom)

       
        if self.expandre_abreviations:
            self._re_abrev = self._construire_regex_abrev()

    # ──────────────────────────────────────
    def _construire_regex_abrev(self) -> re.Pattern:
       
        tokens = sorted(self._dict_abrev.keys(), key=len, reverse=True)
       
        tokens_echappes = [re.escape(t) for t in tokens]
        pattern = r"\b(" + "|".join(tokens_echappes) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def _expandre(self, match: re.Match) -> str:
        
        mot       = match.group(0)
        expansion = self._dict_abrev[mot.lower()]
        
        if mot[0].isupper():
            expansion = expansion.capitalize()
        return expansion

    # ──────────────────────────────────────
    def nettoyer(self, texte: str) -> str:
        
        if not isinstance(texte, str) or not texte.strip():
            return ""

        # 1. Unicode NFC
        texte = unicodedata.normalize("NFC", texte)

        # 2. URLs
        if self.supprimer_urls:
            texte = self._RE_URL.sub(self.TOKEN_URL, texte)

        # 3. Mentions
        if self.remplacer_mentions:
            texte = self._RE_MENTION.sub(self.TOKEN_USER, texte)

        # 4. Hashtags  →  HASHTAG <mot>
        if self.normaliser_hashtags:
            texte = self._RE_HASHTAG.sub(lambda m: f"HASHTAG {m.group(1)}", texte)

        # 5. Expansion des abréviations
        #    (après hashtags/mentions pour ne pas toucher aux tokens spéciaux)
        if self.expandre_abreviations:
            texte = self._re_abrev.sub(self._expandre, texte)

        # 6. Nombres
        if self.remplacer_nombres:
            texte = self._RE_NOMBRE.sub(self.TOKEN_NUMBER, texte)

        # 7. Réduction des caractères répétés  (loooool → looool)
        if self.reduire_repetitions:
            texte = self._RE_REPETE.sub(r"\1\1\1", texte)

        # 8. Minuscules (optionnel — BERTweet est case-sensitive)
        if self.mettre_minuscules:
            texte = texte.lower()

        # 9. Normalisation des espaces
        texte = self._RE_ESPACES.sub(" ", texte).strip()

        return texte

    def nettoyer_batch(self, textes: List[str]) -> List[str]:
        
        return [self.nettoyer(t) for t in textes]

    # ──────────────────────────────────────
    def ajouter_abreviations(self, nouvelles: Dict[str, str]) -> None:
        
        self._dict_abrev.update({k.lower(): v for k, v in nouvelles.items()})
        self._re_abrev = self._construire_regex_abrev()

    def lister_abreviations(self) -> Dict[str, str]:
       
        return dict(self._dict_abrev)


# ══════════════════════════════════════════
# 2.  TOKENISEUR BERTWEET
# ══════════════════════════════════════════

class TokeniseurBERTweet:
   

    def __init__(self, longueur_max: Optional[int] = None):
        self.longueur_max = longueur_max or cfg.texte.longueur_max
        self.nom_modele   = cfg.texte.nom_modele

        # Chargement du tokenizer BERTweet
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.nom_modele,
            normalization=True,   # normalisation BERTweet interne (emoji→texte)
            use_fast=True,
        )

    # ──────────────────────────────────────
    def tokeniser(self, texte: str) -> Dict[str, torch.Tensor]:
        
        encodage = self.tokenizer(
            texte,
            max_length       = self.longueur_max,
            padding          = "max_length",
            truncation       = True,
            return_tensors   = "pt",
            return_attention_mask = True,
        )
        return {
            "input_ids":      encodage["input_ids"].squeeze(0),       # [L]
            "attention_mask": encodage["attention_mask"].squeeze(0),  # [L]
        }

    def tokeniser_batch(
        self,
        textes: List[str],
    ) -> Dict[str, torch.Tensor]:
       
        encodage = self.tokenizer(
            textes,
            max_length       = self.longueur_max,
            padding          = "max_length",
            truncation       = True,
            return_tensors   = "pt",
            return_attention_mask = True,
        )
        return {
            "input_ids":      encodage["input_ids"],       # [B, L]
            "attention_mask": encodage["attention_mask"],  # [B, L]
        }

    # ──────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def id_pad(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def id_cls(self) -> int:
        return self.tokenizer.cls_token_id

    @property
    def id_sep(self) -> int:
        return self.tokenizer.sep_token_id


# ══════════════════════════════════════════
# 3.  PIPELINE COMPLET  (nettoyage + tokenisation)
# ══════════════════════════════════════════

class PipelineTexte:


    def __init__(
        self,
        longueur_max:        Optional[int]  = None,
        supprimer_urls:      bool = True,
        remplacer_mentions:  bool = True,
        normaliser_hashtags: bool = True,
        remplacer_nombres:   bool = True,
        reduire_repetitions: bool = True,
        mettre_minuscules:   bool = False,
    ):
        self.nettoyeur  = NettoyeurTweet(
            supprimer_urls      = supprimer_urls,
            remplacer_mentions  = remplacer_mentions,
            normaliser_hashtags = normaliser_hashtags,
            remplacer_nombres   = remplacer_nombres,
            reduire_repetitions = reduire_repetitions,
            mettre_minuscules   = mettre_minuscules,
        )
        self.tokeniseur = TokeniseurBERTweet(longueur_max=longueur_max)

    # ──────────────────────────────────────
    def __call__(self, texte: str) -> Dict[str, torch.Tensor]:
       
        texte_propre = self.nettoyeur.nettoyer(texte)
        return self.tokeniseur.tokeniser(texte_propre)

    def traiter_batch(self, textes: List[str]) -> Dict[str, torch.Tensor]:
        
        textes_propres = self.nettoyeur.nettoyer_batch(textes)
        return self.tokeniseur.tokeniser_batch(textes_propres)

    def nettoyer_seulement(self, texte: str) -> str:
        
        return self.nettoyeur.nettoyer(texte)


# ══════════════════════════════════════════
# 4.  TEST RAPIDE
# ══════════════════════════════════════════

if __name__ == "__main__":
    exemples = [
        "BREAKING: Huge explosion in #Beirut !! @CNN https://t.co/abc123 2750 dead ???",
        "صوت انفجار ضخم في بيروت... #لبنان 💥🙏",
        "RT @RedCross: Please donate to help victims: https://t.co/xyz999 #Lebanon #Crisis",
        "loooool this is so fake 😂😂😂",
        "",   
    ]

    print("=" * 60)
    print("TEST — NettoyeurTweet")
    print("=" * 60)
    nettoyeur = NettoyeurTweet()
    for t in exemples:
        print(f"  AVANT  : {repr(t)}")
        print(f"  APRÈS  : {repr(nettoyeur.nettoyer(t))}")
        print()

    print("=" * 60)
    print("TEST — PipelineTexte (nettoyage + tokenisation)")
    print("=" * 60)
    pipeline = PipelineTexte()
    tweet_test = exemples[0]
    print(f"  Tweet  : {tweet_test}")
    tenseurs = pipeline(tweet_test)
    print(f"  input_ids      shape : {tenseurs['input_ids'].shape}")
    print(f"  attention_mask shape : {tenseurs['attention_mask'].shape}")
    print(f"  Tokens non-PAD       : {tenseurs['attention_mask'].sum().item()}")

    print()
    print("TEST — Batch de tweets")
    batch = pipeline.traiter_batch(exemples[:3])
    print(f"  input_ids      shape : {batch['input_ids'].shape}")
    print(f"  attention_mask shape : {batch['attention_mask'].shape}")
    print()
    print(" traitement_texte.py — Tout OK")