
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from sources.utilitaires.configuration import cfg

logger = logging.getLogger(__name__)


class ProjectionModule(nn.Module):
    def __init__(self, dim_texte=768, dim_image=2048, dim_commune=256, dropout=0.1):
        super().__init__()
        self.proj_texte = nn.Sequential(nn.Linear(dim_texte, dim_commune), nn.LayerNorm(dim_commune), nn.GELU(), nn.Dropout(dropout))
        self.proj_image = nn.Sequential(nn.Linear(dim_image, dim_commune), nn.LayerNorm(dim_commune), nn.GELU(), nn.Dropout(dropout))
    
    def forward(self, t, i):
        return self.proj_texte(t), self.proj_image(i)


class CrossAttention(nn.Module):
    def __init__(self, dim_commune=256, nb_tetes=8, dropout=0.1):
        super().__init__()
        assert dim_commune % nb_tetes == 0
        self.nb_tetes, self.dim_tete = nb_tetes, dim_commune // nb_tetes
        self.scale = self.dim_tete ** -0.5
        self.W_q = nn.Linear(dim_commune, dim_commune)
        self.W_k = nn.Linear(dim_commune, dim_commune)
        self.W_v = nn.Linear(dim_commune, dim_commune)
        self.W_o = nn.Linear(dim_commune, dim_commune)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_commune)
    
    def _attn(self, Q, K, V):
        return self.dropout(F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)) @ V
    
    def forward(self, tp, ip):
        B, D = tp.size(0), tp.size(-1)
        t, i = tp.unsqueeze(1), ip.unsqueeze(1)
        
        Q_t = self.W_q(t).view(B, 1, self.nb_tetes, self.dim_tete).transpose(1, 2)
        K_i = self.W_k(i).view(B, 1, self.nb_tetes, self.dim_tete).transpose(1, 2)
        V_i = self.W_v(i).view(B, 1, self.nb_tetes, self.dim_tete).transpose(1, 2)
        te = self.norm(t + self.W_o(self._attn(Q_t, K_i, V_i).transpose(1, 2).contiguous().view(B, 1, D))).squeeze(1)
        
        Q_i = self.W_q(i).view(B, 1, self.nb_tetes, self.dim_tete).transpose(1, 2)
        K_t = self.W_k(t).view(B, 1, self.nb_tetes, self.dim_tete).transpose(1, 2)
        V_t = self.W_v(t).view(B, 1, self.nb_tetes, self.dim_tete).transpose(1, 2)
        ie = self.norm(i + self.W_o(self._attn(Q_i, K_t, V_t).transpose(1, 2).contiguous().view(B, 1, D))).squeeze(1)
        
        return te, ie


class GateNetwork(nn.Module):
    def __init__(self, dim_commune=256, dim_cache=128, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim_commune*2, dim_cache), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_cache, 2), nn.Softmax(dim=-1))
    
    def forward(self, te, ie):
        p = self.gate(torch.cat([te, ie], dim=-1))
        return p[:, :1] * te + p[:, 1:] * ie, p


class FusionCrossModule(nn.Module):
    def __init__(self, dim_texte=768, dim_image=2048, dim_projection=256, nb_tetes=8, utiliser_gate=True, dim_gate_cache=128, dropout=0.1):
        super().__init__()
        self.projection = ProjectionModule(dim_texte, dim_image, dim_projection, dropout)
        self.cross_attention = CrossAttention(dim_projection, nb_tetes, dropout)
        self.gate = GateNetwork(dim_projection, dim_gate_cache, dropout) if utiliser_gate else None
        self.dim_sortie = cfg.fusion.dim_sortie
        self.proj_finale = nn.Sequential(nn.Linear(dim_projection, self.dim_sortie), nn.LayerNorm(self.dim_sortie), nn.GELU(), nn.Dropout(dropout))
        logger.info(f" FusionCrossModule : {dim_texte}+{dim_image} → {dim_projection} → {self.dim_sortie} ({sum(p.numel() for p in self.parameters()):,} params)")
    
    def forward(self, emb_texte, emb_image, return_poids=False):
        tp, ip = self.projection(emb_texte, emb_image)
        te, ie = self.cross_attention(tp, ip)
        if self.gate:
            f, p = self.gate(te, ie)
        else:
            f, p = (te + ie) / 2, torch.ones(emb_texte.size(0), 2, device=emb_texte.device) * 0.5
        out = self.proj_finale(f)
        return (out, p) if return_poids else out


FusionSimple = FusionCrossModule

def creer_fusion_crossmodale(**kw):
    return FusionCrossModule(**kw)