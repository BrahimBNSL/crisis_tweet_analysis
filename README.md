# Crisis Tweet Classification

A deep learning system for automatically classifying tweets posted during natural disasters. It combines text and image analysis to identify urgent messages that may require rapid intervention.

**Developed by:** Bensalah Brahim and Rafik Asbai.

## Architecture

**BERTweet + LoRA (Text)** | **Frozen ResNet-50 (Image)** | **8-Head Cross-Attention + Gating Network (Fusion)** | **3 Classes: Urgency, Information, Irrelevant**

Only **2.2 million trainable parameters** out of a 160-million-parameter model.

## Performance

| Metric | Score |
|---|---:|
| Accuracy | 83.69% |
| Macro F1-score | 81.77% |
| Urgency F1-score | 88.7% |
| Urgency Recall | 90.3% |

## Dataset

60,000 tweets collected from CrisisMMD, Multimodal, and CrisisLexT26, covering **33 natural disasters**.
