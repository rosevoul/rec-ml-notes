---
layout: default
title: Home
---

## About

I work on machine learning systems where modeling choices shape user experience, business outcomes, and long-term system behavior.

My focus is on:
- Translating abstract models into production-ready systems  
- Choosing the appropriate level of complexity for the problem  
- Making trade-offs, uncertainty, and failure modes explicit  

I’ve shipped end-to-end ML systems, from data pipelines and modeling to deployment, monitoring, and iteration, with direct ownership of production metrics. In recent work, I owned multiple production ML systems that drove measurable lift in engagement and tens of millions in incremental revenue.


<a id="selected-technical-notes"></a>
## Selected Technical Notes

- **[Representation Learning in Recommendation](./representation/)**  
  → [Notebook](https://github.com/rosevoul/rec-ml-notes/edit/main/docs/representation/representation_notes.ipynb)

- **[Graph Neural Networks for Recommendation](./gnn/)**  
  → [Notebook](https://github.com/rosevoul/rec-ml-notes/edit/main/docs/gnn/gnn_notes.ipynb)

- **[Reasoning-Driven Re-ranking with LLMs](./llm_reasoning/)**  
  → [Notebook](https://github.com/rosevoul/rec-ml-notes/edit/main/docs/llm_reasoning/llm_rerank_notes.ipynb)

- **[Multimodal Modeling Beyond Accuracy](./multimodal/)**  
  → [Notebook](https://github.com/rosevoul/rec-ml-notes/edit/main/docs/multimodal/multimodal_fusion_notes.ipynb)

- **[Generative Recommendation Systems](./generative_rec/)**  
  → [Notebook](https://github.com/rosevoul/rec-ml-notes/edit/main/docs/generative_rec/generative_rec_notes.ipynb)


<a id="metrics-showcases"></a>
## Metrics Showcases
- [Metrics Showcase: Churn-Driven Discount Allocation](notebooks/churn_discount_policy_decision.ipynb)  
  Using churn risk to decide which customers receive discounts. Focuses on decision policies, cost trade-offs, and expected value rather than model accuracy.

- [Metrics Showcase: Recommenders in Production (Retail)](notebooks/metrics_showcase_recommenders_production_retail.ipynb)  
  Choosing a Top-N ranking policy using business, user, and system metrics (CTR, EPM, diversity, freshness, latency) with explicit guardrails.

- [Metrics Showcase: Monitoring a Ranking Model in Production](notebooks/metrics_showcase_monitoring_ranking_model_production.ipynb)  
  Monitoring feature drift (PSI), target drift (CTR/CVR), and performance drift (AUC, calibration/value proxies) with explicit alert rules and drill-down diagnostics.


<a id="reference-papers"></a>
## Reference Papers

### Recommender Systems & Ranking

- **Deep Neural Networks for YouTube Recommendations (2016)**  
  [Paper](https://research.google/pubs/pub45530/) · [Notes](notes/youtube-recsys.md)

- **Sampling-Bias-Corrected Neural Modeling (Two-Tower Retrieval, 2019)**  
  [Paper](https://arxiv.org/abs/1905.13021) · [Notes](notes/two-tower-retrieval.md)

- **Deep Learning Recommendation Model (DLRM, Meta, 2019)**  
  [Paper](https://arxiv.org/abs/1906.00091) · [Notes](notes/dlrm.md)


### GenAI & Multimodal Foundations

- **Attention Is All You Need (2017)**  
  [Paper](https://arxiv.org/abs/1706.03762) · [Notes](notes/transformers.md)

- **CLIP: Learning Transferable Visual Models from Natural Language Supervision (2021)**  
  [Paper](https://arxiv.org/abs/2103.00020) · [Notes](notes/clip.md)

- **Retrieval-Augmented Generation (RAG, 2020)**  
  [Paper](https://arxiv.org/abs/2005.11401) · [Notes](notes/rag.md)


### ML Foundations & Reliability

- **XGBoost: A Scalable Tree Boosting System (2016)**  
  [Paper](https://arxiv.org/abs/1603.02754) · [Notes](notes/xgboost.md)

- **A Few Useful Things to Know About Machine Learning (2012)**  
  [Paper](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) · [Notes](notes/useful-things-ml.md)


