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

1) **Recommender Embeddings**
   - **Behavior item embeddings** (Item2Vec / graph) define the item space.
   - **User embeddings** place each user in that space.
   - **Content embeddings** help when user history is thin or items are new.
   **[Behavior embeddings](technical_notebooks/recsys_behavior_embeddings.ipynb)** · **[User embeddings](technical_notebooks/recsys_user_embeddings.ipynb)** · **[Content embeddings](technical_notebooks/recsys_content_embeddings.ipynb)**

2) **Retrieval + Ranking, the Production Way**
   **[Retrieval](technical_notebooks/recsys_retrieval.ipynb)** · **[User embeddings](technical_notebooks/recsys_ranking.ipynb)**

3) **GenAI Enhancements for Recommenders**
   - Pre-retrieval query context.
   [LLM Query Context](technical_notebooks/recsys_genai_pre_retrieval.ipynb)
   - Reranking, query and item rewriting, hybrid scoring, structured outputs, caching, latency and cost controls, fallback logic.
   [LLM Reranker](technical_notebooks/recsys_genai_reranker.ipynb)

4) **Monitoring + Metrics for Recommenders**
   - Feature and target drift, segment health, ranking quality metrics, online monitoring, alerting, and incident triage.
   [Monitoring & Safeguards](technical_notebooks/recsys_monitoring_guardrails.ipynb)



<a id="reference-papers"></a>
## Reference Papers

### GenAI & Multimodal Foundations

- **CLIP: Learning Transferable Visual Models from Natural Language Supervision (2021)**  
[Paper](https://arxiv.org/abs/2103.00020) · [Notes](paper_notes/clip.md)

- **Retrieval-Augmented Generation (RAG, 2020)**  
[Paper](https://arxiv.org/abs/2005.11401) · [Notes](paper_notes/rag.md)

- **Attention Is All You Need (2017)**  
[Paper](https://arxiv.org/abs/1706.03762) · [Notes](paper_notes/attention.md)


### Recommender Systems

- **Sampling-Bias-Corrected Neural Modeling (Two-Tower Retrieval, 2019)**  
  [Paper](https://dl.acm.org/doi/10.1145/3298689.3346996) · [Notes](paper_notes/two-tower.md)

- **Deep Learning Recommendation Model (DLRM, Meta, 2019)**  
  [Paper](https://arxiv.org/abs/1906.00091) · [Notes](paper_notes/dlrm.md)

- **Deep Neural Networks for YouTube Recommendations (2016)**  
[Paper](https://research.google/pubs/pub45530/) · [Notes](paper_notes/youtube-two-stage-retrieval.md)


### ML Foundations

- **XGBoost: A Scalable Tree Boosting System (2016)**  
  [Paper](https://arxiv.org/abs/1603.02754) · [Notes](paper_notes/xgboost.md)

- **A Few Useful Things to Know About Machine Learning (2012)**  
  [Paper](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) · [Notes](paper_notes/useful-things-ml.md)
