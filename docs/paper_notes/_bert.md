# BERT: Pre-training of Deep Bidirectional Transformers

Link: https://arxiv.org/abs/1810.04805

Pretraining reference.

Core workflow:
- large-scale pretraining
- task-specific fine-tuning

Pretraining effect:
strong representations  
reduced task-specific data needs  

Fine-tuning risk:
overfitting on small datasets  
instability with high learning rates  

Common mitigation:
freeze lower layers
use smaller learning rates
early stopping

Primary contribution: normalize transfer learning.
