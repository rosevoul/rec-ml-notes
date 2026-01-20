# XGBoost: A Scalable Tree Boosting System

Link: https://arxiv.org/abs/1603.02754

Tree-boosting reference.

Key properties:
- explicit regularization
- second-order optimization
- efficient handling of sparsity

Typical workflow:
- start with shallow trees
- low learning rate
- incremental boosting

Failure mode:
overfitting via depth  
high variance despite boosting  

Indicators:
training loss drops quickly  
validation loss diverges  

Mitigation:
reduce depth
increase regularization
early stopping

Primary strength: predictable behavior on tabular data.
