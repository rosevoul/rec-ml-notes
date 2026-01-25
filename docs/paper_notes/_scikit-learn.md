# Scikit-learn: Machine Learning in Python

Link: https://arxiv.org/abs/1309.0238

Tooling reference.

Design principles:
- consistent APIs
- explicit data flow
- composable components

Key construct:
pipelines.

Typical failure avoided:
leakage via preprocessing  
manual feature handling  

Pipeline behavior:
- preprocessing fit on training only
- transformations reused consistently
- evaluation reflects deployment path

Primary role:
baseline construction  
evaluation discipline  
