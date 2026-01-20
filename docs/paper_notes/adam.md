# Adam: A Method for Stochastic Optimization

Link: https://arxiv.org/abs/1412.6980

Optimizer reference.

Core mechanics:
- adaptive learning rates
- momentum on first and second moments
- scale invariance across parameters

Common usage:
- default optimizer for deep models
- fast initial convergence

Observed pattern:
training stabilizes quickly  
loss decreases smoothly  

Known issue:
converges to suboptimal minima in some settings  
generalization weaker than SGD in some regimes  

Typical adjustment:
use Adam early
switch optimizer later if needed

Primary value: fast iteration.
