# Efficient BackProp

Link: https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

Optimization-focused reference.

Recurring themes:
- scaling matters
- initialization matters
- learning rate dominates early behavior
- batch size affects optimization, not just throughput

Typical training failure:

inputs on different scales.

Example sequence:
- raw features fed directly
- some dimensions small, others large
- gradients uneven across parameters

Observed behavior:
- slow convergence
- oscillating loss
- sensitivity to learning rate changes

Root cause:
loss surface poorly conditioned  
gradient descent inefficient by construction  

Corrections:
normalize inputs
use sensible initialization
adjust learning rate before changing architecture

Primary failure source: optimization setup.
