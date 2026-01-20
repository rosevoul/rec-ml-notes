# Attention Is All You Need

Link: https://arxiv.org/abs/1706.03762

Architecture reference.

Core shift:
remove recurrence  
enable parallelism  

Attention behavior:
- weighted aggregation over sequence
- flexible dependency modeling
- quadratic memory cost

Practical constraint:
context length limited by memory  
scaling increases cost nonlinearly  

Resulting focus:
efficiency improvements
approximate attention
sparse variants

Primary impact: scalable sequence modeling.
