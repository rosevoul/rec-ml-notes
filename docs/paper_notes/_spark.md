# Spark: Cluster Computing with Working Sets

Link: https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf

Distributed-systems reference.

Core idea:
iterative workloads need memory, not disk.

Typical large-scale workflow:
- repeated passes over same dataset
- feature generation
- iterative model updates

Failure mode:
disk-based systems thrash  
latency dominates  
costs increase without speedup  

Spark advantage:
in-memory caching  
fault tolerance with lineage  

Primary caution:
distribution adds overhead  
local debugging still required  
