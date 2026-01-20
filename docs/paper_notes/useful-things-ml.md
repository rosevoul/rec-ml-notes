# A Few Useful Things to Know About Machine Learning

Link: https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf

Failure-oriented reference.

Recurring patterns:
- leakage is usually accidental
- metrics encode implicit objectives
- training error is rarely the target
- additional data often dominates algorithmic changes

Common evaluation failure:

dataset processed before split.

Typical sequence:
- full dataset collected
- missing values imputed using global statistics
- features normalized using global mean / variance
- random train / test split applied afterward

At this point:
test set already contaminated  
evaluation no longer independent  
reported performance inflated  

Downstream effects:
complex models look better than they are  
model selection drifts toward overfitting  
offline gains fail to reproduce  

Root cause:
preprocessing treated as “data prep” instead of part of the model  
pipeline boundaries unclear  

Correction:
split first  
fit preprocessing only on training data  
reuse frozen transforms for validation and test  
evaluate the full pipeline, not just the estimator  

Primary source of error: evaluation design.
