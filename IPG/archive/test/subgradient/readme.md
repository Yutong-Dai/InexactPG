streamed lined tests for adpative v.s. schimdt 
using subgradient decent method

for each inner proximal subproblem; the maximum allowed iteration is set to 5e5.

test following parameters

adaptive

1. laststep: min(c*||s_{k-1}||, myepsilon)
2. const: min(c, myepsilon)
3. schimdt: min(c/k^3, myepsilon)

t = 1e-12 / 1e-3 / 1
c = 1e-1 / 1e0 / 1e1


lambda_shrink = 0.1 / 0.01
group = 0.1/0.2


3. schimdt: min(c/k^3, myepsilon)


c = 10^10 / 10^5 / 10^9