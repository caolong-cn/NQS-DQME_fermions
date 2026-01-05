This tutorial is about how to utilize the this repository to compute the steady states governed by DQME via RBM ansatz. It interprets the 'main_st.py' in the 'examples/case' floder. You can run `python main_st.py` in that floder.

You can also solve the steady states by evolving the states to $t=+\infin$ via 'main_evolution.py', which is  more recommended.

If you want to simulate the initial states of case1 or case2, you should replace some documents in 'examples/case' by the corresponding files in 'examples/case/alternative files...'.

## Step 1: Initialization of liouville, RBM, basis, sampler and operators

Be the same as the step 1-3 of evolution

## Step 2: Initialize the solver and optimize the loss function

```python
solver = SS_ll(allcut,rho,liouville,states_torch,operators)

print(rho.bd.device)
solver.hopping(niter=3,T=0.001,step_h=0.65,
               tol=1e-10,maxfun=10000,step_p=1000,step_s=5000)
solver.optimization(meth='BFGS',tol=1e-8,step_p=100,step_s=500)
```
