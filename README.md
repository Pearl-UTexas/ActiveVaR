# ActiveVaR

Code for CoRL'18 paper "Risk-Aware Active Inverse Reinforcement Learning".

- This repo contains the implementation of ActiveVaR and the gridworld simulations 
- Change the `QUERY_TYPE` in `src/active_VaR.cpp` to switch between action query and critique query. 
- To run the test:

```
 mkdir build
 make active_VaR_test
 ./active_VaR_test <random_seed>
```

