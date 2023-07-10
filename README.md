# Control of Time-Dependent Dynamical Systems using Reinforcement Learning
This repository contains the codes that can reproduce the results claimed in the bachelor thesis titled "Control of Dynamical Systems with Intrinsic Nonadiabatic Time Dependence using Deep Reinforcement Learning".

Structure of the repository:
`single_potential_well`: Chapter 5.1.1 (Single Potential Well)
`double_well`: Chapter 5.1.2 (Double-Well Potential)
`2D_potential`: Chapter 5.1.3 (Two Dimensional Potential)
`qubit`: Chapter 5.2 (Dynamical Control Case Studies in Quantum Systems)

In each folder, the file `environments.py` contains the corresponding [Gym](https://github.com/openai/gym)-style environments of the potentials / qubit dynamics. The file `ode.py` contains the differential equations suited for the Scipy integrator.

The `pg_...`-files are the different RL-agents that are trained using the Policy Gradient algorithm.
`pg_fixed.py` corresponds to an agent operating n a static environment (both the agent and the environment are time-independent). `pg_pseudovar.py` corresponds to a time-agnostic agent operating on the time-dependent environment. `pg_var.py` corresponds to an agent receiving cos(omegat) and/or sin(omegat)  `_pseudovar` is the agent 

To train an agent, run the corresponding `pg_...` file. At the end of the training, the program will draw the training curve and the respective phase diagrams.

# Dependencies

You need to have the Python libraries `numpy`, `scipy`, `matplotlib` and `jax`. We recommend using the following minimal library versions: `numpy`: `1.24.3` `scipy`: `1.10.1`, `matplotlib`: `3.7.1`, `jax`: `0.3.25`.
For Numpy versions older than `1.17`, the random number generator would not work. For Scipy versions older than `1.7`, the `ode` functions will not work. For JAX versions older than `0.2.25` you should change `example_libraries` with `experimental`.

When running a `pg_...` file, make sure that the `env.py` and (for the classical mechanics problems) `odes.py` files from the problem are stored in the same folder.