# TTCBF: A Sampled-Time Truncated Taylor Control Barrier Function for High-Order Safety Constraints
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/bassamlab/ttcbf/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2408.07644-b31b1b.svg)](https://arxiv.org/abs/2601.15196)

**Abstract**: Control Barrier Functions (CBFs) enforce safety by rendering a prescribed safe set forward invariant, typically at the controller sampling instants in digital implementations. For safety constraints with relative degree higher than one, existing methods either chain multiple class~$\mathcal{K}$ functions whose number grows with the relative degree, or use Taylor expansions without class~$\mathcal{K}$ functions that regulate the barrier decay rate. We introduce a sampled-time Truncated Taylor Control Barrier Function (TTCBF), which combines a Taylor expansion with a single class~$\mathcal{K}$ function, thereby decoupling the number of tuning parameters from the relative degree while retaining barrier-decay regulation.
We also propose an adaptive variant (aTTCBF) that optimizes a barrier-decay gain online without increasing the number of tuning parameters. We compare TTCBF and aTTCBF against ten baseline methods in a nonlinear obstacle-avoidance scenario, and stress-test all adaptive methods under different tightened control bounds, where our aTTCBF is the only compared method that remains feasible throughout.


## Install
- Requirements
  - Python 3.11 (other versions may also work)
  - All required Python packages are listed in `requirements.txt`

- Create a Virtual Environment (Recommended)
  ```bash
  # Create environment
  conda create -n ttcbf python=3.11 -y

  # Activate environment
  conda activate ttcbf

  # Install dependencies
  pip install -r requirements.txt
  ```

## How to Use
- run `main.py`

## Simulation Videos (Coming Soon!)

## Number of Tuning Parameters
The counts reported in the manuscript follow a consistent methodology across all methods for the obstacle-avoidance scenario (relative degree $r=2$). ``Tuning parameters'' are defined as parameters of the safety constraint and its auxiliary mechanisms, including class $\mathcal{K}$ coefficients, adaptation-related auxiliary-dynamics parameters, and associated QP cost weights. Parameters of the goal-oriented CLF (those in~\eqref{eq:qp-stability}) are excluded. Below, each method's count is itemized.

**Recursive-chain methods (counts scale with $r$)**

| Method    | Count | Breakdown                                                                                                                                                                                                                                                                                                                                                                               |
| --------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CT-HOCBF  | 2     | Two class $\mathcal{K}$ coefficients ($k_1,k_2$), one per HOCBF chain level                                                                                                                                                                                                                                                                                                         |
| DT-HOCBF  | 2     | Two class $\mathcal{K}$ coefficients ($\gamma_1,\gamma_2\in(0,1)$), discrete-time analog of CT-HOCBF                                                                                                                                                                                                                                                                                |
| PACBF     | 9     | 2 base class $\mathcal{K}$ coefficients $+$ penalty-function HOCBF/CLF parameters (class $\mathcal{K}$ coefficient and CLF convergence rate for $p_1(t)$) $+$ penalty bounds and target values ($p_1^*$, $p_{1,\max}$, $p_1(0)$) $+$ QP cost weights ($W_1$ on auxiliary input, $P_1$ on auxiliary slack, $Q$ on $p_2$ deviation)                             |
| RACBF     | 11    | 2 base class $\mathcal{K}$ coefficients $+$ relaxation-variable HOCBF (relative degree 2: 2 class $\mathcal{K}$ coefficients) $+$ CLF parameters for relaxation stabilization (convergence rate $\epsilon$, target $r^*$) $+$ robustness margin $r^a>0$ $+$ initial value $r(0)$ $+$ QP cost weights ($P_r$ on auxiliary input/slack, $p_{acc}$ on CLF slack) |
| AVCBF     | 9     | 2 base class $\mathcal{K}$ coefficients for main chain $+$ auxiliary-variable HOCBF chain (2 class $\mathcal{K}$ coefficients for $a_1(t)$) $+$ QP cost weight $W_1$ on auxiliary input $\nu_1$ $+$ convergence target $a_{1,w}$ $+$ strict positivity constant $\epsilon$ $+$ initial auxiliary states                                                         |
| aDT-HOCBF | 11    | 2 base class $\mathcal{K}$ coefficients (now time-varying) $+$ penalty-function HOCBF/CLF parameters for each penalty $\gamma_i$ (relative degree $m-i$) $+$ penalty bounds (in $(0,1)$) $+$ CLF target values $+$ QP cost weights on auxiliary inputs and slacks                                                                                                       |

**Taylor-expansion methods (counts independent of $r$)**

| Method           | Count       | Breakdown                                                                                                                                |
| ---------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| ZOH-TLC          | 1           | Discretization time step$\Delta t$ (no class $\mathcal{K}$ functions)                                                                |
| rTLC             | 1           | Discretization time step$\Delta t$ (no class $\mathcal{K}$ functions; robust residual bound depends only on $\Delta t$)            |
| ET-TLC           | 3           | $\Delta t$ $+$ event-triggering state bounds ($\mathbf{x}_{\text{lower}}$, $\mathbf{x}_{\text{up}}$)                             |
| ET-aTLC          | 4           | Time-scale bounds ($\tau_{\min},\tau_{\max}$) $+$ rollout look-ahead horizon $T_{\text{look}}$ $+$ event-triggering state bounds |
| **TTCBF** (our)  | **1** | One class $\mathcal{K}$ coefficient                                                                                                    |
| **aTTCBF** (our) | **1** | One adaptive-gain penalty weight $w_\eta$                                                                                               |

The key structural difference is that recursive-chain methods require $r$ class $\mathcal{K}$ functions in their base form, and each adaptive mechanism (penalty functions, relaxation variables, auxiliary variables) introduces its own HOCBF chain—complete with class $\mathcal{K}$ coefficients, CLF-like stabilization, target values, and QP weights—adding 7--9 parameters per mechanism. In contrast, Taylor-expansion methods avoid the recursive chain entirely; TTCBF uses exactly one class $\mathcal{K}$ function regardless of $r$, and aTTCBF replaces manual tuning of that single function with one QP penalty weight.

