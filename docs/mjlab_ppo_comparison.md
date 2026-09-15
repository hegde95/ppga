# Why MJLab RSL-RL PPO is the current bootstrap

## Evidence

The PPGA PPO baseline converged to reward without solving manipulation. Its
stationary evaluation reached an objective of 26.11, but success stayed at 0%
and maximum cube height stayed near 3.5 cm. The converted RSL-RL policy reached
an objective of 56.41, 98.8% success, and 31.1 cm maximum cube height under the
PPGA evaluator.

## Why results differ

PPGA's PPO began as a compact CleanRL-style trainer for Brax locomotion. MJLab
support required later fixes for terminal observations, action distribution,
observation normalization, time-limit bootstrapping, minibatch construction,
and adaptive KL control. Those fixes remove known incompatibilities, but the
custom trainer still falls into the task's dense reach-only optimum. Before the
first successful grasp, reaching reward is easy to improve while the terminal
success bonus provides no useful gradient.

RSL-RL is MJLab's native training path. Its rollout storage, reset semantics,
normalizer, action standard deviation, network sizes, optimizer schedule, and
task defaults are tested together. This tighter integration produced successful
grasps reliably in the observed run.

Current evidence does not prove a mathematical PPO bug. It shows that the PPGA
implementation and its MJLab hyperparameters are not yet a reliable
from-scratch manipulation trainer. PPGA therefore imports a successful RSL-RL
actor, then applies small DQD/PPO updates around that policy. A later ablation
should match RSL-RL settings component by component before removing this
bootstrap dependency.

## Recorded evaluations

- PPGA PPO: `experiments/ppo_mjlab_stationary_lift_cube/42/evaluation.json`
- Converted RSL-RL: `experiments/mjlab_converted_eval/42/evaluation.json`
