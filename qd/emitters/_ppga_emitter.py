"""PPGA Emitter"""
from typing import List, Optional

import numpy as np
import torch
import wandb
from ribs._utils import validate_batch
from ribs.archives import ArchiveBase
from ribs.emitters import EmitterBase
from ribs.emitters.opt import AdamOpt, GradientAscentOpt
from ribs.emitters.rankers import _get_ranker

from qd.emitters.opt import XNES
from RL.ppo import PPO
from utils.utilities import log


class PPGAEmitter(EmitterBase):

    def __init__(self,
                 ppo: PPO,
                 archive: ArchiveBase,
                 *,
                 x0: np.ndarray,
                 sigma0: float,
                 batch_size: int = 100,
                 ranker: str = "2imp",
                 restart_rule: str = 'no_improvement',
                 normalize_grad: bool = True,
                 epsilon: float = 1e-8,
                 seed: Optional[int] = None,
                 use_wandb: bool = False,
                 bounds: Optional[List[float]] = None,
                 grad_opt: str = 'ppo',
                 step_size: Optional[float] = None,
                 normalize_obs: bool = True,
                 normalize_returns: bool = True):
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=len(x0),
            bounds=bounds,
        )

        self._seed_sequence = (seed if isinstance(seed, np.random.SeedSequence)
                               else np.random.SeedSequence(seed))
        ranker_seed, = self._seed_sequence.spawn(1)

        self._epsilon = epsilon
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        self._normalize_grads = normalize_grad
        self._jacobian_batch = None
        self._ranker = _get_ranker(ranker, ranker_seed)
        self._ranker.reset(self, archive)
        self.ppo = ppo

        self._grad_opt = None
        if grad_opt == 'adam':
            self._grad_opt = AdamOpt(self._x0, step_size)
        elif grad_opt == 'gradient_ascent':
            self._grad_opt = GradientAscentOpt(self._x0, step_size)
        elif grad_opt == 'ppo':
            self.ppo.theta = self._x0
            self._grad_opt = self.ppo
        else:
            raise ValueError(f"Invalid Gradient Ascent Optimizer {grad_opt}")

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0
        # Check if the restart_rule is valid.
        _ = self._check_restart(0)
        self._restart_rule = restart_rule

        # We have a coefficient for each measure and an extra coefficient for
        # the objective.
        self._num_coefficients = archive.measure_dim + 1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._initial_bounds = ([-2.0] * (archive.measure_dim + 1),
                                [2.0] * (archive.measure_dim + 1))
        self._initial_bounds[0][
            0] = 0.0  # restrict on-restart sampling of grad f coefficients to be [0.0, 2.0]
        self.opt = XNES(solution_dim=self._num_coefficients,
                        device=self.device,
                        sigma0=sigma0,
                        batch_size=batch_size,
                        seed=seed,
                        initial_bounds=self._initial_bounds)

        self._batch_size = self.opt.batch_size
        self._restarts = 0
        self._itrs = 0
        self._step_size = step_size
        self._use_wandb = use_wandb
        self._normalize_obs = normalize_obs
        self._normalize_returns = normalize_returns
        self._mean_agent_obs_normalizer = None
        self._mean_agent_return_normalizer = None

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def batch_size_dqd(self):
        """int: Number of solutions to return in :meth:`ask_dqd`.

        This is always 1, as we only return the solution point in
        :meth:`ask_dqd`.
        """
        return 1

    @property
    def restarts(self):
        """int: The number of restarts for this emitter."""
        return self._restarts

    @property
    def itrs(self):
        """int: The number of iterations for this emitter."""
        return self._itrs

    @property
    def epsilon(self):
        """int: The epsilon added for numerical stability when normalizing
        gradients in :meth:`tell_dqd`."""
        return self._epsilon

    @property
    def theta(self):
        return self._grad_opt.theta

    @property
    def mean_agent_obs_normalizer(self):
        return self._mean_agent_obs_normalizer

    @mean_agent_obs_normalizer.setter
    def mean_agent_obs_normalizer(self, obs_normalizer):
        self._mean_agent_obs_normalizer = obs_normalizer

    @property
    def mean_agent_return_normalizer(self):
        return self._mean_agent_return_normalizer

    @mean_agent_return_normalizer.setter
    def mean_agent_return_normalizer(self, rew_normalizer):
        self._mean_agent_return_normalizer = rew_normalizer

    def update_theta(self, new_theta):
        self._grad_opt.theta = new_theta

    def ask_dqd(self):
        """Samples a new solution from the gradient optimizer.

        **Call :meth:`ask_dqd` and :meth:`tell_dqd` (in this order) before
        calling :meth:`ask` and :meth:`tell`.**

        Returns:
            a new solution to evaluate.
        """
        return self._grad_opt.theta[None]

    def ask(self):
        """Samples new solutions from a gradient arborescence parameterized by a
        multivariate Gaussian distribution.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self._opt``.

        This method returns ``batch_size`` solutions, even though one solution
        is returned via ``ask_dqd``.

        Returns:
            (:attr:`batch_size`, :attr:`solution_dim`) array -- a batch of new
            solutions to evaluate.
        Raises:
            RuntimeError: This method was called without first passing gradients
                with calls to ask_dqd() and tell_dqd().
        """
        if self._jacobian_batch is None:
            raise RuntimeError("Please call ask_dqd() and tell_dqd() "
                               "before calling ask().")

        grad_coeffs = self.opt.ask()[:, :, None]
        return (self._grad_opt.theta +
                np.sum(self._jacobian_batch * grad_coeffs, axis=1))

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.
        Raises:
          ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, (int, np.integer)):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell_dqd(self, solution, objective, measures, jacobian, add_info,
                 **fields):
        """Gives the emitter results from evaluating the gradient of the
        solutions.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solutions generated by this emitter's :meth:`ask()` method.
            objective (array-like): 1D array containing the objective function
                value of each solution.
            measures (array-like): (batch_size, measure space dimension) array
                with the measure space coordinates of each solution.
            jacobian (array-like): (batch_size, 1 + measure_dim, solution_dim)
                array consisting of Jacobian matrices of the solutions obtained
                from :meth:`ask_dqd`. Each matrix should consist of the
                objective gradient of the solution followed by the measure
                gradients.
            add_info (dict): Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        """
        data, add_info, jacobian = validate_batch(  # pylint: disable = unused-variable
            self.archive,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
            add_info,
            jacobian,
        )

        if self._normalize_grads:
            norms = (np.linalg.norm(jacobian, axis=2, keepdims=True) +
                     self._epsilon)
            jacobian /= norms
        self._jacobian_batch = jacobian

    def tell(self, solution, objective, measures, add_info, **fields):
        """Gives the emitter results from evaluating solutions.

        The solutions are ranked based on the `rank()` function defined by
        `self._ranker`.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solutions generated by this emitter's :meth:`ask()` method.
            objective (array-like): 1D array containing the objective function
                value of each solution.
            measures (array-like): (batch_size, measure space dimension) array
                with the measure space coordinates of each solution.
            add_info (dict): Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        Raises:
            RuntimeError: This method was called without first passing gradients
                with calls to ask_dqd() and tell_dqd().
        """
        data, add_info = validate_batch(
            self.archive,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
            add_info,
        )

        if self._jacobian_batch is None:
            raise RuntimeError("Please call ask_dqd(), tell_dqd(), and ask() "
                               "before calling tell().")

        # Count number of new solutions.
        new_sols = add_info["status"].astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(self, self.archive, data,
                                                    add_info)

        # Select the number of parents.
        num_parents = self._batch_size

        # Record metrics.
        value_batch_parents = add_info["value"][indices][:num_parents]
        mean_value = np.mean(value_batch_parents)
        max_value = np.max(add_info["value"])
        log.debug(f'{mean_value=}, {max_value=}')
        if self._use_wandb:
            wandb.log({
                'QD/mean_value': mean_value,
                'QD/max_value': max_value,
                'QD/iteration': self._itrs,
                'QD/new_sols': new_sols
            })

        # Increase iteration counter.
        self._itrs += 1

        # Update Evolution Strategy.
        self.opt.tell(add_info["value"])  # XNES

        # Check for reset and maybe reset.
        stop_status = self.opt.check_stop(
            ranking_values) or self._check_restart(new_sols)
        if stop_status:
            new_elite = self.archive.sample_elites(1)
            new_theta, measures, obj = new_elite.solution_batch[
                0], new_elite.measures_batch[0], new_elite.objective_batch[0]
            log.debug(
                f'XNES is restarting with a new solution whose measures are {measures} and objective is {obj}'
            )
            if self._normalize_obs:
                self.mean_agent_obs_normalizer.load_state_dict(
                    data["metadata"][0]['obs_normalizer'])
            if self._normalize_returns:
                self._mean_agent_return_normalizer.load_state_dict(
                    data["metadata"][0]['return_normalizer'])

            self._grad_opt.theta = new_theta
            self.opt = XNES(solution_dim=self._num_coefficients,
                            device=self.device,
                            sigma0=self._sigma0,
                            batch_size=self._batch_size,
                            seed=self._seed_sequence.spawn(1)[0],
                            initial_bounds=self._initial_bounds)

            self._ranker.reset(self, self.archive)
            self._restarts += 1

        return stop_status
