"""
start_optimization: The start_optimization global optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import math
from numpy import cos, sin
import scipy.optimize
from scipy._lib._util import check_random_state

__all__ = ['basinhopping']


class Storage(object):
    """
    Class used to store the lowest energy structure
    """
    def __init__(self, minres):
        self._add(minres)

    def _add(self, minres):
        self.minres = minres
        self.minres.x = np.copy(minres.x)

    def update(self, minres):
        if minres.fun < self.minres.fun:
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        return self.minres


class BasinHoppingRunner(object):
    """This class implements the core of the start_optimization algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly.  Signature should
        be ``x_new = step_taking(x)``.  Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`.  These tests will be used to judge whether or not to accept
        the step.  The acceptable return values are True, False, or ``"force
        accept"``.  If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``start_optimization`` is trapped in.
    disp : bool, optional
        Display status messages.

    """
    def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.disp = disp

        self.nstep = 0

        # initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.minimization_failures = 0

        # do initial minimization
        minres = minimizer(self.x)
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: start_optimization: local minimization failure")
        self.x = np.copy(minres.x)
        self.energy = minres.fun
        if self.disp:
            print("start_optimization step %d: f %g" % (self.nstep, self.energy))

        # initialize storage class
        self.storage = Storage(minres)

        if hasattr(minres, "nfev"):
            self.res.nfev = minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev = minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev = minres.nhev

    def _monte_carlo_step(self):
        """Do one Monte Carlo iteration

        Randomly displace the coordinates, minimize, and decide whether
        or not to accept the new coordinates.
        """
        # Take a random step.  Make a copy of x because the step_taking
        # algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        # do a local minimization
        minres = self.minimizer(x_after_step)
        x_after_quench = minres.x
        energy_after_quench = minres.fun
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: start_optimization: local minimization failure")

        if hasattr(minres, "nfev"):
            self.res.nfev += minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev += minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev += minres.nhev

        # accept the move based on self.accept_tests. If any test is False,
        # then reject the step.  If any test returns the special string
        # 'force accept', then accept the step regardless.  This can be used
        # to forcefully escape from a local minimum if normal basin hopping
        # steps are not sufficient.
        accept = True
        for test in self.accept_tests:
            testres = test(f_new=energy_after_quench, x_new=x_after_quench,
                           f_old=self.energy, x_old=self.x)
            if testres == 'force accept':
                accept = True
                break
            elif testres is None:
                raise ValueError("accept_tests must return True, False, or "
                                 "'force accept'")
            elif not testres:
                accept = False

        # Report the result of the acceptance test to the take step class.
        # This is for adaptive step taking
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(accept, f_new=energy_after_quench,
                                    x_new=x_after_quench, f_old=self.energy,
                                    x_old=self.x)

        return accept, minres

    def one_cycle(self):
        """Do one cycle of the start_optimization algorithm
        """
        self.nstep += 1
        new_global_min = False

        accept, minres = self._monte_carlo_step()

        if accept:
            self.energy = minres.fun
            self.x = np.copy(minres.x)
            new_global_min = self.storage.update(minres)

        # print some information
        if self.disp:
            self.print_report(minres.fun, accept)
            if new_global_min:
                print("found new global minimum on step %d with function"
                      " value %g" % (self.nstep, self.energy))

        # save some variables as BasinHoppingRunner attributes
        self.xtrial = minres.x
        self.energy_trial = minres.fun
        self.accept = accept

        return new_global_min

    def print_report(self, energy_trial, accept):
        """print a status update"""
        minres = self.storage.get_lowest()
        print("start_optimization step %d: f %g trial_f %g accepted %d "
              " lowest_f %g" % (self.nstep, self.energy, energy_trial,
                                accept, minres.fun))


class AdaptiveStepsize(object):
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """
    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
                 verbose=True):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)

    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            # We're accepting too many steps.  This generally means we're
            # trapped in a basin.  Take bigger steps
            self.takestep.stepsize /= self.factor
        else:
            # We're not accepting enough steps.  Take smaller steps
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print("adaptive stepsize: acceptance rate %f target %f new "
                  "stepsize %g old stepsize %g" % (accept_rate,
                  self.target_accept_rate, self.takestep.stepsize,
                  old_stepsize))

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        "called by start_optimization to report the result of the step"
        if accept:
            self.naccept += 1


class RandomDisplacement(object):
    """
    Add a random displacement of maximum size `stepsize` to each coordinate

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_state : None or `np.random.RandomState` instance, optional
        The random number generator that generates the displacements
    """
    def __init__(self, stepsize=0.5, random_state=None):
        self.stepsize = stepsize
        self.random_state = check_random_state(random_state)

    def __call__(self, x):
        x += self.random_state.uniform(-self.stepsize, self.stepsize,
                                       np.shape(x))
        return x


class MinimizerWrapper(object):
    """
    wrap a minimizer function as a minimizer class
    """
    def __init__(self, minimizer, func=None, **kwargs):
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x0):
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)


class Metropolis(object):
    """
    Metropolis acceptance criterion

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    random_state : None or `np.random.RandomState` object
        Random number generator used for acceptance test
    """
    def __init__(self, T, random_state=None):
        # Avoid ZeroDivisionError since "MBH can be regarded as a special case
        # of the BH framework with the Metropolis criterion, where temperature
        # T = 0."  (Reject all steps that increase energy.)
        self.beta = 1.0 / T if T != 0 else float('inf')
        self.random_state = check_random_state(random_state)

    def accept_reject(self, energy_new, energy_old):
        """
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted,
        less likely for larger differences.
        """
        w = math.exp(min(0, -float(energy_new - energy_old) * self.beta))
        rand = self.random_state.rand()
        return w >= rand

    def __call__(self, **kwargs):
        """
        f_new and f_old are mandatory in kwargs
        """
        return bool(self.accept_reject(kwargs["f_new"],
                    kwargs["f_old"]))


class BasinHopping:

    def __init__(self):
        super().__init__()

    def start_optimization(self, func, x0, niter=100, T=1.0, stepsize=0.5,
                           minimizer_kwargs=None, take_step=None, accept_test=None,
                           callback=None, interval=50, disp=False, niter_success=None,
                           seed=None):
        """
        Find the global minimum of a function using the basin-hopping algorithm

        Basin-hopping is a two-phase method that combines a global stepping
        algorithm with local minimization at each step.  Designed to mimic
        the natural process of energy minimization of clusters of atoms, it works
        well for similar problems with "funnel-like, but rugged" energy landscapes
        [5]_.

        As the step-taking, step acceptance, and minimization methods are all
        customizable, this function can also be used to implement other two-phase
        methods.

        Parameters
        ----------
        func : callable ``f(x, *args)``
            Function to be optimized.  ``args`` can be passed as an optional item
            in the dict ``minimizer_kwargs``
        x0 : array_like
            Initial guess.
        niter : integer, optional
            The number of basin-hopping iterations
        T : float, optional
            The "temperature" parameter for the accept or reject criterion.  Higher
            "temperatures" mean that larger jumps in function value will be
            accepted.  For best results ``T`` should be comparable to the
            separation (in function value) between local minima.
        stepsize : float, optional
            Maximum step size for use in the random displacement.
        minimizer_kwargs : dict, optional
            Extra keyword arguments to be passed to the local minimizer
            ``scipy.optimize.minimize()`` Some important options could be:

                method : str
                    The minimization method (e.g. ``"L-BFGS-B"``)
                args : tuple
                    Extra arguments passed to the objective function (``func``) and
                    its derivatives (Jacobian, Hessian).

        take_step : callable ``take_step(x)``, optional
            Replace the default step-taking routine with this routine.  The default
            step-taking routine is a random displacement of the coordinates, but
            other step-taking algorithms may be better for some systems.
            ``take_step`` can optionally have the attribute ``take_step.stepsize``.
            If this attribute exists, then ``start_optimization`` will adjust
            ``take_step.stepsize`` in order to try to optimize the global minimum
            search.
        accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
            Define a test which will be used to judge whether or not to accept the
            step.  This will be used in addition to the Metropolis test based on
            "temperature" ``T``.  The acceptable return values are True,
            False, or ``"force accept"``. If any of the tests return False
            then the step is rejected. If the latter, then this will override any
            other tests in order to accept the step. This can be used, for example,
            to forcefully escape from a local minimum that ``start_optimization`` is
            trapped in.
        callback : callable, ``callback(x, f, accept)``, optional
            A callback function which will be called for all minima found.  ``x``
            and ``f`` are the coordinates and function value of the trial minimum,
            and ``accept`` is whether or not that minimum was accepted.  This can
            be used, for example, to save the lowest N minima found.  Also,
            ``callback`` can be used to specify a user defined stop criterion by
            optionally returning True to stop the ``start_optimization`` routine.
        interval : integer, optional
            interval for how often to update the ``stepsize``
        disp : bool, optional
            Set to True to print status messages
        niter_success : integer, optional
            Stop the run if the global minimum candidate remains the same for this
            number of iterations.
        seed : int or `np.random.RandomState`, optional
            If `seed` is not specified the `np.RandomState` singleton is used.
            If `seed` is an int, a new `np.random.RandomState` instance is used,
            seeded with seed.
            If `seed` is already a `np.random.RandomState instance`, then that
            `np.random.RandomState` instance is used.
            Specify `seed` for repeatable minimizations. The random numbers
            generated with this seed only affect the default Metropolis
            `accept_test` and the default `take_step`. If you supply your own
            `take_step` and `accept_test`, and these functions use random
            number generation, then those functions are responsible for the state
            of their random number generator.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``fun`` the value
            of the function at the solution, and ``message`` which describes the
            cause of the termination. The ``OptimizeResult`` object returned by the
            selected minimizer at the lowest minimum is also contained within this
            object and can be accessed through the ``lowest_optimization_result``
            attribute.  See `OptimizeResult` for a description of other attributes.

        See Also
        --------
        minimize :
            The local minimization function called once for each start_optimization step.
            ``minimizer_kwargs`` is passed to this routine.

        Notes
        -----
        Basin-hopping is a stochastic algorithm which attempts to find the global
        minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
        [4]_.  The algorithm in its current form was described by David Wales and
        Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.

        The algorithm is iterative with each cycle composed of the following
        features

        1) random perturbation of the coordinates

        2) local minimization

        3) accept or reject the new coordinates based on the minimized function
           value

        The acceptance test used here is the Metropolis criterion of standard Monte
        Carlo algorithms, although there are many other possibilities [3]_.

        This global minimization method has been shown to be extremely efficient
        for a wide variety of problems in physics and chemistry.  It is
        particularly useful when the function has many minima separated by large
        barriers. See the Cambridge Cluster Database
        http://www-wales.ch.cam.ac.uk/CCD.html for databases of molecular systems
        that have been optimized primarily using basin-hopping.  This database
        includes minimization problems exceeding 300 degrees of freedom.

        See the free software program GMIN (http://www-wales.ch.cam.ac.uk/GMIN) for
        a Fortran implementation of basin-hopping.  This implementation has many
        different variations of the procedure described above, including more
        advanced step taking algorithms and alternate acceptance criterion.

        For stochastic global optimization there is no way to determine if the true
        global minimum has actually been found. Instead, as a consistency check,
        the algorithm can be run from a number of different random starting points
        to ensure the lowest minimum found in each example has converged to the
        global minimum.  For this reason ``start_optimization`` will by default simply
        run for the number of iterations ``niter`` and return the lowest minimum
        found.  It is left to the user to ensure that this is in fact the global
        minimum.

        Choosing ``stepsize``:  This is a crucial parameter in ``start_optimization`` and
        depends on the problem being solved.  The step is chosen uniformly in the
        region from x0-stepsize to x0+stepsize, in each dimension.  Ideally it
        should be comparable to the typical separation (in argument values) between
        local minima of the function being optimized.  ``start_optimization`` will, by
        default, adjust ``stepsize`` to find an optimal value, but this may take
        many iterations.  You will get quicker results if you set a sensible
        initial value for ``stepsize``.

        Choosing ``T``: The parameter ``T`` is the "temperature" used in the
        Metropolis criterion.  Basinhopping steps are always accepted if
        ``func(xnew) < func(xold)``.  Otherwise, they are accepted with
        probability::

            exp( -(func(xnew) - func(xold)) / T )

        So, for best results, ``T`` should to be comparable to the typical
        difference (in function values) between local minima.  (The height of
        "walls" between local minima is irrelevant.)

        If ``T`` is 0, the algorithm becomes Monotonic Basin-Hopping, in which all
        steps that increase energy are rejected.

        .. versionadded:: 0.12.0

        References
        ----------
        .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
            Cambridge, UK.
        .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
            the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
            110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.
        .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the
            multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,
            1987, 84, 6611.
        .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,
            crystals, and biomolecules, Science, 1999, 285, 1368.
        .. [5] Olson, B., Hashmi, I., Molloy, K., and Shehu1, A., Basin Hopping as
            a General and Versatile Optimization Framework for the Characterization
            of Biological Macromolecules, Advances in Artificial Intelligence,
            Volume 2012 (2012), Article ID 674832, :doi:`10.1155/2012/674832`
        """
        x0 = np.array(x0)

        # set up the np.random.RandomState generator
        rng = check_random_state(seed)

        # set up minimizer
        if minimizer_kwargs is None:
            minimizer_kwargs = dict()
        wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
                                             **minimizer_kwargs)

        # set up step-taking algorithm
        if take_step is not None:
            if not callable(take_step):
                raise TypeError("take_step must be callable")
            # if take_step.stepsize exists then use AdaptiveStepsize to control
            # take_step.stepsize
            if hasattr(take_step, "stepsize"):
                take_step_wrapped = AdaptiveStepsize(take_step, interval=interval,
                                                     verbose=disp)
            else:
                take_step_wrapped = take_step
        else:
            # use default
            displace = RandomDisplacement(stepsize=stepsize, random_state=rng)
            take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
                                                 verbose=disp)

        # set up accept tests
        accept_tests = []
        if accept_test is not None:
            if not callable(accept_test):
                raise TypeError("accept_test must be callable")
            accept_tests = [accept_test]

        # use default
        metropolis = Metropolis(T, random_state=rng)
        accept_tests.append(metropolis)

        if niter_success is None:
            niter_success = niter + 2

        bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
                                accept_tests, disp=disp)

        # start main iteration loop
        count, i = 0, 0
        message = ["requested number of start_optimization iterations completed"
                   " successfully"]
        for i in range(niter):
            new_global_min = bh.one_cycle()

            if callable(callback):
                # should we pass a copy of x?
                callback(bh.xtrial, bh.energy_trial, bh.accept)

            count += 1
            if new_global_min:
                count = 0
            elif count > niter_success:
                message = ["success condition satisfied"]
                break

        # prepare return object
        res = bh.res
        res.lowest_optimization_result = bh.storage.get_lowest()
        res.x = np.copy(res.lowest_optimization_result.x)
        res.fun = res.lowest_optimization_result.fun
        res.message = message
        res.nit = i + 1
        return res


if __name__ == "__main__":
    pass