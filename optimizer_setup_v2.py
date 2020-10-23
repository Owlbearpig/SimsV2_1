from scipy.optimize import OptimizeResult
from scipy.optimize import basinhopping
from modules.identifiers.dict_keys import DictKeys
from modules.utils.constants import *


class Output:
    def __init__(self, f, angles, widths, stripes, process_name, iter_cnt, accepted_ratio):
        self.f = f
        self.angles, self.widths, self.stripes = angles, widths, stripes
        self.process_name = process_name
        self.iter_cnt = iter_cnt
        self.accepted_ratio = accepted_ratio
        self.new_best = False

    def __str__(self):
        l0 = "*****************************************************************\n"
        l1 = str(self.process_name) + "\n"
        f, ar = round(self.f, 4), round(self.accepted_ratio, 2)
        l2 = f"{f} at iteration {self.iter_cnt}. AR: {ar}\n"
        angles = np.round(np.rad2deg(self.angles), 2)
        l3 = f"Angles: {angles} (deg)\n"
        widths = np.round(self.widths * 10 ** 6, 2)
        l4 = f"Widths: {widths} (μm)\n"
        stripes = np.round(self.stripes * 10 ** 6, 2)
        l5 = f"Stripe widths {stripes} (μm)\n"
        l6 = "*****************************************************************\n"
        return l0 + l1 + l2 + l3 + l4 + l5 + l6


class OptimizerSetup(DictKeys):
    def __init__(self, erf_setup_instance, settings, queue):
        # settings
        super().__init__()
        self.settings = settings
        self.erf_setup_instance = erf_setup_instance
        self.queue = queue
        self.process_name = settings[self.process_name_key]

        self.basinhopper = None
        self.erf = self.erf_setup_instance.erf
        self.x0 = self.erf_setup_instance.x0
        self.erf_bounds = self.erf_setup_instance.bounds

        self.iterations = settings[self.iterations_key]
        self.angle_step = settings[self.angle_step_key]
        self.width_step = settings[self.width_step_key] * um
        self.stripe_width_step = settings[self.stripe_width_step_key] * um
        self.temperature = settings[self.temperature_key]
        self.local_min_kwargs = {'tol': settings[self.local_opt_tol_key]}
        self.print_precision = settings[self.print_precision_key]
        self.print_interval = settings[self.print_interval_key]
        self.periodic_restart = settings[self.periodic_restart_key]
        self.disable_callback = settings[self.disable_callback_key]

        # variables
        self.best_f = np.inf
        self.iter_cnt = 0
        self.accepted_cnt = 0

        self.run_on_init()

    def run_on_init(self):
        """
        sets settings for optimization
        :return: None
        """
        self.set_seed()
        self.set_local_minimizer_bounds()
        self.set_print_settings()

    def set_seed(self):
        np.random.seed(seed=self.erf_setup_instance.randomizer_seed)

    def set_local_minimizer_bounds(self):
        """
        sets bounds for minimization, set in Setup

        :return: None
        """
        erf_bounds = self.erf_bounds

        x_angle_min, x_angle_max = np.array(erf_bounds["min_angles"]), np.array(erf_bounds["max_angles"])
        x_width_min, x_width_max = np.array(erf_bounds["min_widths"]), np.array(erf_bounds["max_widths"])
        x_stripe_min, x_stripe_max = np.array(erf_bounds["min_stripes"]), np.array(erf_bounds["max_stripes"])

        min_values = np.concatenate((x_angle_min, x_width_min, x_stripe_min))
        max_values = np.concatenate((x_angle_max, x_width_max, x_stripe_max))

        self.local_min_kwargs["bounds"] = list(zip(min_values, max_values))

    def set_print_settings(self):
        np.set_printoptions(precision=self.print_precision)
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=150)

    def callback(self, x, f, accepted):
        """
        called after each optimization step

        :param x: np.array, x (array containing parameter from current step)
        :param f: float, error function value given x
        :param accepted: bool, 1 if optimization step was accepted, else 0
        :return: None
        """
        angle_x_indices = self.erf_setup_instance.x_slices[0]
        width_x_indices = self.erf_setup_instance.x_slices[1]
        stripe_x_indices = self.erf_setup_instance.x_slices[2]

        angles = self.erf_setup_instance.const_angles.copy().astype(np.float)
        angles[angles == 0] = x[angle_x_indices[0]:angle_x_indices[1]]

        const_widths = self.erf_setup_instance.const_widths
        width_pattern = self.erf_setup_instance.width_pattern

        d = const_widths.astype(np.float)

        relevant_identifiers = width_pattern[np.where(const_widths == 0)]
        unique_relevant_identifiers = np.unique(relevant_identifiers)
        open_spots = d[d == 0]
        for i in range(len(unique_relevant_identifiers)):
            open_spots[relevant_identifiers == unique_relevant_identifiers[i]] = \
                x[width_x_indices[0]:width_x_indices[1]][i]
        d[d == 0] = open_spots

        widths = d

        stripes = x[stripe_x_indices[0]:stripe_x_indices[1]]

        self.iter_cnt += 1

        self.accepted_cnt += int(accepted)

        ar = round(self.accepted_cnt / self.iter_cnt, 2)

        self.queue.put(Output(f, angles, widths, stripes, self.process_name, self.iter_cnt, ar))

        if f < self.best_f and accepted:
            self.best_f = f
            output = Output(f, angles, widths, stripes, self.process_name, self.iter_cnt, ar)
            output.new_best = True
            self.queue.put(output)

    @staticmethod
    def local_min_method(fun, x0, args=(), **unknownoptions):
        """
        for testing
        does nothing -> skips local minimization step

        :param fun: method, optimization function
        :param x0: array, start value
        :param args: tuple, handles additional arguments
        :param unknownoptions: handles the rest
        :return: scipy OptimizeResult
        """
        return OptimizeResult(x=x0, fun=fun(x0, *args), success=1)

    def start_optimization(self):
        """
        main optimization method, settings for the optimization

        :return: scipy OptimizeResult
        """

        stack_err = self.erf
        x0 = self.x0
        iterations = self.iterations
        angle_step, width_step, stripe_step = self.angle_step, self.width_step, self.stripe_width_step

        if not self.disable_callback:
            callback = self.callback
        else:
            callback = None

        local_min_kwargs = self.local_min_kwargs
        temperature = self.temperature
        take_step = CustomStep(angle_step, width_step, stripe_step, self.erf_setup_instance, self)
        bounds_callable = Bounds(self)

        opt_res = basinhopping(stack_err, x0, niter=iterations, stepsize=angle_step,
                               callback=callback, take_step=take_step, T=temperature,
                               accept_test=bounds_callable,
                               minimizer_kwargs=local_min_kwargs, interval=50)

        return opt_res


class CustomStep:
    """
    custom monte-carlo step
    """

    def __init__(self, angle_step, width_step, stripe_step, erf_setup_instance, optimizer_instance, step_size=1):
        self.step_size = step_size
        self.angle_step = angle_step * step_size
        self.width_step = width_step * step_size
        self.stripe_step = stripe_step * step_size
        self.erf_setup_instance = erf_setup_instance
        self.optimizer_instance = optimizer_instance

        slicing = erf_setup_instance.x_slices

        self.angle_slice = slicing[0]
        self.width_slice = slicing[1]
        self.stripe_slice = slicing[2]

    def __call__(self, x):
        angle_s = self.angle_step
        width_s = self.width_step
        stripe_s = self.stripe_step

        x[self.angle_slice[0]:self.angle_slice[1]] += \
            np.random.uniform(-angle_s, angle_s, x[self.angle_slice[0]:self.angle_slice[1]].shape)

        x[self.width_slice[0]:self.width_slice[1]] += \
            np.random.uniform(-width_s, width_s, x[self.width_slice[0]:self.width_slice[1]].shape)

        if self.stripe_slice[1] - self.stripe_slice[0]:
            x[self.stripe_slice[0]:self.stripe_slice[1]] += \
                np.random.uniform(-stripe_s, stripe_s, x[self.stripe_slice[0]:self.stripe_slice[1]].shape)

        if (not self.optimizer_instance.iter_cnt % 10 ** 2) and self.optimizer_instance.periodic_restart:
            x = np.random.random(x.shape)
            self.optimizer_instance.force_accept = True

        return x


class Bounds:
    """
    checks if the new x is within bounds
    """

    def __init__(self, optimizer_instance):
        self.optimizer_instance = optimizer_instance
        erf_bounds = self.optimizer_instance.erf_bounds
        erf_setup = self.optimizer_instance.erf_setup_instance

        self.angle_slice = erf_setup.x_slices[0]
        self.width_slice = erf_setup.x_slices[1]
        self.stripe_slice = erf_setup.x_slices[2]

        self.x_angle_min, self.x_angle_max = np.array(erf_bounds["min_angles"]), np.array(erf_bounds["max_angles"])
        self.x_width_min, self.x_width_max = np.array(erf_bounds["min_widths"]), np.array(erf_bounds["max_widths"])
        self.x_stripe_min, self.x_stripe_max = np.array(erf_bounds["min_stripes"]), np.array(erf_bounds["max_stripes"])

    def __call__(self, **kwargs):
        x = kwargs["x_new"]

        t_min_angle = bool(np.all(x[self.angle_slice[0]:self.angle_slice[1]] >=
                                  self.x_angle_min[self.angle_slice[0]:self.angle_slice[1]]))

        t_max_angle = bool(np.all(x[self.angle_slice[0]:self.angle_slice[1]] <=
                                  self.x_angle_max[self.angle_slice[0]:self.angle_slice[1]]))

        t_min_width = bool(np.all(x[self.width_slice[0]:self.width_slice[1]] >= self.x_width_min))
        t_max_width = bool(np.all(x[self.width_slice[0]:self.width_slice[1]] <= self.x_width_max))

        t_min_stripe = bool(np.all(x[self.stripe_slice[0]:self.stripe_slice[1]] >= self.x_stripe_min))
        t_max_stripe = bool(np.all(x[self.stripe_slice[0]:self.stripe_slice[1]] <= self.x_stripe_max))

        return (t_max_angle and t_min_angle) and (t_min_width and t_max_width) and (t_min_stripe and t_max_stripe)


if __name__ == '__main__':
    from modules.settings.settings import Settings
    from erf_setup_v21 import ErfSetup
    import time
    from modules.identifiers.dict_keys import DictKeys

    settings_path = 'modules/results/saved_results/14-10-2020/18-24-27_OptimizationProcess-1/settings.json'
    settings_module = Settings()
    settings = settings_module.load_settings(settings_path)
    settings[DictKeys().calculation_method_key] = 'Jones'
    erf_setup = ErfSetup(settings)

    erf = erf_setup.erf

    iterations = 500

    np.seterr(all='ignore')
    x0 = np.ones(12) * 10 ** -6
    start_time = time.time()

    func = lambda x: np.sum(x ** 2)

    res = basinhopping(erf, x0, niter=iterations)
    print(res.x)
    total = time.time() - start_time

    # print(res)
    print(round(iterations / total, 2))
