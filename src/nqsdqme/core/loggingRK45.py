import numpy as np
from scipy.integrate._ivp.rk import RK45, rk_step, MAX_FACTOR, MIN_FACTOR, SAFETY, norm



class LoggingRK45(RK45):
    def __init__(self, fun, t0, y0, t_bound, **kwargs):
        super().__init__(fun, t0, y0, t_bound, **kwargs)
        self.step_count = 0

    
    def _step_impl(self):
        message = {}
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs
        # message['max_step'] = max_step
        # message['min_step'] = min_step
        message['initial_step'] = h_abs
        count = 0
        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)
            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)
            count += 1
            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)
                message['step_size'] = h_abs
                h_abs *= factor

                step_accepted = True
                
                message['scale'] = norm(scale)
                message['error'] = error_norm
                message['e_45'] = error_norm*norm(scale)/message['step_size']
                message['count_iteration'] = count
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True
                

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, message
