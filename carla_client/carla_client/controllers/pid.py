import time

class PID:
    def __init__(self):
        self.setpoint = None
        self.Kp = 0.8
        self.Ki = 0.2
        self.Kd = 0.05  

        self.output_min = 0.0
        self.output_max = 1.0
        self.integral_limit = None

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint
    
    def compute_throttle(self, velocity):
        now = time.time()
        dt = (now - self.prev_time) if self.prev_time else 0.0
        self.prev_time = now

        # Proportional term
        error = self.setpoint - velocity

        # Integral term (with optional anti-windup)
        if dt > 0:
            self.integral += error * dt
            if self.integral_limit: # clamping
                # inner min clamps postive integral to integral_limit
                # outer max clamps negative integral to -integral_limit
                # effectively clamps to [-integral_limit, integral_limit]
                self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # PID formula
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Clamp output to limits
        if self.output_min is not None:
            output = max(output, self.output_min)
        if self.output_max is not None:
            output = min(output, self.output_max)

        return output