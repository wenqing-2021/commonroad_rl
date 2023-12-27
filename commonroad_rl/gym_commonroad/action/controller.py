from commonroad_rl.gym_commonroad.action.vehicle import *
from commonroad_rl.gym_commonroad.action.planner import PlanTrajectory

MAX_KCONSTV = 22.0


class Controller:
    def __init__(
        self,
        config: dict = None,
        vehicle_parameters: VehicleParameters = None,
        control_frequency: int = 50,
    ) -> None:
        self._control_dt = float(1 / control_frequency)
        self._vehicle_parameters = vehicle_parameters
        self._scale_q = config["scale_q"]
        self._scale_r = config["scale_r"]
        self._max_iter = config["ricatti_iter"]
        self._pid_a_kp = config["pid_a_kp"]
        self._pid_a_kd = config["pid_a_kd"]
        self._pid_v_kp = config["pid_v_kp"]
        self._pid_v_kd = config["pid_v_kd"]
        self._last_p_error = 0
        self._last_v_error = 0
        self.control_output = np.array([0, 0])
        self.c_f = 0.0
        self.c_r = 0.0
        self._lateral_controller = config["lateral_controller"]
        self._stanley_k = config["stanley_k"]

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def match_trajectory(
        self,
        current_state: State = None,
        desired_trajectory: PlanTrajectory = None,
        time_step: float = None,
    ) -> State:
        current_pos = np.array([current_state.position[0], current_state.position[1]])
        current_head = current_state.orientation
        if self._lateral_controller == "stanley":
            current_pos = current_pos + np.array([np.cos(current_head), np.sin(current_head)]) * (
                self._vehicle_parameters.a + self._vehicle_parameters.b
            )
        desired_pos = np.array([desired_trajectory.cart_x, desired_trajectory.cart_y]).transpose()
        delta_pos = current_pos - desired_pos
        dis = np.hypot(delta_pos[:, 0], delta_pos[:, 1])
        min_dis_index = np.argmin(dis)
        closest_pos = desired_pos[min_dis_index]
        pose_vec = np.array([np.cos(current_head), np.sin(current_head)])
        dis_vec = closest_pos - current_pos
        if np.dot(pose_vec, dis_vec) < 0:
            min_dis_index += 1
        if min_dis_index == len(desired_trajectory.cart_x):
            min_dis_index -= 1

        # current_v = current_state.velocity
        # if current_v < desired_trajectory.frenet_ds[-1]:
        #     target_v = np.mean(desired_trajectory.frenet_ds[min_dis_index:])
        # else:
        target_v = desired_trajectory.frenet_ds[-1]
        values = {
            "position": np.array(
                [
                    desired_trajectory.cart_x[min_dis_index],
                    desired_trajectory.cart_y[min_dis_index],
                ]
            ),
            "yaw": desired_trajectory.cart_theta[min_dis_index],
            "kappa": desired_trajectory.cart_kappa[min_dis_index],
            "velocity": target_v,
            "acceleration": desired_trajectory.frenet_dds[min_dis_index],
        }

        return CustomState(**values, time_step=time_step)

    def _solve_ricatti(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        P = Q
        eps = 0.01
        for _ in range(self._max_iter):
            P_next: np.ndarray = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            if abs(P - P_next).max() < eps:
                break
            P = P_next

        return P_next

    def _compute_stiffness(self):
        acc = self.control_output[1]
        g = 9.81  # m/s^2
        m = self._vehicle_parameters.m
        lf = self._vehicle_parameters.a
        lr = self._vehicle_parameters.b
        wheel_base = lf + lr
        hcg = self._vehicle_parameters.h_cg
        f_z_f = m * (g * lr - acc * hcg) / wheel_base
        f_z_r = m * (g * lf + acc * hcg) / wheel_base
        self.c_f = self._vehicle_parameters.tire.p_ky1 * f_z_f
        self.c_r = self._vehicle_parameters.tire.p_ky1 * f_z_r

    def _forward_backward(self, k_3, desired_curvature, v_x):
        b = self._vehicle_parameters.b
        a = self._vehicle_parameters.a
        L = a + b
        m = self._vehicle_parameters.m
        steer_angle_feedback = desired_curvature * (
            L - b * k_3 - m * v_x * v_x / L * (b / (2 * self.c_f) + a / (2 * self.c_r) * k_3 - a / (2 * self.c_r))
        )

        return steer_angle_feedback

    def _lateral_LQR_control(self, desired_state: State = None, current_state: State = None):
        # LQR control
        # step 1. compute e_1, e_1_dot, e_2, e_2_dot
        KCONST_V = 1e-3
        current_pos = current_state.position
        ref_theta = desired_state.yaw
        ref_pos = desired_state.position
        delta_pos = current_pos - ref_pos
        ed = delta_pos[1] * math.cos(ref_theta) - delta_pos[0] * math.sin(ref_theta)
        if current_state.has_value("slip_angle"):
            v_x = current_state.velocity * math.cos(current_state.slip_angle) + KCONST_V
        else:
            v_x = current_state.velocity = KCONST_V
        # v_y = current_state.velocity * math.sin(current_state.slip_angle)
        delta_theta = make_valid_orientation(current_state.orientation - ref_theta)
        ed_dot = v_x * math.sin(delta_theta)
        ephi = delta_theta
        ephi_dot = current_state.yaw_rate - v_x * desired_state.kappa
        # step 2. compute matrix A, B, Q, P
        # NOTE: use discrete time model, X_{t+1} = (E + A * dt) * X_t + (B * dt) * u_t
        E = np.eye(4)
        A = np.zeros((4, 4))
        m = self._vehicle_parameters.m
        lf = self._vehicle_parameters.a
        lr = self._vehicle_parameters.b
        I_z = self._vehicle_parameters.I_z
        A[0, 1] = 1
        A[2, 3] = 1
        A[1, 1] = (self.c_f + self.c_r) / (m * v_x)
        A[1, 2] = -(self.c_f + self.c_r) / m
        A[1, 3] = (self.c_f * lf - self.c_r * lr) / (m * v_x)
        A[3, 1] = (self.c_f * lf - self.c_r * lr) / (I_z * v_x)
        A[3, 2] = -(self.c_f * lf - self.c_r * lr) / I_z
        A[3, 3] = (self.c_f * lf**2 + self.c_r * lr**2) / (I_z * v_x)
        B = np.zeros((4, 1))
        B[1, 0] = -self.c_f / m
        B[3, 0] = -self.c_f * lf / I_z
        A_d = E + A * self._control_dt
        B_d = B * self._control_dt
        Q = self._scale_q * np.ones((4, 4))
        R = self._scale_r * np.ones(1)
        # step 3 solve ricatti equation && compute gain matrix K
        P = self._solve_ricatti(A_d, B_d, Q, R)
        K = np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        X = np.array([ed, ed_dot, ephi, ephi_dot])
        # step 4. compute forwardback controller
        k_3 = K[0, 2]
        steer_angle_feedback = self._forward_backward(k_3=k_3, desired_curvature=desired_state.kappa, v_x=v_x)

        # step 5. compute final steering angle
        steer_angle = -K @ X + steer_angle_feedback
        if np.isnan(steer_angle[0]):
            steer_angle[0] = 0.0
        steer_angle = make_valid_orientation(steer_angle[0])
        return steer_angle

    def _lateral_stanley_control(self, desired_state: State = None, current_state: State = None):
        vehicle_speed = current_state.velocity
        vehicle_yaw = current_state.orientation
        ref_head = desired_state.yaw
        head_phi = make_valid_orientation(ref_head - vehicle_yaw)
        front_axle_vec = [-np.cos(ref_head + np.pi / 2), -np.sin(ref_head + np.pi / 2)]
        current_pos = current_state.position + np.array(np.cos(vehicle_yaw), np.sin(vehicle_yaw)) * (
            self._vehicle_parameters.a + self._vehicle_parameters.b
        )
        diff_pos = current_pos - desired_state.position
        e_y = np.dot(diff_pos, front_axle_vec)
        head_e = math.atan2(self._stanley_k * e_y / (vehicle_speed + 0.1), 1)

        steering_angle = head_e + head_phi
        steering_angle = make_valid_orientation(steering_angle)

        return steering_angle

    def _longitudinal_control(self, desired_state: State = None, current_state: State = None):
        # use pid controller for longitudinal control
        current_position = current_state.position
        desired_position = desired_state.position
        desired_head = desired_state.yaw
        # consider control for position error
        desired_head_vec = np.array([math.cos(desired_head), math.sin(desired_head)])
        delta_pos_vec = current_position - desired_position

        error_lon_pos = np.dot(desired_head_vec, delta_pos_vec)
        v_command = (
            self._pid_v_kp * error_lon_pos + self._pid_v_kd * (error_lon_pos - self._last_p_error) / self._control_dt
        )
        self._last_p_error = error_lon_pos
        # v_command = 0
        # consider control for velocity error
        if current_state.has_value("slip_angle"):
            current_lon_v = current_state.velocity * math.cos(current_state.slip_angle)
        else:
            current_lon_v = current_state.velocity
        desired_lon_v = min(desired_state.velocity, MAX_KCONSTV)
        error_lon_v = desired_lon_v - current_lon_v + v_command
        a_commond = (
            self._pid_a_kp * error_lon_v + self._pid_a_kd * (error_lon_v - self._last_v_error) / self._control_dt
        )
        self._last_v_error = error_lon_v
        # a_commond += desired_state.acceleration
        # print("a_commond is ", a_commond)
        return a_commond

    def compute_control_input(self, current_state, desired_state):
        self._compute_stiffness()
        if self._lateral_controller == "stanley":
            steer_angle = self._lateral_stanley_control(desired_state=desired_state, current_state=current_state)
        elif self._lateral_controller == "lqr":
            steer_angle = self._lateral_LQR_control(desired_state=desired_state, current_state=current_state)
        a_commond = self._longitudinal_control(desired_state=desired_state, current_state=current_state)

        # clamp control input
        MAX_STEER_ANGLE = 0.75
        steer_angle = np.clip(
            steer_angle,
            -MAX_STEER_ANGLE,
            MAX_STEER_ANGLE,
        )
        MAX_ACC = 5.0
        a_commond = np.clip(
            a_commond,
            -MAX_ACC,
            MAX_ACC,
        )

        self.control_output = np.array([steer_angle, a_commond])

        return self.control_output
