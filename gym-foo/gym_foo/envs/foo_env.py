import gym
from gym import error, spaces, logger
from gym import utils
from gym.utils import seeding
import math
import numpy as np
from scipy.integrate import odeint



class InversePendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50}

    def __init__(self):
        self.g = -9.81
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.r = 0.5  # half the pole's length

        self.force_mag = 0.0
        self.tau = 0.01  # seconds between update steps
        self.timestep = 0.0
        #self.integrator = 'euler'
        self.friction = False
        if self.friction:
            self.mu_c = 0.05
            self.mu_p = 0.05

        # Angle at which the episode fails
        self.theta_fail_rad = 400 * (math.pi/180)
        self.x_fail = 2.5

        high_obs = np.array([self.x_fail,
                         10e20,  # np.finfo(np.float32).max
                         self.theta_fail_rad,
                         10e20],
                        dtype=np.float32)

        high_action = np.array([self.force_mag])

        self.action_space = spaces.Box(-high_action, high_action, dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        X0 = self.state
        self.force = action[0]
        #t = [self.timestep, self.timestep + self.tau]
        t = np.linspace(self.timestep, self.timestep + self.tau, 10001)

        sol = odeint(self.dX, X0, t, rtol=1e-10, atol=1e-10)
        #print(sol)
        x = sol[-1, 0]
        dx = sol[-1, 1]
        phi = sol[-1, 2]
        dphi = sol[-1, 3]

        self.state = (x, dx, phi, dphi)

        done = bool(
            x < -self.x_fail
            or x > self.x_fail
            or phi < -self.theta_fail_rad
            or phi > self.theta_fail_rad
        )

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        self.stp += 1
        self.timestep += self.tau

        return np.array(self.state), reward, done, {}

    def dX(self, X, t):
        force = self.force
        x, dx, phi, dphi = self.state

        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        if self.friction:
            if self.stp == 0:
                self.N_c = 1.0
                self.N = 1.0
            ddphi = (self.g * sin_phi + cos_phi * ((-force - self.mass_pole * self.r * dphi * dphi * (
                    sin_phi + self.mu_c * cos_phi * np.sign(
                self.N_c * dx))) / self.total_mass + self.mu_c * self.g * np.sign(
                self.N_c * dx)) - self.mu_p * dphi / (self.mass_pole * self.r)) / (self.r * (
                    4 / 3 - (self.mass_pole * cos_phi / self.total_mass) * (
                    cos_phi - self.mu_c * np.sign(self.N_c * dx))))
            self.N_c = self.total_mass * self.g - self.mass_pole * self.r * (ddphi * sin_phi + cos_phi * dphi * dphi)
            if np.sign(self.N_c) != self.N:
                ddphi = (self.g * sin_phi + cos_phi * ((-force - self.mass_pole * self.r * dphi * dphi * (
                        sin_phi + self.mu_c * cos_phi * np.sign(
                    self.N_c * dx))) / self.total_mass + self.mu_c * self.g * np.sign(
                    self.N_c * dx)) - self.mu_p * dphi / (self.mass_pole * self.r)) / (self.r * (
                        4 / 3 - (self.mass_pole * cos_phi / self.total_mass) * (
                        cos_phi - self.mu_c * np.sign(self.N_c * dx))))
                self.N_c = self.total_mass * self.g - self.mass_pole * self.r * (
                        ddphi * sin_phi + cos_phi * dphi * dphi)
            ddx = (force + self.mass_pole * self.r * (
                        dphi * dphi * sin_phi - ddphi * cos_phi) - self.mu_c * self.N_c * np.sign(
                self.N_c * dx)) / self.total_mass
            self.N = self.N_c
        else:
            ddphi = (self.g * sin_phi + cos_phi * (
                        (-force - self.mass_pole * self.r * sin_phi * dphi * dphi) / self.total_mass)) \
                    / (self.r * (4 / 3 - self.mass_pole * cos_phi * cos_phi / self.total_mass))
            ddx = (force + self.mass_pole * self.r * (dphi * dphi * sin_phi - ddphi * cos_phi)) / self.total_mass

        return dx, ddx, dphi, ddphi

    def reset(self):
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        #self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = np.array([0.0, 0.0, 1.4, 0.0])
        self.state = tuple(self.state)
        self.stp = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_fail * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.r)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None