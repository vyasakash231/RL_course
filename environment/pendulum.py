import numpy as np

class PendulumEnv:
    def __init__(self, g=10.0):
        self.max_omega = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.high = np.array([1.0, 1.0, self.max_omega])
        self.observation_space = np.array([-self.high, self.high])

        # discritization step should always be odd numbers
        self.d_theta = 15
        self.d_omega = 15
        self.d_torque = 15
            
    def step(self, u):
        theta_t0, omega_t0 = self.state

        u = self.scalling(u, -self.max_torque, self.max_torque)[0] # To limit Torque value btw upper and lower torque limit, we'll scalling function
        costs = pow(self.angle_normalize(theta_t0),2) + 0.1 * pow(omega_t0,2) + 0.001 * pow(u,2) # To limit angle btw [-pi, pi] we'll use angle_normalize function

        omega_t1 = omega_t0 + (3 * self.g / (2 * self.l) * np.sin(theta_t0) + 3.0 / (self.m * pow(self.l,2)) * u) * self.dt
        omega_t1 = self.scalling(omega_t1, -self.max_omega, self.max_omega) # To limit omega value btw upper and lower omega limit, we'll scalling function
        
        theta_t1 = theta_t0 + omega_t1 * self.dt

        self.state = np.array([theta_t1, omega_t1])

        return np.array([np.cos(theta_t1), np.sin(theta_t1), omega_t1]), -costs # return observations [cos(theta), sin(theta), omega], cost
    
    def reset(self): 
        self.state = np.random.uniform(np.array([3*np.pi/4,-1.0]), np.array([5*np.pi/4,1.0])) # Choosing initial state [theta, omega] randomly
        theta, omega = self.state
        return np.array([np.cos(theta), np.sin(theta), omega]) # return observations [cos(theta), sin(theta), omega]
    
    def scalling(self, x, lower, upper):
        if x < lower:
            x = lower
        elif x > upper:
            x = upper
        else:
            pass
        return x

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
    
    
    """Discritize state-space"""
    def discretize_state(self, observation):
        q = [] 
        high = [1, 1, 8]
        for i,j in zip(high,[self.d_theta, self.d_theta, self.d_omega]):
            q.append(np.linspace(-i,i,j+1))

        disc_state = []
        for s, q_set in zip(observation, q):
            disc_state.append(np.digitize(s, q_set).clip(max=q_set.size - 1)-1) # With the help of np.digitize() method, we can get the indices of the bins to which the each value is belongs to an array
        return disc_state

    """changing discrete-actions to continuous action-space"""
    def continualize_action(self, disc_action):
        limit = 2
        interval_length = 2 / (self.d_torque - 1)
        norm_action = disc_action * interval_length
        cont_action = (norm_action - 1) * limit
        return np.array(cont_action).flatten()