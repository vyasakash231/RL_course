import numpy as np

class MountainCar:
    def __init__(self):
        self.max_position = 0.6 
        self.min_position = -1.2
        self.max_velocity = 0.07
        self.min_velocity = -0.07
        
        self.force = 0.001
        self.gravity = 0.0025

    def reset(self):
        # starting position should be btw -0.6 & -0.4
        # starting velocity is 0
        self.state = np.array([np.random.uniform(-0.6,-0.4,1)[0],0]) 
        return self.state

    def step(self,action):
        position_t0, velocity_t0 = self.state

        velocity_t1 = velocity_t0 + (action-1)*self.force - np.cos(3*position_t0)*self.gravity # calculate velocity

        velocity_t1 = self.scalling(velocity_t1,self.min_velocity,self.max_velocity) # scalling velocity within limits

        position_t1 = position_t0 + velocity_t1  # calculate position

        position_t1 = self.scalling(position_t1,self.min_position,self.max_position) # scalling position within limits

        if position_t1 == self.min_position and velocity_t1 < 0:
            velocity_t1 = 0

        self.state = np.array([position_t1, velocity_t1]) # update state

        reward = -1 # fixed reward

        return self.state, reward
    
    def scalling(self,x,lower,upper):
        if x < lower:
            x = lower
        elif x > upper:
            x = upper
        else:
            pass
        return x