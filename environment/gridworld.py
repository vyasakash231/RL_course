import numpy as np

# class Gridworld:
#     def __init__(self,tot_row,tot_column):
#         self.world_row = tot_row
#         self.world_column = tot_column
        
#         self.state_matrix = np.zeros((tot_row,tot_column))
#         self.reward = np.zeros((tot_row,tot_column))

#         self.no_action = 4 # a = {up, left, down, right}
#         self.trans_matrix = np.zeros((self.no_action,self.no_action))

#         self.position = [] # initial position matrix
        
#     def State_Matrix(self,data):
#         self.state_matrix = data
    
#     def Transition_Matrix(self,data):
#         self.trans_matrix = data
        
#     def Reward(self,data):
#         self.reward = data
    
#     def reset(self,exploring_starts=False):
#         if exploring_starts == True:
#             while True:
#                 row = np.random.randint(0,self.world_row)
#                 column = np.random.randint(0,self.world_column)
#                 if self.state_matrix[row,column] == 0:
#                     break
#             self.position = [row,column]
#         else:
#             self.position = [self.world_row-1,0] # reset starting position to bottom left 
#         return self.position
    
#     def step(self, action):
#         action_applied = np.random.choice(4,1,p=self.trans_matrix[int(action),:]) # Generate a non-uniform random sample (following p-distribution) from np.arange(4) of size 1

#         if action_applied[0] == 0: # UP-action
#             new_position = [self.position[0]-1, self.position[1]]
#         elif action_applied[0] == 1: # LEFT-action
#             new_position = [self.position[0], self.position[1]-1]
#         elif action_applied[0] == 2: # DOWN-action
#             new_position = [self.position[0]+1, self.position[1]]
#         elif action_applied[0] == 3: # RIGHT-action
#             new_position = [self.position[0], self.position[1]+1]
#         else: 
#             raise ValueError('The action is not included in the action space!')
    
#         # Check if the new_postion is possible or not
#         if new_position[0] >= 0 and new_position[0] < self.world_row:
#             if new_position[1] >=0 and new_position[1] < self.world_column:
#                 if self.state_matrix[new_position[0],new_position[1]] != -1:
#                     self.position = new_position

#         return self.position



class Gridworld:
    def __init__(self,tot_row=3,tot_column=4):
        self.world_row = tot_row
        self.world_column = tot_column
        
        self.state_matrix = np.zeros((tot_row,tot_column))
        self.reward = np.zeros((tot_row,tot_column))

        self.no_action = 4 # a = {up, left, down, right}
        self.action_transition_matrix = np.zeros((self.no_action,self.no_action))

        self.position = [] # initial position matrix

        # we'll assign 0 to all state except(1 for terminal, stairs ans -1 for block)
        self.state_matrix = np.array([[0, 0, 0, 1],
                                      [0,-1, 0, 1],
                                      [0, 0, 0, 0]])

        # we'll assign -0.04 to all state except(+1 for terminal, -1 for stairs)
        self.reward = np.array([[-0.04, -0.04, -0.04,     1],
                                [-0.04, -0.04, -0.04,    -1],
                                [-0.04, -0.04, -0.04, -0.04]])

        # for actions not for states {up, left, down, right}
        self.action_transition_matrix = np.array([[0.8 , 0.1 ,  0  , 0.1],
                                                  [0.1 , 0.8 , 0.1 ,  0 ],
                                                  [  0 , 0.1 , 0.8 , 0.1],
                                                  [0.1 ,  0  , 0.1 , 0.8]])
    
    def reset(self,exploring_starts=False):
        if exploring_starts == True:
            while True:
                row = np.random.randint(0,self.world_row)
                column = np.random.randint(0,self.world_column)
                if self.state_matrix[row,column] == 0:
                    break
            self.position = [row,column]
        else:
            self.position = [self.world_row-1,0] # reset starting position to bottom left 
        return self.position
    
    def step(self, action):
        action_applied = np.random.choice(4,1,p=self.action_transition_matrix[int(action),:]) # Generate a non-uniform random sample (following p-distribution) from np.arange(4) of size 1

        if action_applied[0] == 0: # UP-action
            new_position = [self.position[0]-1, self.position[1]]
        elif action_applied[0] == 1: # LEFT-action
            new_position = [self.position[0], self.position[1]-1]
        elif action_applied[0] == 2: # DOWN-action
            new_position = [self.position[0]+1, self.position[1]]
        elif action_applied[0] == 3: # RIGHT-action
            new_position = [self.position[0], self.position[1]+1]
        else: 
            raise ValueError('The action is not included in the action space!')
    
        # Check if the new_postion is possible or not
        if new_position[0] >= 0 and new_position[0] < self.world_row:
            if new_position[1] >=0 and new_position[1] < self.world_column:
                if self.state_matrix[new_position[0],new_position[1]] != -1:
                    self.position = new_position

        return self.position