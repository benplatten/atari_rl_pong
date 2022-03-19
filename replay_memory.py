import numpy as np

class ReplayMemory(object):
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
                                      
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# memory should store state, actions, rewards, new states, dones at each time step
    # all of these will go into the calculation of the target value and the loss function
    # save dones because the target is different in terminal states
# be able take arbitrary state shapes
# method to uniformly sample memories up to most recent
# should not repeat memories
# use numpy arrays for each memory
