import numpy as np
import gymnasium as gym
from gymnasium import spaces

import copy


class CAIEnv(gym.Env):
    def __init__(self, n, L, v_max, m, delta):
        super(CAIEnv, self).__init__()

        self.n = n  # The number of agents
        self.L = L  # The length of route
        self.v_max = v_max  # The maximum of agents' velocity
        self.m = m  # The number of adversaries
        self.delta = delta  # Time penalty per time step
        self.adversaries = self.adversaries_generator()

        # Choose one and comment the other
        # Observation (Discrete)--> The positions (one-hot vector) for each agent
        self.observation_space = spaces.MultiDiscrete([2] * (self.n * (self.L + 1)))
        # Observation (Hybrid)--> The positions (weighted-hot vector) for each agent
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.n * (self.L + 1),), dtype=np.float64)

        # Choose one and comment the other (correspond to the observation)
        # Action (Discrete) --> Velocity: integers in [0, V_max]; Guard: [0, m)
        self.action_space = spaces.MultiDiscrete([self.v_max + 1] * n + [self.m] * n)
        # Action (Hybrid) --> Velocity: continuous space [0, V_max]; Guard: discrete space [0, m)
        # self.action_space = spaces.Tuple((
        #     spaces.Box(low=0, high=self.v_max, shape=(self.n,), dtype=np.float64),
        #     spaces.MultiDiscrete([self.m + 1] * self.n)  # Might be self.m
        # ))

        # Initial state (also weighted-hot state)
        self.state_c = None
        self.state_onehot = None
        # self.state_norm = None
        self.reset()

    def weighted_hot_encoder(self, nums, length):
        """
        Creates a set of weighted-hot encoding vectors for continuous space
        :param nums:  A list of floating-point numbers that are to be converted into weighted-hot encodings.
        :param length: The length of the weighted-hot encoding vectors.
        :return: A NumPy array containing all the concatenated weighted-hot encoding vectors.
        """
        one_hot_vecs = []

        for num in nums:
            vec = np.zeros(length)

            # Integer
            integer_part = int(num)
            vec[integer_part] = 1 - (num - integer_part)

            # Fractional part
            if integer_part + 1 < length:
                vec[integer_part + 1] = num - integer_part

            one_hot_vecs.append(vec)

        return np.concatenate(one_hot_vecs)

    def weighted_hot_decoder(self, vec, length):
        """
        Decodes a set of weighted-hot encoded vectors back into floating-point numbers.
        :param vec: A NumPy array containing concatenated weighted-hot encoded vectors.
        :param length: The length of each weighted-hot encoding vector within 'vec'.
        :return: A list of decoded floating-point numbers.
        """
        num_encoded = len(vec) // length
        decoded_numbers = []

        for i in range(num_encoded):
            segment = vec[i * length: (i + 1) * length]

            # Index capture
            non_zero_indices = np.where(segment > 0)[0]

            # Compute
            if len(non_zero_indices) == 1:
                decoded_numbers.append(non_zero_indices[0])
            else:
                integer_part = non_zero_indices[0]
                fractional_part = segment[non_zero_indices[1]]
                decoded_number = integer_part + fractional_part
                decoded_numbers.append(decoded_number)

        return decoded_numbers

    def adversaries_generator(self):
        """
        Generate adversaries (positions) on the environment.
        Randomly generation or specific generation.
        An adversary is represented by a list composed of four elements:
        the first element represents the starting point of the adversary,
        the second element represents the endpoint,
        the third element indicates the position of the peak risk,
        and the fourth element represents the peak value.
        :return: Adversaries
        """
        # adversaries = []
        # for _ in range(self.m):
        #     start = np.random.randint(0, self.L // 2)  # Randomly generate start point
        #     end = np.random.randint(self.L // 2 + 1, self.L)  # Randomly generate end point
        #     slope = np.random.uniform(0, 1)  # Randomly generate slope
        #     adversaries.append([start, end, slope])
        adversaries = [[10, 34, 22, 150], [26, 50, 38, 150], [10, 50, 40, 100]]

        return adversaries

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Agents' start points
        initial_states = np.zeros(self.n, )  # All agents start at 0.
        # initial_states = np.array([0, 1, 0])

        self.state_c = initial_states

        self.state_onehot = self.weighted_hot_encoder(initial_states, self.L + 1)
        # self.state_norm = initial_states
        info = {}
        return self.state_c, self.state_onehot.astype(np.float32)

    def step(self, action):
        velocities = action[:self.n]
        guard_status = action[self.n:]

        ini_corrected = np.copy(guard_status)

        self.state_c = self.weighted_hot_decoder(self.state_onehot, self.L + 1)
        # self.state_c = np.array(self.state_c)

        # Check the guard status
        corrected = self.status_correction(self.n, self.m, self.adversaries, self.state_c, ini_corrected)
        state_now = copy.deepcopy(self.state_c)

        # Update the state
        self.state_c = np.array(self.state_c) + np.array(velocities)
        # print(self.state_c)
        self.state_c = np.clip(self.state_c, 0, self.L)
        # self.state_norm = np.array(self.state) / 50

        self.state_onehot = self.weighted_hot_encoder(self.state_c, self.L + 1)

        # Calculate the reward
        reward = - self.calculate_total_cost(velocities, corrected)

        # Add reward for moving towards the goal
        move_dis = self.state_c - state_now
        if sum(move_dis) == 0:
            reward -= 100
        reward += 2 * sum(move_dis)

        # Add time penalty
        base = 0 if np.all(self.state_c >= 10) else self.delta
        reward = reward - base
        # cnt = np.sum(self.state_c < self.L)
        # reward -= cnt * self.delta

        # Check if the episode is done (The episode is done once all agents reach the goal)
        terminated = np.all(self.state_c >= self.L)
        if terminated:
            reward += 100
        # There is no specific info to return
        info = {}

        return self.state_onehot.astype(np.float32), reward / 100, terminated, info

    def discount_factor(self, v):
        """
        Define how the velocity influences the adversary
        :param v: the velocity of the agent
        :return: discount factor
        """
        return 1 + 0.6 * v / self.v_max - 0.6

    def risk_function(self, x, zone):
        """
        Define the risk function for each adversary
        :param x: the position in adversary control area
        :param zone: the adversary
        :return: the risk value for the position put in
        """
        start, end, midpoint, top = zone
        # midpoint = (start + end) / 2
        if x < start or x > end:
            return 0
        elif x <= midpoint:
            if (midpoint - start) == 0:
                return top
            slope1 = top / (midpoint - start)
            return abs(slope1 * (x - start))
        else:
            if (end - midpoint) == 0:
                return top
            slope2 = - top / (end - midpoint)
            return abs(slope2 * (end - x))

    def calculate_risk_values(self, car_position, danger_zones):
        return [self.risk_function(car_position, zone) for zone in danger_zones]

    # def formation_function(self, distance):
    #     if distance <= 1:
    #         return 0.8
    #     elif 1 < distance <= 3:
    #         return 2/3 + distance/15 # 0.8+0.05*distance  # 2/3 + distance/15
    #     else:
    #         return 1

    def calculate_total_cost(self, velocities, guard_status):
        total_cost = 0

        for agent in range(self.n):
            risk_values = self.calculate_risk_values(self.state_c[agent], self.adversaries)

            # Calculate the guard discount for each adversary
            guard_discounts = []
            # discounts = []
            for zone in range(self.m):
                discounts = []
                for car_idx in range(self.n):
                    if guard_status[car_idx] == zone:
                        discounts.append(self.discount_factor(velocities[car_idx]))
                    else:
                        discounts.append(1)
                # discounts = self.calculate_guard_discounts_for_zone(velocities, guard_status, zone)
                guard_discounts.append(np.prod(discounts))

            risk_value_with_discount = np.sum(np.array(risk_values) * np.array(guard_discounts))

            agent_cost = risk_value_with_discount
            total_cost += agent_cost
        return total_cost

    def status_correction(self, n, m, adversaries, positions, s):
        """
        Check the correctness for guard behaviour.
        1. Check whether the guard choice for the agent is out of the numbers of adversaries;
        2. Check whether the agent is located in the adversary which he chooses to guard.
        :param n: the number of agents
        :param m: the number of adversaries
        :param adversaries: the information of adversaries
        :param positions: the position
        :param s: initial guard status for all agents
        :return: guard status for all agents after correcting
        """
        for i, _s in enumerate(s):
            if _s not in range(m):
                s[i] = -1
        for agent in range(n):
            for zone in range(m):
                start, end, mid, top = adversaries[zone]
                if not (start <= positions[agent] <= end):
                    if s[agent] == zone:
                        s[agent] = -1
        return s


# Sample for the environment
# env = CAIEnv(2, 50, 3, 3, 10)
#
# n_steps = 20
# for step in range(n_steps):
#   action = env.action_space.sample()
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   # env.render(mode='console')
#   if done:
#     print("Goal reached!", "reward=", reward)
#     break
#
# env.close()