import numpy as np
import copy

class FedPE:
	def __init__(self, bandit):

		self.bandit = bandit     # the underlying bandit instance
		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0
		self.p = 0
		self.phase_length = self.bandit.K + 1
		self.action_set = copy.deepcopy(self.bandit.action_set)
		self.action_idxs = [np.arange(self.bandit.K) for _ in range(self.bandit.M)]
		self.optimal_actions = [np.zeros(self.bandit.d) for _ in range(self.bandit.M)]

	def set_parameters(self, transmit='full', alpha=0, delta=0):
		self.transmit = transmit
		if transmit == 'full':
			self.transmit_size = 32
		else:
			self.transmit_size = int(np.ceil(np.log2(self.bandit.T)))
		if delta == 0:
			self.delta = 0.01
		else:
			self.delta = delta
		if alpha == 0:
			self.alpha = min(np.sqrt(2*np.log2(self.bandit.K*np.log2(self.bandit.T)/self.delta)+ 5*self.bandit.d), np.sqrt(2*np.log2(self.bandit.M*self.bandit.K*np.log2(self.bandit.T)/self.delta)))
		else:
			self.alpha = alpha
		self.LOG_K = np.ceil(np.log2(self.bandit.K))

	def reset(self):

		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0
		self.p = 0
		self.phase_length = self.bandit.K + 1
		self.action_set = copy.deepcopy(self.bandit.action_set)
		self.action_idxs = [np.arange(self.bandit.K) for _ in range(self.bandit.M)]
		self.optimal_actions = [np.zeros(self.bandit.d) for _ in range(self.bandit.M)]


	def update_action_set_agent(self, agent_idx, theta_p, V_p): 
		action_set_loc = self.action_set[agent_idx]
		# if action_set_loc.size == 0:
		# 	print('here')
		n_actions = len(action_set_loc)
		means = np.zeros(n_actions)
		std_devs = np.zeros(n_actions)
		for n in range(n_actions):
			means[n] = theta_p @ action_set_loc[n].T
			std_devs[n] = np.sqrt(action_set_loc[n] @ V_p @ action_set_loc[n].T)  # V_p is already the inverse by definition.

		UCB = means + self.alpha*std_devs
		LCB = means - self.alpha*std_devs

		optimal_action_idx = np.where(means == means.max())
		self.optimal_actions[agent_idx] = action_set_loc[optimal_action_idx]
		retain_actions = np.where(UCB >= LCB[optimal_action_idx])
		self.action_set[agent_idx] = action_set_loc[retain_actions]
		self.action_idxs[agent_idx] = self.action_idxs[agent_idx][retain_actions]


	def collaborative_exploration_agent(self, agent_idx, n_plays):
		action_set_loc = self.action_set[agent_idx]
		n_actions = len(action_set_loc)
		theta_estimates = [np.zeros(self.bandit.d) for _ in range(n_actions)]
		regret = np.zeros(self.phase_length)
		curr_idx = 0
		for n in range(n_actions):
			if n_plays[n] > 0:
				true_reward = self.bandit.theta_star @ action_set_loc[n].T
				theta_estimates[n] = (true_reward + np.random.normal(scale=self.bandit.noise_sigma)/np.sqrt(n_plays[n]))*action_set_loc[n]
				regret[curr_idx:(curr_idx + n_plays[n])] = self.bandit.max_reward[agent_idx] - true_reward
				curr_idx += n_plays[n]
				self.uplink_comm_cost += self.bandit.d*self.transmit_size

		if curr_idx < self.phase_length:
			true_reward = self.bandit.theta_star @ self.optimal_actions[agent_idx].T
			regret[curr_idx:self.phase_length] = self.bandit.max_reward[agent_idx] - true_reward

		return theta_estimates, regret

	def initialize_theta_estimates(self):
		self.theta_estimates = []
		for m in range(self.bandit.M):
			true_rewards = self.action_set[m] @ self.bandit.theta_star.T
			y = true_rewards + np.random.normal(scale=self.bandit.noise_sigma, size=np.size(true_rewards))
			self.theta_estimates.append(np.diag(y) @ self.action_set[m])

	def run(self):

		# Initialization
		self.initialize_theta_estimates()
		V_p_local = [np.zeros((self.bandit.d, self.bandit.d)) for _ in range(self.bandit.K)]
		V_p_global = np.zeros((self.bandit.d, self.bandit.d))
		theta_sum = np.zeros(self.bandit.d)
		for k in range(self.bandit.K):
			for m in range(self.bandit.M):
				V_p_local[k] += np.outer(self.theta_estimates[m][k], self.theta_estimates[m][k])/(np.linalg.norm(self.theta_estimates[m][k])**2)
				theta_sum += self.theta_estimates[m][k]
			V_p_local[k] = np.linalg.pinv(V_p_local[k])
			V_p_global += V_p_local[k]

		V_p_global = np.linalg.pinv(V_p_global)
		theta_p = V_p_global @ theta_sum
		self.p += 1
		reg_idx = 0
		terminate = False

		# Looping
		while not(terminate):
			self.phase_length = int(2**self.p + self.bandit.K)
			active_agents = np.array([])
			for m in range(self.bandit.M):
				self.update_action_set_agent(agent_idx=m, theta_p=theta_p, V_p=V_p_global)
				active_agents = np.unique(np.concatenate((active_agents, self.action_idxs[m])))

			# Solve the multi-client G-optimal design
			agent_idxs = []
			all_action_list = []
			for m in range(self.bandit.M):
				for k in range(len(self.action_idxs[m])):
					agent_idxs.append(m)
					theta_hat = self.theta_estimates[m][self.action_idxs[m][k]]
					all_action_list.append(theta_hat/np.linalg.norm(theta_hat))

			g_opt_soln , g_optimal_idxs = self.bandit.g_optimal_design(A=copy.deepcopy(np.array(all_action_list)), n_actions=int(self.bandit.d*3))

			#### ORIGINAL CODE

			V_p_local = [np.zeros((self.bandit.d, self.bandit.d)) for _ in range(self.bandit.K)]
			V_p_global = np.zeros((self.bandit.d, self.bandit.d))
			theta_sum = np.zeros(self.bandit.d)

			ctr = 0
			play_per_action = int(np.ceil(2**self.p/len(g_optimal_idxs)))
			for m in range(self.bandit.M):
				n_plays = np.zeros(len(self.action_idxs[m]), dtype=int)
				for k in range(len(self.action_idxs[m])):
					if ctr in g_optimal_idxs:
						n_plays[k] = play_per_action
						self.downlink_comm_cost += self.LOG_K
					ctr += 1

				theta_estimates_loc, reg_loc = self.collaborative_exploration_agent(agent_idx=m, n_plays=n_plays)
				self.regret[reg_idx:(reg_idx + self.phase_length)] += reg_loc

				# Update the V_p_matrices and theta_estimates
				for k in range(len(self.action_idxs[m])):
					if n_plays[k] > 0:
						self.theta_estimates[m][self.action_idxs[m][k]] = theta_estimates_loc[k]
						theta_hat = theta_estimates_loc[k]
						V_p_local[self.action_idxs[m][k]] += n_plays[k]*np.outer(theta_hat, theta_hat)/(np.linalg.norm(theta_hat)**2)
						theta_sum += n_plays[k]*theta_estimates_loc[k]


			#### SHORTHAND IMPLEMENTATION

			# X_loc = 0
			# V_loc = 0
			# agent_plays = np.zeros(self.bandit.M, dtype=int)
			# play_per_action = int(np.ceil(2**self.p/len(g_optimal_idxs)))
			# reg_loc = np.zeros(shape=(self.bandit.M, self.phase_length))
			# for i in range(len(g_optimal_idxs)):
			# 	true_reward = self.bandit.theta_star @ g_opt_soln[i].T 
			# 	y = true_reward + np.random.normal(scale=self.bandit.noise_sigma/np.sqrt(play_per_action))
			# 	X_loc += play_per_action*y*g_opt_soln[i]
			# 	V_loc += play_per_action*np.outer(g_opt_soln[i], g_opt_soln[i])/(np.linalg.norm(g_opt_soln[i])**2)
			# 	agent_idx_loc = agent_idxs[int(g_optimal_idxs[i])]
			# 	reg_loc[agent_idx_loc][(agent_plays[agent_idx_loc]*play_per_action):((agent_plays[agent_idx_loc]+1)*play_per_action)] = self.bandit.max_reward[agent_idx_loc] - true_reward
			# 	agent_plays[agent_idx_loc] += 1

			# V_p_global = np.linalg.pinv(V_loc)
			# if np.linalg.matrix_rank(g_opt_soln) < self.bandit.d:
			# 	theta_p = V_p_global @ X_loc
			# else:
			# 	theta_p = np.linalg.solve(V_loc, X_loc)

			# for m in range(self.bandit.M):
			# 	true_reward = self.bandit.theta_star @ self.optimal_actions[m].T
			# 	reg_loc[m][(agent_plays[m]*play_per_action):] = self.bandit.max_reward[m] - true_reward

			# self.regret[reg_idx:(reg_idx + self.phase_length)] = np.sum(reg_loc, axis=0)

			reg_idx += self.phase_length
			if reg_idx > self.bandit.T:
				terminate = True
				break
			else:
				for k in list(active_agents):
					k = int(k)
					V_p_global += V_p_local[k]

				if np.linalg.matrix_rank(g_opt_soln) < self.bandit.d:
					V_p_global = np.linalg.pinv(V_p_global)
					theta_p = V_p_global @ theta_sum
				else:
					theta_p = np.linalg.solve(V_p_global, theta_sum)
					V_p_global = np.linalg.pinv(V_p_global)

				self.p += 1

				self.downlink_comm_cost += (self.bandit.d + self.bandit.d**2)*self.transmit_size 












		

		


			






