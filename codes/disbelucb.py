import numpy as np
import copy

class DisBELUCB:
	def __init__(self, bandit):

		self.bandit = bandit    # The underlying bandit instance
		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0

	def set_parameters(self, transmit='full'):
		self.transmit = transmit
		if transmit == 'full':
			self.transmit_size = 32
		else:
			self.transmit_size = int(np.ceil(np.log2(self.bandit.T)))
		self._lambda = 1*np.log(8*self.bandit.d*self.bandit.T)
		self.beta = 1*np.sqrt(np.log(4*self.bandit.K*self.bandit.M*self.bandit.T)) + np.sqrt(self._lambda)
		z = (self.bandit.M*self.bandit.T/self.bandit.d)
		self.a = np.sqrt(self.bandit.T)*np.power(z, 1/np.log2(z))

	def reset(self):

		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0

	def softmax_policy(self, X, M):
		alpha = np.log(self.bandit.K)
		policy = np.zeros(np.size(X, 1))
		for l in len(policy):
			policy[l] = np.power(X[l] @ M @ X[l].T, alpha)

		policy /= np.sum(policy)

		return policy


	def identify_core_set(self, S, thresh):

		core_set = np.copy(S)
		L = np.size(S, 1)
		expected_g_opt_vars = np.zeros(size=(L, self.bandit.d, self.bandit.d))
		g_optimal_size = self.bandit.d + 2
		A_matrix_core_set = self._lambda*np.eye(self.bandit.d)
		for i in range(L):
			g_optimal_solution, _ = bandit.g_optimal_design(A=copy.deepcopy(core_set[i]), n_actions=g_optimal_size)
			cov_matrix = 0
			for x in g_optimal_solution:
				cov_matrix += np.outer(x, x)
			expected_g_opt_vars[i, :, :] = cov_matrix/n_actions
			A_matrix_core_set += expected_g_opt_vars[i, :, :]/L

		A_matrix_core_set_inv = np.linalg.inv(A_matrix_core_set)
		

		while True:
			max_vars = np.zeros(np.size(core_set))
			for l in range(np.size(core_set)):
				max_val = 0
				X = core_set[l]
				for x in X:
					max_val = max(max_val, x @ A_matrix_core_set_inv @ x.T)
				max_vars[l] = max_val

			if max_vars.max() >= thresh:
				break
			else:
				retain_indices = np.where(max_vars <= thresh/2)
				A_matrix_core_set = self._lambda*np.eye(self.bandit.d) + np.sum(expected_g_opt_vars[retain_indices], axis=0)/L
				A_matrix_core_set_inv = np.linalg.inv(A_matrix_core_set)
				core_set = core_set[retain_indices]
				expected_g_opt_vars = expected_g_opt_vars[retain_indices]

		return core_set

	def disbelucb_agent(self, curr_action_set, T_n):
		g_optimal_list = [self.bandit.g_optimal_design(A=copy.deepcopy(x), n_actions=self.bandit.d+2)[0] for x in curr_action_set]
		action_set_idxs = np.floor(np.random.random(T_n)*self.bandit.N)
		regret = np.zeros(T_n)
		u = 0
		for t in range(T_n):
			curr_g_optimal_design = g_optimal_list[int(action_set_idxs[t])]
			n_actions = len(curr_g_optimal_design)
			play_idx = int(np.floor(np.random.random()*n_actions))
			true_reward = self.bandit.theta_star @ curr_g_optimal_design[play_idx].T
			reward = true_reward + np.random.normal(scale=self.bandit.noise_sigma)
			regret[t] = self.bandit.max_reward[int(action_set_idxs[t])] - true_reward
			u += curr_g_optimal_design[play_idx]*reward

		self.uplink_comm_cost += self.transmit_size*self.bandit.d

		# For computation of lambda_i
		Lambda_agent = 0
		for n in range(self.bandit.N):
			g_opt_set = g_optimal_list[n]
			cov_matrix = 0
			for x in g_opt_set:
				cov_matrix += np.outer(x,x)
			cov_matrix /= len(g_opt_set)
			Lambda_agent += cov_matrix

		Lambda_agent /= self.bandit.N

		return u, regret, Lambda_agent

	def update_action_set(self, action_set, Lambda_inv, theta):
		n_actions = len(action_set)
		means = np.zeros(n_actions)
		std_devs = np.zeros(n_actions)
		for n in range(n_actions):
			std_devs[n] = np.sqrt(action_set[n] @ Lambda_inv @ action_set[n].T)
			means[n] = theta @ action_set[n].T

		UCB = means + self.beta*std_devs
		LCB = means - self.beta*std_devs

		action_set = action_set[np.where(UCB >= LCB.max())]

		return action_set


	def run(self):
		updated_action_set = [list(np.copy(self.bandit.action_set)) for _ in range(self.bandit.M)]
		n = 1
		t = 0
		while t < self.bandit.T:
			if n <= 2:
				T_n = (np.ceil(self.a*np.sqrt(self.bandit.d/self.bandit.M)))
			else:
				T_n = (np.floor(np.sqrt(T_n)*self.a))

			T_n = int(min(self.bandit.T - t, T_n))
			u_sum = 0
			lambda_agents = []
			for m in range(self.bandit.M):
				u_loc, reg_loc, lambda_agent_loc = self.disbelucb_agent(updated_action_set[m], T_n)
				self.regret[t:(t+T_n)] += reg_loc
				u_sum += u_loc
				lambda_agents.append(lambda_agent_loc)

			self.downlink_comm_cost += self.transmit_size*self.bandit.d

			for m in range(self.bandit.M):
				lambda_m = self._lambda*np.eye(self.bandit.d) + lambda_agents[m]*(self.bandit.M*T_n)/2
				theta_m = np.linalg.solve(lambda_m, u_sum)
				action_set_m = updated_action_set[m]  
				for j in range(self.bandit.N):
				 	action_set_m[j] = self.update_action_set(action_set_m[j], np.linalg.inv(lambda_m), theta_m)
				updated_action_set[m] = action_set_m

			t += T_n
			n += 1


			 
			



