import numpy as np


class LinearBandits:
	def __init__(self, d=20, M=50, K=100, T=5e5, N=100, theta_star=-1, noise_sigma=1):

		self.d = d      # dimension of the linear bandit
		self.M = M      # number of agents
		self.K = K      # number of actions in finite case
		self.T = T      # time horizon
		self.N = N      # number of possible action sets, only valid for stochastic case
		if theta_star == -1:
			theta_star = np.random.normal(size=d)
			self.theta_star = theta_star/np.linalg.norm(theta_star)
		else:
			self.theta_star = theta_star
		self.noise_sigma = noise_sigma
		self.max_reward = 1

	def generate_discrete_action_space(self, n_actions=0):

		if n_actions == 0:
			n_actions = self.K

		action_set = np.random.normal(size=(n_actions, self.d))
		for i in range(n_actions):
			action_set[i, :] = action_set[i,:]/np.linalg.norm(action_set[i, :])

		return action_set

	def generate_discrete_unit_ball_instance(self):

		self.action_set = self.generate_discrete_action_space()

		rewards = self.action_set @ self.theta_star.T
		self.max_reward = rewards.max()

	def generate_heterogeneous_instance(self):

		action_set = [np.zeros(shape=(self.K, self.d)) for _ in range(self.M)]
		self.max_reward = np.zeros(self.M)
		self.rewards = [np.zeros(self.K) for _ in range(self.M)]
		for m in range(self.M):
			action_set[m] = self.generate_discrete_action_space()
			rewards = action_set[m] @ self.theta_star.T
			# print(np.size(rewards))
			self.max_reward[m] = np.max(rewards)
			self.rewards[m] = rewards
			# print(self.max_reward[m])

		self.action_set = action_set
		self.action_set_duplicate = action_set

	def generate_stochastic_instance(self):

		action_set = np.zeros(shape=(self.N, self.K, self.d))
		self.max_reward = np.zeros(self.N)
		for n in range(self.N):
			action_set[n, :, :] = self.generate_discrete_action_space()
			rewards = action_set[n, :, :] @ self.theta_star.T
			self.max_reward[n] = rewards.max()

		self.action_set = action_set


	def g_optimal_design(self, A, n_actions=0):

		if n_actions <= 0:
			n_actions = self.d + 1

		if np.size(A, 0) <= n_actions:
			g_optimal_set = A
			idxs_included = list(range(np.size(A, 0)))
		else:
			g_optimal_set = np.zeros(shape=(n_actions, self.d))
			g_optimal_set[0, :] = A[0, :]
			A_updated = A[1:, :]
			A_orig = A[1:, :]
			idxs = np.arange(start=1, stop=np.size(A, 0))
			idxs_included = [0]
			for n in range(1, n_actions):
				# print(np.shape(A_updated))
				# print(np.shape(g_optimal_set[n-1, :]))
				inner_prods = A_updated @ g_optimal_set[n-1, :].T / np.linalg.norm(g_optimal_set[n-1, :])
				abs_inner_prods = np.abs(inner_prods)
				new_loc = np.where(abs_inner_prods == abs_inner_prods.min())
				for i in range(np.size(A_updated, 0)):
					A_updated[i] -= inner_prods[i] * g_optimal_set[n-1, :]
				g_optimal_set[n, :] = A_orig[new_loc, :]
				A_orig = np.delete(A_orig, new_loc, axis=0)
				A_updated = np.delete(A_updated, new_loc, axis=0)
				# 
				# print(np.shape(A_updated))
				# print(np.shape(g_optimal_set[n-1, :]))
				idxs_included.append(idxs[new_loc])
				idxs = np.delete(idxs, new_loc)

		return g_optimal_set, idxs_included





