import numpy as np
import copy

class DELB:
	def __init__(self, bandit, is_unit_ball=False):

		self.bandit = bandit     # the underlying bandit instance
		self.is_unit_ball = is_unit_ball
		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0
		if is_unit_ball:
			self.current_arm_set = np.eye(self.bandit.d)
			self.curr_g_optimal_set = np.eye(self.bandit.d)
		else:
			self.current_arm_set = self.bandit.action_set
			self.curr_g_optimal_set, _ = self.bandit.g_optimal_design(A=copy.deepcopy(self.bandit.action_set), n_actions=2*self.bandit.d)

	def set_parameters(self, transmit='full', C=600):
		self.transmit = transmit
		if transmit == 'full':
			self.transmit_size = 32
		else:
			self.transmit_size = int(np.ceil(np.log2(self.bandit.T*self.bandit.M)))
		self.C = C


	def g_optimal_unit_ball(self, theta_hat=0, l=1):

		if l == 1:
			g_optimal = np.eye(self.bandit.d)
		else:
			theta_hat_norm = np.linalg.norm(theta_hat)
			theta_hat = theta_hat/theta_hat_norm
			thresh = 2**(-l + 2)  # we use updated l
			if theta_hat_norm <= thresh:
				g_optimal = np.eye(self.bandit.d)
			else:
				max_reward = min(1, theta_hat_norm)
				# print(theta_hat_norm)
				B = np.eye(self.bandit.d) - np.outer(theta_hat, theta_hat)
				D, V = np.linalg.eig(B)
				V_new = np.transpose(V[:, D > 1e-10])
				
				ortho_thresh = np.sqrt(1 - (max_reward - thresh)**2)
				A = np.concatenate((ortho_thresh*V_new, -ortho_thresh*V_new))
				g_optimal = A + (max_reward - thresh)*theta_hat

		return g_optimal

	def reset(self):

		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0
		# self.set_parameters(C=C)

	def delb_agent(self, arm_indices, n_plays):

		rewards = []
		total_plays = 0
		for n in n_plays:
			total_plays += n
		regret_loc = np.zeros(int(total_plays))
		# print('function val')
		# print(arm_indices)
		# print(n_plays)
		curr_idx = 0
		for i in range(len(arm_indices)):
			true_reward = self.bandit.theta_star @ self.curr_g_optimal_set[arm_indices[i]].T
			rewards.append(true_reward + np.random.normal(loc=0.0, scale=self.bandit.noise_sigma)/np.sqrt(n_plays[i]))
			regret_loc[curr_idx:(curr_idx + int(n_plays[i]))] = self.bandit.max_reward - true_reward
			curr_idx += int(n_plays[i])
			# self.downlink_comm_cost += self.transmit_size
			self.uplink_comm_cost += self.transmit_size
		# print(curr_idx)
		# if self.transmit == 'log':
		# Add this code

		return rewards, regret_loc


	def run(self):

		theta_hat = np.zeros(self.bandit.d)
		l = 1
		reg_idx = 0
		terminate = False

		while not(terminate):
			# in our case it is always a uniform distribution over the support of G optimal design
			n_arms = np.size(self.curr_g_optimal_set, 0)
			pulls_per_arm = int(np.ceil(self.C*(4**l)*(self.bandit.d**2)*np.log(self.bandit.M*self.bandit.T)/n_arms))
			pulls_per_agent = int(np.ceil(pulls_per_arm*n_arms/self.bandit.M))
			# print(pulls_per_agent)
			curr_arm_idx, curr_agent_idx = 0, 0
			pulls_rem_arm, pulls_rem_agent = pulls_per_arm, pulls_per_agent
			arm_indices, n_plays = [], []
			arm_indices_loc, n_plays_loc = [curr_arm_idx], []
			# print(curr_agent_idx < self.bandit.M)
			while ((curr_arm_idx < n_arms) and (curr_agent_idx < self.bandit.M)):
				if pulls_rem_agent >= pulls_rem_arm:
					n_plays_loc.append(pulls_rem_arm)
					if not(curr_arm_idx in arm_indices_loc):
						arm_indices_loc.append(curr_arm_idx) 
					pulls_rem_agent -= pulls_rem_arm
					curr_arm_idx += 1
					pulls_rem_arm = pulls_per_arm
					if pulls_rem_agent == 0:
						arm_indices.append(arm_indices_loc)
						n_plays.append(n_plays_loc)
						arm_indices_loc, n_plays_loc = [], []
						curr_agent_idx += 1
						pulls_rem_agent = pulls_per_agent
				else:
					n_plays_loc.append(pulls_rem_agent)
					if not(curr_arm_idx in arm_indices_loc):
						arm_indices_loc.append(curr_arm_idx)
					pulls_rem_arm -= pulls_rem_agent
					arm_indices.append(arm_indices_loc)
					n_plays.append(n_plays_loc)
					curr_agent_idx += 1
					arm_indices_loc, n_plays_loc = [curr_arm_idx], []     
					pulls_rem_agent = pulls_per_agent

			if not(n_plays_loc == []):
				arm_indices.append(arm_indices_loc)
				n_plays.append(n_plays_loc)
				arm_indices_loc, n_plays_loc = [curr_arm_idx], []    


			# print(arm_indices)
			# print(arm_indices_loc)
			# print(n_plays_loc)
				# print(arm_indices_loc)
				# print(n_plays_loc)

			rewards = np.zeros(int(n_arms))

			# print(n_plays)

			for m in range(self.bandit.M):
				reward_loc, reg_loc = self.delb_agent(arm_indices[m], n_plays[m])
				self.downlink_comm_cost += len(arm_indices[m])*(np.ceil(np.log2(n_arms)) + np.ceil(np.log2(pulls_per_agent)))
				for r, a, n in zip(reward_loc, arm_indices[m], n_plays[m]):
					rewards[a] += r*n
				# print(n_plays[m])
				# print(np.size(reg_loc))
				self.regret[reg_idx:(reg_idx + len(reg_loc))] += reg_loc

			reg_idx += pulls_per_agent
			if reg_idx > self.bandit.T:
				terminate = True
				break

			# rewards /= pulls_per_arm

			X = 0
			V = 0
			for n in range(n_arms):
				arm = self.curr_g_optimal_set[n, :]
				X += rewards[n]*arm
				V += pulls_per_arm*np.outer(arm, arm)

			theta_hat = np.linalg.solve(V, X)
			theta_hat /= np.linalg.norm(theta_hat)
			self.downlink_comm_cost += self.bandit.d*self.transmit_size
			# print(theta_hat)
			l += 1
			if self.is_unit_ball:
				self.curr_g_optimal_set = self.g_optimal_unit_ball(theta_hat, l)
			else:
				inner_prods = self.current_arm_set @ theta_hat.T
				thresh = 2**(-l + 2)  # note that l is already updated
				self.current_arm_set = self.current_arm_set[np.where(inner_prods >= inner_prods.max() - thresh)]
				self.curr_g_optimal_set, _ = self.bandit.g_optimal_design(A=copy.deepcopy(self.current_arm_set), n_actions=2*self.bandit.d)



