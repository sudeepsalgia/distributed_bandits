import numpy as np
import copy

class PLS:
	def __init__(self, bandit, is_continuous=True, is_homogenous=True, is_stochastic=False):

		self.bandit = bandit     # the underlying bandit instance
		self.is_continuous = is_continuous
		self.is_homogenous = is_homogenous
		self.is_stochastic = is_stochastic
		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0

	def set_parameters(self, C=600):
	
		self.C = C
		self.R = 1/2
		self.tau = 1/(4*np.sqrt(self.bandit.M)) 
		self.B = 1.5
		self.gamma_UCB = 3
		self.alpha_0 = 2
		self.beta_0 = 1

		if not(self.is_homogenous):
			self.g_optimal_actions = [self.bandit.g_optimal_design(A=copy.deepcopy(self.bandit.action_set[m]), n_actions=self.bandit.d+2)[0] for m in range(self.bandit.M)]
		elif not(self.is_continuous):
			self.g_optimal_actions = self.bandit.g_optimal_design(A=copy.deepcopy(self.bandit.action_set), n_actions=self.bandit.d+2)[0] 
		else:
			self.g_optimal_actions = np.eye(self.bandit.d)

	def reset(self):

		self.regret = np.zeros(int(10*self.bandit.T))
		self.uplink_comm_cost = 0
		self.downlink_comm_cost = 0

	def quantize(self, x_vec, r, err):
		x_q = np.zeros(np.size(x_vec))
		n_pts = np.ceil(2*r*np.sqrt(self.bandit.d)/err) + 1
		pts = np.linspace(-r, r, int(n_pts))
		if n_pts%2 == 0:
			cost_add = 0.5
		else:
			cost_add = 0
		n_mid = (n_pts - 1)/2
		cost = 0

		for i in range(self.bandit.d):
			x  = x_vec[i]
			idx = min(int(np.floor((x + r)*n_pts/(2*r))), int(n_pts-1))
			p = x - pts[idx]
			val = idx
			if np.random.random() < p:
				val += 1

			cost += np.abs(val - n_mid) + cost_add
			if val > n_pts:
				val = int(n_pts-2)
			x_q[i] = pts[val]

		cost += self.bandit.d

		return x_q, cost

	def LinUCB(self, theta_hat_init, action_set, time_horizon, max_reward):

		X = 0
		V = np.eye(self.bandit.d)
		V_inv = np.eye(self.bandit.d)
		theta_hat = theta_hat_init
		n_actions = len(action_set)
		means = np.zeros(n_actions)
		std_devs = np.zeros(n_actions)
		regret = np.zeros(time_horizon)
		T_0 = int(3e4)                         
		t_min = min(T_0, time_horizon)     # Allows for approximate and faster implementation
		for t in range(t_min):
			means = action_set @ theta_hat.T
			for n in range(n_actions):
				std_devs[n] = np.sqrt(action_set[n] @ V_inv @ action_set[n].T)

			UCB = means + self.gamma_UCB*std_devs
			action_play = action_set[np.where(UCB == UCB.max())][0]
			true_reward = self.bandit.theta_star @ action_play.T
			X += (true_reward - theta_hat_init @ action_play.T + np.random.normal(scale=self.bandit.noise_sigma))*action_play
			z = V_inv @ action_play.T
			V_inv -= np.outer(z, z)/(1 + action_play @ z.T)
			theta_hat = theta_hat_init + V_inv @ X
			regret[t] = max_reward - true_reward

		if time_horizon > T_0:
			regret[T_0:] = regret[T_0-1]


		return regret


	def explore_agent(self, agent_idx, explore_length, theta_hat_init):

		if self.is_stochastic:
			regret = np.zeros(int(explore_length))
			action_set_idxs = np.floor(np.random.random(explore_length)*self.bandit.N)
			V = 0
			X = 0
			for t in range(explore_length):
				play_action = self.bandit.action_set[int(action_set_idxs[t])][int(np.floor(np.random.random()*self.bandit.K))]
				true_reward = self.bandit.theta_star @ play_action.T
				X += (true_reward + np.random.normal(scale=self.bandit.noise_sigma))*play_action
				V += np.outer(play_action, play_action)
				regret[t] = self.bandit.max_reward[int(action_set_idxs[t])] - true_reward
		else:
			if not(self.is_homogenous):
				action_set_loc = self.g_optimal_actions[agent_idx]
				max_reward_loc = self.bandit.max_reward[agent_idx]
			elif not(self.is_continuous):
				action_set_loc = self.g_optimal_actions 
				max_reward_loc = self.bandit.max_reward
			else:
				action_set_loc = self.g_optimal_actions 
				max_reward_loc = self.bandit.max_reward

			n_actions = len(action_set_loc)
			regret = np.zeros(int(n_actions*explore_length))
			V = 0
			X = 0
			for n in range(n_actions):
				true_reward = self.bandit.theta_star @ action_set_loc[n].T
				X += (true_reward + np.random.normal(scale=self.bandit.noise_sigma)/np.sqrt(explore_length))*action_set_loc[n]
				V += np.outer(action_set_loc[n], action_set_loc[n])
				regret[(n*explore_length):((n+1)*explore_length)] = max_reward_loc - true_reward

		theta_hat = np.linalg.solve(V, X)

		theta_hat_q, cost = self.quantize(theta_hat - theta_hat_init, self.R+self.B, self.alpha_k)

		self.uplink_comm_cost += cost

		return theta_hat_q, regret


	def exploit_agent(self, agent_idx, theta_hat, exploit_length):

		if not(self.is_stochastic):
			if not(self.is_homogenous):
				action_set_loc = copy.deepcopy(self.bandit.action_set[agent_idx])
				max_reward_loc = self.bandit.max_reward[agent_idx]
				means = theta_hat @ action_set_loc.T
				retain_indices = np.where(means >= means.max() - 2*self.tau)
				action_set_loc = action_set_loc[retain_indices]
				regret = self.LinUCB(theta_hat_init=theta_hat, action_set=action_set_loc, time_horizon=exploit_length, max_reward=max_reward_loc)
			elif not(self.is_continuous):
				action_set_loc = copy.deepcopy(self.bandit.action_set)
				max_reward_loc = self.bandit.max_reward
				means = theta_hat @ action_set_loc.T
				retain_indices = np.where(means >= means.max() - 2*self.tau)
				action_set_loc = action_set_loc[retain_indices]
				regret = self.LinUCB(theta_hat_init=theta_hat, action_set=action_set_loc, time_horizon=exploit_length, max_reward=max_reward_loc)
			else:
				theta_hat = theta_hat/np.linalg.norm(theta_hat)
				true_reward = self.bandit.theta_star @ theta_hat.T
				regret = np.ones(exploit_length)*(self.bandit.max_reward - true_reward)
		else:

			X = 0
			V_inv = np.eye(self.bandit.d)
			theta_hat_init = theta_hat
			std_devs = np.zeros(self.bandit.K)
			regret = np.zeros(exploit_length)
			action_set_idxs = np.floor(np.random.random(exploit_length)*self.bandit.N)
			T_0 = int(3e4)
			t_min = min(T_0, exploit_length)            # Allows for approximate and faster implementation
			for t in range(t_min):
				action_set_loc = self.bandit.action_set[int(action_set_idxs[t])]
				means = action_set_loc @ theta_hat.T
				retain_indices = np.where(means >= means.max() - 2*self.tau)
				action_set_loc = action_set_loc[retain_indices]
				means = means[retain_indices]
				for n in range(len(means)):
					std_devs[n] = np.sqrt(action_set_loc[n] @ V_inv @ action_set_loc[n].T)

				UCB = means + self.gamma_UCB*std_devs[:len(means)]
				action_play = action_set_loc[np.where(UCB == UCB.max())][0]
				true_reward = self.bandit.theta_star @ action_play.T

				X += (true_reward - theta_hat_init @ action_play.T + np.random.normal(scale=self.bandit.noise_sigma))*action_play
				z = V_inv @ action_play.T
				V_inv -= np.outer(z, z)/(1 + action_play @ z.T)
				theta_hat = theta_hat_init + V_inv @ X

			if exploit_length > t_min:
				true_rewards_opt_actions = np.zeros(self.bandit.N)
				for n in range(self.bandit.N):
					means = self.bandit.action_set[n] @ theta_hat.T
					optimal_action = self.bandit.action_set[n][np.where(means == means.max())]
					true_rewards_opt_actions[n] = optimal_action @ self.bandit.theta_star.T

				for t in range(t_min, exploit_length):
					regret[t] = self.bandit.max_reward[int(action_set_idxs[t])] - true_rewards_opt_actions[int(action_set_idxs[t])]

		return regret


	def run(self):

		k = 1
		terminate = False
		reg_idx = 0
		mu_0 = 0
		to_refine =  False

		theta_estimate_agent = 0 
		theta_estimate_server = 0

		while not(terminate):
			s_k = int(np.ceil(self.C*(4**k)*(self.bandit.d)*np.log(self.bandit.M*np.log2(self.bandit.T))))
			self.alpha_k = self.alpha_0*self.bandit.noise_sigma*np.sqrt(self.bandit.d*1.0/s_k)
			self.beta_k = self.beta_0*self.tau

			theta_serv_update = 0

			for m in range(self.bandit.M):
				theta_hat_q, reg_loc = self.explore_agent(agent_idx=m, explore_length=s_k, theta_hat_init=theta_estimate_agent)
				self.regret[reg_idx:(reg_idx + len(reg_loc))] += reg_loc
				theta_serv_update += theta_hat_q

			reg_idx += len(reg_loc)
			if reg_idx > self.bandit.T:
				terminate = True
				break

			theta_serv_update /= self.bandit.M
			theta_estimate_server += theta_serv_update
			theta_hat_q, cost = self.quantize(theta_serv_update, self.B + self.tau, self.beta_k)
			theta_estimate_agent += theta_hat_q
			self.downlink_comm_cost += cost

			if (np.linalg.norm(theta_estimate_server) >= 4*self.tau) and not(to_refine):
				to_refine = True
				mu_0 = np.linalg.norm(theta_estimate_server)

			if to_refine:
				t_k = int(self.bandit.M*(s_k**2)*(mu_0**2))
				if self.is_stochastic:
					t_k /= (4*self.C**2)
					t_k = int(t_k)
				for m in range(self.bandit.M):
					reg_loc = self.exploit_agent(agent_idx=m, theta_hat=theta_estimate_agent, exploit_length=t_k)
					if reg_idx + t_k < self.bandit.T:
						self.regret[reg_idx:(reg_idx+t_k)] += reg_loc
					else:
						self.regret[reg_idx:int(self.bandit.T)] += reg_loc[:(int(self.bandit.T) - reg_idx)]
						
				reg_idx += t_k
				if reg_idx > self.bandit.T:
					terminate = True
					break

			if not(terminate):
				k += 1
				self.tau /= 2
				self.R /= 2
				self.B = 5*self.tau
					















