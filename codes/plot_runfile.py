import numpy as np
from bandit import *
from delb import *
from disbelucb import *
from fedpe import *
from pls import *
import matplotlib.pyplot as plt

T = int(1e6)
bandit_instance = LinearBandits(d=20, M=10, T=T, K=40, N=50)
# bandit_instance.generate_discrete_unit_ball_instance()
bandit_instance.generate_stochastic_instance()
# bandit_instance.generate_heterogeneous_instance()

n_sim = 10
# delb_regrets = np.zeros((n_sim, T))
# delb_comm_cost = np.zeros(2)
# fedpe_regrets = np.zeros((n_sim, T))
# fedpe_comm_cost = np.zeros(2)
disbelucb_regrets = np.zeros((n_sim, T))
disbelucb_comm_cost = np.zeros(2)
pls_regrets = np.zeros((n_sim, T))
pls_comm_cost = np.zeros(2)

### DELB

# delb_obj = DELB(bandit=bandit_instance, is_unit_ball=True)
# delb_obj = DELB(bandit=bandit_instance)

# n = 0
# while n < n_sim:
# 	delb_obj = DELB(bandit=bandit_instance)
# 	delb_obj.set_parameters(transmit='log', C=10)
# 	delb_obj.run()
# 	reg = np.cumsum(delb_obj.regret[:T])
# 	if reg[-1] <= 3.2*T:
# 		delb_regrets[n] = reg
# 		delb_comm_cost[0] += delb_obj.uplink_comm_cost
# 		delb_comm_cost[1] += delb_obj.downlink_comm_cost
# 		n += 1
# 	delb_obj.reset()

# print('DELB done')

## DISBELUCB

disbelucb_obj = DisBELUCB(bandit_instance)


for n in range(n_sim):
	disbelucb_obj.set_parameters(transmit='log')
	disbelucb_obj.run()
	disbelucb_regrets[n] = np.cumsum(disbelucb_obj.regret[:T])
	disbelucb_comm_cost[0] += disbelucb_obj.uplink_comm_cost
	disbelucb_comm_cost[1] += disbelucb_obj.downlink_comm_cost
	disbelucb_obj.reset()

print('DISBE-LUCB done')

## FED-PE

# fedpe_obj = FedPE(bandit_instance)


# for n in range(n_sim):
# 	fedpe_obj.set_parameters(transmit='log', alpha=10)
# 	fedpe_obj.run()
# 	fedpe_regrets[n] = np.cumsum(fedpe_obj.regret[:T])
# 	fedpe_comm_cost[0] += fedpe_obj.uplink_comm_cost
# 	fedpe_comm_cost[1] += fedpe_obj.downlink_comm_cost
# 	fedpe_obj.reset()

# print('FED-PE done')

#### PLS

pls_obj = PLS(bandit=bandit_instance, is_stochastic=True)

for n in range(n_sim):
	pls_obj.set_parameters(C=6)
	pls_obj.run()
	pls_regrets[n] = np.cumsum(pls_obj.regret[:T])
	pls_comm_cost[0] += pls_obj.uplink_comm_cost
	pls_comm_cost[1] += pls_obj.downlink_comm_cost
	pls_obj.reset()


time_axis = np.arange(T)

# mean_delb_regret = np.mean(delb_regrets, axis=0)
# std_delb_regret = np.std(delb_regrets, axis=0) / np.sqrt(delb_regrets.shape[0])
# mean_fedpe_regret = np.mean(fedpe_regrets, axis=0)
# std_fedpe_regret = np.std(fedpe_regrets, axis=0) / np.sqrt(fedpe_regrets.shape[0])
mean_disbelucb_regret = np.mean(disbelucb_regrets, axis=0)
std_disbelucb_regret = np.std(disbelucb_regrets, axis=0) / np.sqrt(disbelucb_regrets.shape[0])
mean_pls_regret = np.mean(pls_regrets, axis=0)
std_pls_regret = np.std(pls_regrets, axis=0) / np.sqrt(pls_regrets.shape[0])

# delb_comm_cost /= n_sim
# fedpe_comm_cost /= n_sim
disbelucb_comm_cost /= n_sim
pls_comm_cost /= n_sim

# print(delb_comm_cost)
# print(fedpe_comm_cost)
print(disbelucb_comm_cost)
print(pls_comm_cost)

fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols=1)

# ax.plot(time_axis, mean_delb_regret, label='DELB')
# ax.fill_between(time_axis, mean_delb_regret - std_delb_regret, mean_delb_regret + std_delb_regret, alpha=0.25)
# print(std_delb_regret[-1])

# ax.plot(time_axis, mean_fedpe_regret, label='FED-PE')
# ax.fill_between(time_axis, mean_fedpe_regret - std_fedpe_regret, mean_fedpe_regret + std_fedpe_regret, alpha=0.25)
# print(std_fedpe_regret[-1])

ax.plot(time_axis, mean_disbelucb_regret, label='DisBE-LUCB')
ax.fill_between(time_axis, mean_disbelucb_regret - std_disbelucb_regret, mean_disbelucb_regret + std_disbelucb_regret, alpha=0.25)
print(std_disbelucb_regret[-1])

ax.plot(time_axis, mean_pls_regret, label='PLS')
ax.fill_between(time_axis, mean_pls_regret - std_pls_regret, mean_pls_regret + std_pls_regret, alpha=0.25)
print(std_pls_regret[-1])

ax.legend()
ax.set_title('Regret vs Time')
plt.tight_layout()
plt.show()