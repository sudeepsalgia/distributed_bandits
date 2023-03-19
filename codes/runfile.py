import numpy as np
from bandit import *
from delb import *
from disbelucb import *
from fedpe import *
from pls import *
import matplotlib.pyplot as plt

T = int(1e6)
bandit_instance = LinearBandits(d=20, M=10, T=T, K=120, N=100)
# bandit_instance.generate_discrete_unit_ball_instance()
# bandit_instance.generate_stochastic_instance()
bandit_instance.generate_heterogeneous_instance()
# print(bandit_instance.theta_star)


### DELB

# delb_obj = DELB(bandit=bandit_instance)
# # delb_obj = DELB(bandit=bandit_instance)
# delb_obj.set_parameters(transmit='log', C=10)
# delb_obj.run()
# print(delb_obj.uplink_comm_cost)
# print(delb_obj.downlink_comm_cost)

## DISBELUCB

# disbelucb_obj = DisBELUCB(bandit_instance)
# disbelucb_obj.set_parameters()
# disbelucb_obj.run()

## FED-PE

fedpe_obj = FedPE(bandit_instance)
fedpe_obj.set_parameters(alpha=10)
fedpe_obj.run()
print(fedpe_obj.uplink_comm_cost)
print(fedpe_obj.downlink_comm_cost)

#### PLS

pls_obj = PLS(bandit=bandit_instance, is_continuous=False, is_homogenous=False)
pls_obj.set_parameters(C=0.5)
pls_obj.run()
print(pls_obj.uplink_comm_cost)
print(pls_obj.downlink_comm_cost)


time_axis = np.arange(T)

# regret = pls_obj.LinUCB(theta_hat_init=np.zeros(bandit_instance.d), action_set=bandit_instance.action_set, time_horizon=int(1e4), max_reward=bandit_instance.max_reward)

# plt.plot(time_axis, np.cumsum(delb_obj.regret[:T]), label='DELB')
plt.plot(time_axis, np.cumsum(fedpe_obj.regret[:T]), label='FED-PE')
plt.plot(time_axis, np.cumsum(pls_obj.regret[:T]), label='PLS')
# plt.plot(time_axis[:int(1e4)], np.cumsum(regret), label='LinUCB')

plt.legend()
plt.title('Regret vs Time')
plt.tight_layout()
plt.show()