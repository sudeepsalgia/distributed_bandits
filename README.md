# distributed_bandits

This is the repository for the algorithm Progressive Learning and Sharing.

We implement the following distributed linear bandit algorithms in this repository:

1. Distributed Elimination for Linear Bandits (DELB) (Wang et al.)        
2. Federated Phased Elimination (Fed-PE) (Huang et al.)    
3. Distributed Batch Elimination Linear Upper Confidence Bound (DisBE-LUCB) (Amani et al.)       
4. Progressive Learning and Sharing (PLS) (This work)          


We compare PLS against the other three algorithms and plot the cumulative regret and evaluate the communication cost incurred by the different algorithms. Since different algorithms are designed for different settings, we perform a pairwise comparison with each algorithm based on the setting for which they are original designed for. 

We briefly describe the experimental setup for different experiments and algorithms. Note that in all the experiments, we consider a distributed linear bandit instance with d = 20, 10 agents which is run for a time horizon of T = 1,000,000 steps. In all the cases, the underlying mean reward vector is drawn uniformly from the surface of a unit ball. The observation noise is zero mean Gaussian with unit variance. We plot the averaged cumulative regret for different algorithms considered over the time horizon of T and report the uplink and downlink communication costs. Recall that the uplink communication cost is defined as the number of bits sent by _one_ agent to the server. The downlink cost is defined as the number of bits broadcast by the server. All the results are reported after averaging over 10 Monte Carlo runs. For a fair comparison, we assume that the real numbers are represented by log(MT) bits, which comes out to 24 bits in our setting as opposed to 32 for regular floats. We, however, for the purpose of implementation transfer the full float representation, which is only in favor of the other algorithms.

1. DELB: Since the underlying setting in DELB is the same as that considered in PLS, we consider two setups here. In the first experiment, we consider a linear bandit instance with unit ball as the action space. In the second experiment, we consider an action space consisting of K = 120 actions drawn from a unit ball at random. The cumulative regret for the first experiment is shown in the figure DELB_PLS_unit_ball.png and that for the second is shown in DELB_PLS_discrete.png. As one can note from the plots, PLS offers a significantly lower cumulative regret as compared to DELB in both the cases. The communication costs represented as (uplink, downlink) pair for two experiments are as follows:
- Unit Ball: DELB = (531.8, 7400.2), PLS = (103.0, 139.6)
- Discrete: DELB = (336.0, 4700), PLS = (112.9, 155.1)    
The significant improvement of PLS in terms of communication also evident from the above results. In particular, the difference in downlink cost is significant due to the linear scaling with the number of agents for DELB.

2. FED-PE: We consider the shared parameter case for FED-PE. For each agent, we choose K = 120 actions randomly from the unit ball, independent of other agents. The phase lengths are set to be growing in powers of 2 as chosen in the simulation setup shown in (Huang et al.). The cumulative regret for this experiment is shown in FEDPE_PLS.png. Once again, PLS outperforms FED-PE despite not being designed for this heterogeneous setting. The communication costs represented as (uplink, downlink) pair are as follows:
- FED-PE = (72960.0, 249900), PLS = (301.1, 402.5)   
Note that the scaling with respect to the number of action significantly deteriorates the uplink cost for FED-PE and the dependence on d^2 worsens the downlink cost. On the other hand, PLS continues to enjoy both small uplink and downlink costs.

3. DisBE-LUCB: We adopt the same experimental setup as considered in Amani et al. Instead of a distribution over 100 instances, we consider a distribution over 50 instances with K = 40 actions in each of them. The cumulative regret for this experiment is shown in DISBELUCB_PLS.png. Once again, PLS outperforms DisBE-LUCB despite not being designed for this stochastic setting. The communication costs represented as (uplink, downlink) pair are as follows:
- DisBE-LUCB = (2000, 2000), PLS = (188.9, 309.5)
PLS continues to have a lower communication cost even for this setting establishing its improved performance over existing studies.

