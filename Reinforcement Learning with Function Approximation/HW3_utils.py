import numpy as np
import copy
import lqg1d
from scipy import stats
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('text', usetex = True)
mpl.rc('font', family = 'serif')
mpl.rc('font', size = 15)

######################################################################
######################################################################
###############             REINFORCE                #################
######################################################################
######################################################################

class ConstantStep(object):
    def __init__(self, learning_rate, annealing=False, decay=1):
        self.learning_rate = learning_rate
        self.annealing = annealing
        self.decay = decay
        self.it = 0

    def update(self, gt):
        if self.annealing:
            self.it+=1
            return self.learning_rate/(self.it**self.decay) * gt
            
        return self.learning_rate * gt

class AdamStep():
    
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8, alpha=0.1):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.alpha = alpha
        
        self.it = 0
        self.mt = 0
        self.vt = 0
        
    def update(self, gt):
        self.it += 1
        self.mt = self.beta_1*self.mt + (1-self.beta_1)*gt
        self.vt = self.beta_2*self.vt + (1-self.beta_2)*(gt**2)
        self.mt_hat = self.mt / (1-self.beta_1**self.it)
        self.vt_hat = self.vt / (1-self.beta_2**self.it)
        
        return self.alpha * self.mt_hat/ (np.sqrt(self.vt_hat) + self.epsilon)    

class gaussian_policy():
    def __init__(self, theta, Sigma):
        self.theta = theta
        self.Sigma = np.eye(1)*Sigma
    
    def draw_action(self, state):
        return np.random.multivariate_normal(self.theta*np.array(state), self.Sigma)

    
class REINFORCE():    
    def __init__(self, env, sigma_w, N, T, n_itr, discount, learning_rate, nb_simu, stepper_type='Adam'):
        
        self.env = env
        self.sigma = sigma_w
        self.N = N
        self.T = T
        self.n_itr = n_itr
        self.discount = discount
        self.learning_rate = learning_rate
        self.nb_simu = nb_simu
        self.stepper = stepper_type
        
        self.mean_parameters = None
        self.avg_return = None          
    
    def discretization_2d(self, x, y, binx, biny):
        _, _, _, binid = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
        return binid
    
    def collect_episodes(self, mdp, policy=None, horizon=None, n_episodes=1, render=False):
        paths = []

        for _ in range(n_episodes):
            observations = []
            actions = []
            rewards = []
            next_states = []

            state = self.env.reset()
            for _ in range(horizon):
                action = policy.draw_action(state)
                next_state, reward, terminal, _ = mdp.step(action)
                if render:
                    mdp.render()
                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = copy.copy(next_state)
                if terminal:
                    # Finish rollout if terminal state reached
                    break
                    # We need to compute the empirical return for each time step along the
                    # trajectory

            paths.append(dict(
                states=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_states=np.array(next_states)
            ))
        return paths    
        
    def run(self, explore_bonus=False, beta=3000, fixed_beta=False):
        if self.stepper == 'Constant':
            stepper = ConstantStep(self.learning_rate, annealing=False)
        if self.stepper == 'Adam':
            stepper = AdamStep(alpha=self.learning_rate)
        gammas = [self.discount**k for k in range(self.T)]
        self.mean_parameters = []
        self.avg_return = []
        self.beta = beta

        for k in tqdm(range(self.nb_simu), desc="Running {} simulations".format(self.nb_simu)):
            thetas=[0]
            episodes_perf=[]
            for _ in range(self.n_itr):
                theta = thetas[-1]
                policy = gaussian_policy(theta, self.sigma)
                paths = self.collect_episodes(mdp=self.env, policy=policy, horizon=self.T, n_episodes=self.N)
                gts = []
                rewards_disc = []
                
                for path in paths:
                    rewards = path["rewards"]
                    if explore_bonus:
                        labels = self.discretization_2d(path["states"].flatten(),path["actions"].flatten(),5,5)
                        uniques, counts = np.unique(labels, return_counts=True)
                        get_nb = np.vectorize(lambda x : counts[list(uniques).index(x)])
                        if not fixed_beta: self.beta = np.max(rewards)/(1-self.discount)
                        rewards = rewards + self.beta * np.sqrt(1/get_nb(labels))                     
                    rewards_disc += [np.dot(rewards, gammas)]
                    gts += [ rewards_disc[-1] * np.sum(path["states"]*(path["actions"]-theta*path["states"])/(self.sigma**2))]            

                theta += stepper.update(np.mean(gts))
                thetas += [theta]
                episodes_perf += [np.mean(rewards)]
            self.mean_parameters+=[thetas] 
            self.avg_return+=[episodes_perf]

    def plot_results(self, theta_true=-0.6, show_std=False):        
        
        theta=np.mean(self.mean_parameters,axis=0) - theta_true
        perf=np.mean(self.avg_return,axis=0)
        
        f, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))

        ax1.plot(theta) 
        if show_std:
            ax1.fill_between(np.arange(self.n_itr+1), theta+np.array(self.mean_parameters).std(axis=0), theta-np.array(self.mean_parameters).std(axis=0), alpha=0.2)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title("Convergence of $\Theta_k$")
        ax1.set_ylabel("$\Theta_k - \Theta^*$")
        ax1.set_xlabel("Iterations K")
        
        ax2.plot(perf)
        if show_std:
            ax2.fill_between(np.arange(self.n_itr), perf+np.array(self.avg_return).std(axis=0), perf-np.array(self.avg_return).std(axis=0), alpha=0.2)
        ax2.set_title("Performance along iterations")
        ax2.set_ylabel("Discounted rewards over iterations")
        ax2.set_xlabel("Iterations")
        
        plt.tight_layout()
        plt.show()
        
        
######################################################################
######################################################################
###################             FQI                ###################
######################################################################
######################################################################


class behav_policy():
    def __init__(self, actions):
        self.actions = actions
    
    def draw_action(self, state):
        return self.actions[np.random.randint(self.actions.shape[0])]

class FQI():
    def __init__(self, env, states, actions, discount,lambd=0.1):
        self.env = env
        self.states = states
        self.actions = actions
        self.discount = discount
        self.lambd = lambd
        self.theta = None
        self.theta_history = None
        
    def collect_episodes(self, policy=None, horizon=None, n_episodes=1, render=False, performance=False):
        paths = []

        for _ in range(n_episodes):
            observations = []
            actions = []
            rewards = []
            next_states = []

            state = self.env.reset()
            for _ in range(horizon):
                if performance:
                    state = min(list(self.states), key=lambda x:abs(x-state))
                    action=self.actions[policy[list(self.states).index(state)]]
                else:
                    action = policy.draw_action(state)
                next_state, reward, terminal, _ = self.env.step(action)
                if render:
                    self.env.render()
                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = copy.copy(next_state)
                if terminal:
                    # Finish rollout if terminal state reached
                    break
                    # We need to compute the empirical return for each time step along the
                    # trajectory

            paths.append(dict(
                states=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_states=np.array(next_states)
            ))
        return paths
    
    def estimate_performance(self, policy):
        paths = self.collect_episodes(policy, horizon=50, n_episodes=50, performance=True)
        gammas = [self.discount**k for k in range(50)]
        J = [path["rewards"].dot(gammas) for path in paths]
        return np.mean(J)
    
    def create_dataset(self, n_episodes, horizon):
        return self.collect_episodes(n_episodes=n_episodes, policy=behav_policy(self.actions), horizon=horizon)
    
    def bellman_op(self, theta, states, actions):
        states=states.flatten()
        return np.max(theta[0]*actions + theta[1]*(actions*states[:,None]) +  theta[2]*((actions**2) + states[:,None]**2), axis=1)

    def compute_Q(self, theta, states, actions):
        states=states.flatten()
        return theta[0]*actions + theta[1]*(actions*states[:,None]) +  theta[2]*((actions**2) + states[:,None]**2)
    
    def run_fqi(self, dataset):
        self.theta_history=[np.random.uniform(-2,2,3)]
        self.policy_history=[np.argmax(self.compute_Q(self.theta_history[-1], self.states, self.actions), axis=1)]
        
        for path in tqdm(dataset, desc="FQI on {} episodes: ".format(len(dataset))):
            p_actions = path["actions"].reshape(-1,1)
            p_states = path["states"].reshape(-1,1)
            X = np.hstack([p_actions, p_states*p_actions, p_states**2 + p_actions**2])
            y = path["rewards"] + self.discount*self.bellman_op(self.theta_history[-1],path["next_states"], self.actions)
            self.theta_history += [np.linalg.solve(np.matmul(X.T,X) + self.lambd*np.eye(X.shape[1]), np.dot(X.T,y))]
            self.policy_history += [np.argmax(self.compute_Q(self.theta_history[-1], self.states, self.actions), axis=1)]
        
        self.theta = self.theta_history[-1]
    
    def plot_performance(self):
        plt.plot([self.estimate_performance(policy) for policy in self.policy_history])
        plt.ylabel("$J(\pi_k)$")
        plt.xlabel("Iterations k")
        plt.title("Performance of the algorithm")
        plt.show()         
        
    def plot_Q(self):       
        # Compute Q optimal
        def make_grid(x, y):
            m = np.meshgrid(x, y, copy=False, indexing='ij')
            return np.vstack(m).reshape(2, -1).T

        SA = make_grid(self.states, self.actions)
        S, A = SA[:, 0], SA[:, 1]
        
        K, cov = self.env.computeOptimalK(self.discount), 0.001
        print('Optimal K: {} Covariance S: {}'.format(K, cov))
        
        Q_fun_ = np.vectorize(lambda s, a: self.env.computeQFunction(s, a, K, cov, self.discount, 1))
        Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])

        Q_opt = Q_fun(SA)
        
        # Compute resulted Q with FQI
        self.Q = self.compute_Q(self.theta, self.states, self.actions)

        # Plot Q and Q_opt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(S, A, Q_opt, label="Q opt")
        ax.scatter(S, A, self.Q, label="Q FQI")
        plt.legend(loc="upper left")
        plt.show()