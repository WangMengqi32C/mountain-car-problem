import os
import sys
import time
import datetime
from contextlib import contextmanager
from joblib import dump, load
from pathlib import Path
import pickle

import random
from math import sqrt, exp
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans

from INFO8003.continuous_domain.display_caronthehill import save_caronthehill_image

import cv2
import matplotlib
matplotlib.use("TkAgg")  # Fixing a bug of matplotlib on MacOS
from matplotlib import pyplot as plt


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label, datetime.timedelta(seconds=end-start)))


# ----------------------------------------------------------------------------
class Domain:
    def __init__(self):
        """
        Initialize the domain.
        """
        self.discrete_time = 0.1
        self.integration_time_step = 0.001
        self.discount = 0.95
        self.m = 1.0
        self.g = 9.81
        self.actions = [-4, 4]

    def is_terminal_state(self, p, s):
        """
        Check if the agent has reached a terminal state.
        NB : A terminal state is reached if |p| > 1 or |s| > 3.
        """
        if abs(p) > 1 or abs(s) > 3:
            return True
        return False

    def dynamics(self, p, s, u):
        """
        Implementation of the dynamics of the domain.
        Given a position p, a speed s and an action u, returns the speed
        and the acceleration.
        """
        speed = s
        dh = self.hill_first_deriv(p)
        ddh = self.hill_second_deriv(p)
        den = 1 + dh**2
        acc = (u/(self.m * den)) - ((self.g * dh)/den) - \
            ((s**2 * dh * ddh)/den)
        return speed, acc

    def discrete_time_dynamics(self, p, s, u):
        """
        Implementation of the discrete-time dynamics with the Euler
        integration method.
        Given a position p, a speed s and an action u at time t, returns the
        next position and speed at time t+1.
        """
        next_p = p
        next_s = s
        time_steps = int(self.discrete_time/self.integration_time_step)

        for t in range(time_steps):
            if self.is_terminal_state(next_p, next_s):
                break
            speed, acc = self.dynamics(next_p, next_s, u)
            next_p += speed * self.integration_time_step
            next_s += acc * self.integration_time_step
        return next_p, next_s

    def reward_signal(self, p, s, u):
        """
        Get the reward from doing action u in state (p, s).
        """
        next_p, next_s = self.discrete_time_dynamics(p, s, u)

        # If terminal state, reward is 0
        if self.is_terminal_state(next_p, next_s) and (next_p == p and
                                                       next_s == s):
            return 0
        # If did not climb the hill, reward is -1
        elif next_p < -1 or abs(next_s) > 3:
            return -1
        # If did climb the hill, reward is 1
        elif next_p > 1 and abs(next_s) <= 3:
            return 1
        else:
            return 0

    def hill(self, p):
        """
        Given a position p, compute the Hill function.
        """
        return p**2+p if p < 0 else p/pow(1 + 5 * p**2, 0.5)

    def hill_first_deriv(self, p):
        """
        Given a position p, computes the first derivative of the Hill
        function.
        """
        return 2*p+1 if p < 0 else 1/pow(1 + 5 * p**2, 1.5)

    def hill_second_deriv(self, p):
        """
        Given a position p, computes the second derivative of the Hill
        function.
        """
        return 2 if p < 0 else (-15*p)/pow(1 + 5 * p**2, 2.5)


# ----------------------------------------------------------------------------
class Agent:
    def __init__(self, domain):
        """
        Initialize a simple agent.
        """
        self.domain = domain

    def select_action(self, p, s):
        """
        Select a random action.
        """
        return random.choice(self.domain.actions)

    def compute_score(self):
        """
        Routine which estimates the expected return of a policy by computing
        an average over a set of random initial states.
        """
        J_values = np.zeros((15, 15))
        for i in range(-7, 8):
            for j in range(-7, 8):
                # Set the initial state
                p = 0.125*i
                s = 0.375*j
                # Compute the J-function in that state
                J_values[i, j] = self.expected_return(p, s)

        # Compute the expected return of the policy
        score = np.mean(J_values)
        return score

    def expected_return(self, p, s):
        """
        Compute the J function recursively for a given state (p, s)
        of the domain.
        """
        ret = 0
        for i in range(200):
            # Choose the action according to the policy
            u = self.select_action(p, s)
            # Get the associated reward
            r = self.domain.reward_signal(p, s, u)
            # Get the next state
            p, s = self.domain.discrete_time_dynamics(p, s, u)
            # Update the cumulative reward
            ret += (self.domain.discount**i) * r
            if self.domain.is_terminal_state(p, s):
                break
        return ret

    def generate_four_tuples(self, n, max_size, start_p=None, start_s=None):
        """
        Generate the set of four-tuples F from n random trajectories of
        a given maximal size, all starting in a given state.
        """
        set_f = []
        # Generate n random trajectories
        for traj in range(n):
            # If no initial state given, choose one at random
            p = random.uniform(-1, 1) if (start_p is None) else start_p
            s = random.uniform(-3, 3) if (start_s is None) else start_s

            # Generate one trajectory of maximum max_size
            for i in range(max_size):
                # Choose a random action
                u = random.choice(self.domain.actions)
                # Get the associated reward
                r = self.domain.reward_signal(p, s, u)
                # Compute the next state according to this action
                next_p, next_s = self.domain.discrete_time_dynamics(p, s, u)
                # Add to the set
                set_f.append([p, s, u, r, next_p, next_s])
                # Check a terminal state
                if self.domain.is_terminal_state(next_p, next_s):
                    break
                # Update the state
                p = next_p
                s = next_s
        return np.array(set_f)

    def compute_bellman_residuals(self, current_Q, previous_Q):
        """
        Given Q_N and Q_{N-1}, compute the Bellman residual as defined
        in the report.
        """
        diff = np.subtract(current_Q, previous_Q)
        squared_diff = np.square(diff)
        return np.sum(squared_diff) / current_Q.shape[0]

    def test_policy(self, p=None, s=None):
        """
        Generate a trajectory by following the current policy.
        """
        h_t = []
        # If no initial state given, choose one at random
        if (p is None) or (s is None):
            p = random.uniform(-1, 1)
            s = random.uniform(-3, 3)

        # Generate the trajectory
        h_t.append((p, s))
        count = 0
        while not self.domain.is_terminal_state(p, s) and count <= 2000:
            # Choose the action according to the policy
            u = self.select_action(p, s)
            # Get the associated reward
            r = self.domain.reward_signal(p, s, u)
            # Get the next state
            p, s = self.domain.discrete_time_dynamics(p, s, u)
            # Add to the trajectory
            h_t.append(u)
            h_t.append(r)
            h_t.append((p, s))
            count += 1
        return h_t

    def plot_distances(self, distances, model_name):
        """
        Plot the Bellman residuals as defined in the report through N.
        """
        plt.plot(range(len(distances)), distances)
        plt.title('Bellman residuals')
        plt.xlabel('N')
        plt.ylabel('d(Q_N, Q_N-1)')
        plt.savefig("Figures/{}/Distances/distances.eps".format(model_name))
        plt.close()

    def plot_score(self, returns, model_name):
        """
        Plot the expected returns of a policy through N.
        """
        plt.plot(range(len(returns)), returns)
        plt.title('Score of the policy')
        plt.xlabel('N')
        plt.ylabel('J')
        plt.savefig("Figures/{}/Expected_returns/score.eps".format(model_name))
        plt.close()

    def produce_video(self, trajectory, model_name, iteration):
        """
        Routine that produces a video from a "car on the hill" trajectory.
        """
        images = []
        video_name = "Figures/{}/Videos/video_mu{}.avi".format(model_name, iteration)
        # Create the images
        i = 0
        while i <= len(trajectory):
            p, s = trajectory[i]
            save_caronthehill_image(p, s, "tmp_img.jpeg")
            images.append(cv2.imread("tmp_img.jpeg"))
            i += 3
        # Remove useless files
        os.remove("tmp_img.jpeg")
        try:
            os.remove(video_name)
        except OSError:
            pass
        # Create the movie (8 FPS)
        vid = cv2.VideoWriter(video_name,
                              cv2.VideoWriter_fourcc(*'DIVX'), 10, (400, 400))
        for j in range(len(images)):
            vid.write(images[j])
        vid.release()

    def plot_trajectory(self, trajectory, model_name, iteration):
        """
        Plot (p,s) for each transition of a given trajectory.
        """
        i = 0
        j = 0
        positions = np.zeros(int((len(trajectory)-1)/3)+1)
        speeds = np.zeros(int((len(trajectory)-1)/3)+1)
        while i <= len(trajectory):
            p, s = trajectory[i]
            positions[j] = p
            speeds[j] = s
            i += 3
            j += 1
        # plt.scatter(positions, speeds, s=1.5)
        plt.plot(positions, speeds)
        plt.title("Trajectory")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.xlim(-1, 1)
        plt.ylim(-3, 3)
        plt.savefig("Figures/{}/Optimal_trajectory/traj_mu_{}.eps".format(model_name, iteration))
        plt.close()


# ----------------------------------------------------------------------------
class Fitted_Q_Agent(Agent):
    figure_path="/home/wmq1/Study/Intern/mountain-car-problem"
    def __init__(self, domain,path):
        """
        Initialize a simple agent.
        """
        super().__init__(domain)
        self.model = None
        self.figure_path=path

    def select_action(self, p, s):
        """
        Select an action either to the current policy of the agent if
        one exists, or random.
        """
        if self.model is None:
            return random.choice(self.domain.actions)
        else:
            # Compute Q for u=4 and u=-4 and choose max
            X = np.zeros((len(self.domain.actions), 3))
            for u_index, u in enumerate(self.domain.actions):
                X[u_index, :] = p, s, u
            y = self.model.predict(X)
            return X[np.argmax(y), 2]

    def fitted_Q_iteration(self, n, model, model_name):
        """
        Routine which computes Q_N (N=1,2,...) using Fitted-Q-Iteration with
        a given model as the supervised learning technique.
        """
        # Reuse same four-tuple set
        set_file = Path("four_tuple_set.npy")
        if set_file.is_file():
            print("Loading the four-tuple set...")
            four_tuples = load('four_tuple_set.npy')
        else:
            print("Generating a four-tuple set...")
            four_tuples = self.generate_four_tuples(1000, 1000, start_p=-0.5, start_s=0)
            dump(four_tuples, 'four_tuple_set.npy')
        print("Total number of samples in the four-tuples set : {}".format(len(four_tuples)))

        # # Plot variables
        returns = np.zeros(n)
        distances = np.zeros(n)
        previous_Q = np.zeros(four_tuples.shape[0])
        current_Q = np.zeros(four_tuples.shape[0])

        with measure_time('Computing Q...'):
            # Iteration N=1
            print("Computing iteration 1 ...")
            X = four_tuples[:, 0:3]  # Take all {p,s,u} of F
            y = four_tuples[:, 3]  # Take all corresponding {r}
            model.fit(X, y)  # Train the model
            self.model = model
            directory='Figures/{}/Models/Q1.pkl'.format(model_name)
            path = os.path.join(self.figure_path, directory) 
            os.makedirs(path)
            print("Saving model iteration 1...")
            pickle.dump(model, open('Figures/{}/Models/Q1.pkl'.format(model_name)))

            # # Compute the Bellman residual
            print("Computing Bellman residual iteration 1 ...")
            current_Q = model.predict(X)  # Predict Q
            distances[0] = self.compute_bellman_residuals(current_Q, previous_Q)
            previous_Q = current_Q

            # # Compute the expected return
            print("Computing expected return iteration 1 ...")
            returns[0] = self.compute_score()

            # Iteration N > 1
            for i in range(n-1):
                print("Computing iteration {} ...".format(i+2))

                # Build the training set (only update output, input unchanged)
                print("Building training set iteration {} ...".format(i+2))
                Q_array = np.zeros((four_tuples.shape[0], 2))
                for u_index, u in enumerate(self.domain.actions):
                    X_next = np.zeros((four_tuples.shape[0], 3))  # {p', s', u}
                    X_next[:, 0:2] = four_tuples[:, 4:6]  # {p',s'}
                    X_next[:, 2] = u  # {u}
                    Q_array[:, u_index] = model.predict(X_next)  # Q for all samples for both actions
                rewards = four_tuples[:, 3]
                new_y = rewards + self.domain.discount * np.max(Q_array, axis=1)

                # Train the model
                print("Training model iteration {} ...".format(i+2))
                model.fit(X, new_y)
                self.model = model

        #         # Save the model
                print("Saving model iteration {} ...".format(i+2))
                save_model=open("Figures/{}/Models/Q{}.pkl".format(model_name,i+2),"wb")
                pickle.dump(model, save_model)

        #         # Compute the Bellman residual
                print("Computing Bellman residuals iteration {} ...".format(i+2))
                current_Q = model.predict(X)
                distances[i+1] = self.compute_bellman_residuals(current_Q, previous_Q)
                previous_Q = current_Q

        #         # Compute the expected return
                print("Computing expected return iteration {} ...".format(i+2))
                returns[i+1] = self.compute_score()

                # For N = 5, 10, 20, 50
                if i in [0,3,8]:
                    # Create heatmaps
                    print("Create heatmaps iteration {} ...".format(i+2))
                    self.create_heatmaps(model_name, i+2)

                    # Create optimal trajectory
                    print("Create optimal trajectory iteration {} ...".format(i+2))
                    opt_traj = self.test_policy(p=-0.5, s=0)
                    dump(opt_traj,'Figures/{}/Optimal_trajectory/traj_mu_{}.txt'.format(model_name, i+2))
                    self.plot_trajectory(opt_traj, model_name, i+2)

                    # Create a video from that trajectory
                    # print("Create video iteration {} ...".format(i+2))
                    # self.produce_video(opt_traj, model_name, i+2)

        # # Save distances (Bellman residuals) and plot them
        print("Save and plot distances...")
        dump(distances,'Figures/{}/Distances/distances.txt'.format(model_name))
        self.plot_distances(distances, model_name)

        # # Save returns and plot them
        print("Save and plot returns...")
        dump(returns, 'Figures/{}/Expected_returns/score.txt'.format(model_name))
        self.plot_score(returns, model_name)

    def create_heatmaps(self, model_name, iteration):
        """
        Create heatmaps of the policy, of Q(.,-4) and of Q(.,4)
        """
        heatmap_policy = np.zeros([100, 100])
        heatmap_acc = np.zeros([100, 100])
        heatmap_decc = np.zeros([100, 100])
        p = np.linspace(-1, 1, 100)
        s = np.linspace(-3, 3, 100)
        for x in range(100):
            for y in range(100):
                Q_acc = self.model.predict([[p[x], s[y], 4]])[0]
                Q_decc = self.model.predict([[p[x], s[y], -4]])[0]
                heatmap_acc[x, y] = Q_acc
                heatmap_decc[x, y] = Q_decc
                if Q_acc - Q_decc > 0:
                    heatmap_policy[x, y] = 4
                elif Q_decc - Q_acc > 0:
                    heatmap_policy[x, y] = -4
                else:
                    heatmap_policy[x, y] = 0

        # Heatmap of Q(.,4)
        cs = plt.contourf(p, s, heatmap_acc, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(., 4)")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.savefig("Figures/{}/Heatmaps/Q_acc_{}.eps".format(model_name, iteration))
        print("here")
        plt.close()

        # Heatmap of Q(.,-4)
        cs = plt.contourf(p, s, heatmap_decc, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(., -4)")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.savefig("Figures/{}/Heatmaps/Q_decc_{}.eps".format(model_name, iteration))
        plt.close()

        # Heatmap of mu
        cs = plt.contourf(p, s, heatmap_policy, cmap='coolwarm', levels=[-4, 0, 4])
        plt.colorbar(cs)
        plt.title("mu(.)")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.savefig("Figures/{}/Heatmaps/mu_{}.eps".format(model_name, iteration))
        plt.close()


# ----------------------------------------------------------------------------
class Parametric_Q_Agent(Agent):
    def __init__(self, domain, n, alpha, epsilon):
        """
        Initialize a the parametric Q agent.
        """
        super().__init__(domain)
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.weights = np.zeros(self.n)
        self.centers = np.zeros(self.n)
        self.sigma = 1
        self.greedy = True

    def compute_sigma(self):
        """
        Compute the standard deviation of the Gaussians.
        """
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                distances[i, j] = np.linalg.norm(self.centers[i]-self.centers[j])
        d_max = np.max(distances)
        return d_max/sqrt(2*self.n)

    def basis_function(self, p, s, u):
        """
        Compute the weights vector for a given state-action pair (p, s, u).
        """
        phi = np.zeros(self.n)
        x = np.array([p, s, u])
        # For each neuron, compute RBF
        for i in range(self.n):
            phi[i] = self.rbf(x, self.centers[i], self.sigma)
        return phi

    def rbf(self, x, c, sigma):
        """
        Given a point and the center of a neuron, returns the Gaussian RBF.
        """
        d = np.linalg.norm(x-c)
        return exp(-d**2/(2*self.sigma**2))

    def select_action(self, p, s):
        """
        Select action according to e-greedy policy.
        """
        if (self.greedy) and (random.random() < self.epsilon):
            return random.choice(self.domain.actions)
        else:
            # Compute Q for u=4 and u=-4 and choose max
            Q_acc = np.dot(self.basis_function(p, s, 4), self.weights)
            Q_decc = np.dot(self.basis_function(p, s, -4), self.weights)
            if Q_acc > Q_decc:
                return 4
            elif Q_decc > Q_acc:
                return -4
            else:
                return random.choice(self.domain.actions)

    def parametric_Q_learning(self, episodes):
        """
        Parametric Q learning algorithm.
        """
        # # Plot variable
        # returns = []

        # Get four-tuple set in order to cluster
        four_tuple_file = Path("four_tuple_set.npy")
        if four_tuple_file.is_file():
            print("Loading the four-tuple set...")
            four_tuples = load('four_tuple_set.npy')
        else:
            print("Generating a four-tuple set of 1000 episodes...")
            four_tuples = self.generate_four_tuples(1000, 1000, start_p=-0.5, start_s=0)
            dump(four_tuples, 'four_tuple_set.npy')
        print("Total number of samples in the four-tuples set : {}".format(len(four_tuples)))

        with measure_time('Training...'):
            # Cluster
            print("Clustering...")
            X = four_tuples[:, 0:3]
            clusters = KMeans(n_clusters=self.n).fit(X)
            self.centers = clusters.cluster_centers_

            # Compute sigma
            self.sigma = self.compute_sigma()

            # Update weights through multiple episodes
            for i in range(episodes):
                # Begin a new episode
                print("Episode : {}".format(i))
                # Initial state
                p = -0.5
                s = 0
                count = 0
                while not self.domain.is_terminal_state(p, s) and count <= 200:
                    # Choose the action according to the policy
                    u = self.select_action(p, s)
                    # Get the associated reward
                    r = self.domain.reward_signal(p, s, u)
                    # Get the next state
                    p_next, s_next = self.domain.discrete_time_dynamics(p, s, u)
                    # Compute max Q
                    Q_acc = np.dot(self.basis_function(p_next, s_next, 4), self.weights)
                    Q_decc = np.dot(self.basis_function(p_next, s_next, -4), self.weights)
                    Q_max = max(Q_acc, Q_decc)
                    # Compute current Q
                    phi_curr = self.basis_function(p, s, u)
                    Q_curr = np.dot(phi_curr, self.weights)
                    # Update the weights
                    self.weights = np.add(self.weights, (self.alpha * (r + self.domain.discount*Q_max - Q_curr)) * phi_curr)
                    # Update the state
                    p = p_next
                    s = s_next
                    # Update counter
                    count += 1

        #         if i in range(0, episodes, 40):
        #             # Compute the score
        #             print("Computing score episode {} ...".format(i))
        #             returns.append(self.compute_score())

        #         if i in range(100, episodes, 200):
        #             # Create heatmaps
        #             print("Create heatmaps episode {} ...".format(i))
        #             self.create_heatmaps(i)

        #             # Create optimal trajectory
        #             print("Create optimal trajectory episode {} ...".format(i))
        #             self.greedy = False
        #             opt_traj = self.test_policy(p=-0.5, s=0)
        #             dump(opt_traj, 'Figures/ParamQ/Optimal_trajectory/{}/traj_mu_episode{}.txt'.format(self.n, i))
        #             self.plot_trajectory(opt_traj, "ParamQ", i)
        #             self.greedy = True

        #             # Save the weights
        #             dump(self.weights,'Figures/ParamQ/Weights/{}/weights_episode{}.txt'.format(self.n, i))

        # # Create a video from that trajectory
        # print("Create video ...")
        # self.produce_video(opt_traj, "ParamQ", self.n)

        # # Save returns and plot them
        # print("Save and plot scores...")
        # dump(returns, 'Figures/ParamQ/Expected_returns/scores{}neurons.txt'.format(self.n))
        # self.plot_score(returns, "ParamQ")

    def create_heatmaps(self, episode):
        """
        Create heatmaps of the policy, of Q(.,-4) and of Q(.,4)
        """
        heatmap_policy = np.zeros([100, 100])
        heatmap_acc = np.zeros([100, 100])
        heatmap_decc = np.zeros([100, 100])
        p = np.linspace(-1, 1, 100)
        s = np.linspace(-3, 3, 100)
        for x in range(100):
            for y in range(100):
                Q_acc = np.dot(self.basis_function(p[x], s[y], 4), self.weights)
                Q_decc = np.dot(self.basis_function(p[x], s[y], -4), self.weights)
                heatmap_acc[x, y] = Q_acc
                heatmap_decc[x, y] = Q_decc
                if Q_acc - Q_decc > 0:
                    heatmap_policy[x, y] = 4
                elif Q_decc - Q_acc > 0:
                    heatmap_policy[x, y] = -4
                else:
                    heatmap_policy[x, y] = 0

        # Heatmap of Q(.,4)
        cs = plt.contourf(p, s, heatmap_acc, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(., 4)")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.savefig("Figures/ParamQ/Heatmaps/{}/Q_acc_episode{}.eps".format(self.n, episode))
        plt.close()

        # Heatmap of Q(.,-4)
        cs = plt.contourf(p, s, heatmap_decc, cmap='Spectral')
        plt.colorbar(cs)
        plt.title("Q(., -4)")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.savefig("Figures/ParamQ/Heatmaps/{}/Q_decc_episode{}.eps".format(self.n, episode))
        plt.close()

        # Heatmap of mu
        cs = plt.contourf(p, s, heatmap_policy, cmap='coolwarm', levels=[-4, 0, 4])
        plt.colorbar(cs)
        plt.title("mu(.)")
        plt.xlabel("p")
        plt.ylabel("s")
        plt.savefig("Figures/ParamQ/Heatmaps/{}/mu_episode{}.eps".format(self.n, episode))
        plt.close()


# ----------------------------------------------------------------------------
def plot_trajectory_distribution(self, trajectory):
    """
    Given a set of trajectories, plot the state space distribution.
    """
    positions = []
    speeds = []
    sample_count = 0
    for sample in trajectory:
        positions.append(sample[0])
        speeds.append(sample[1])
        sample_count += 1
    print("Samples considered in the trajectory: {}".format(sample_count))
    plt.scatter(positions, speeds,  s=0.2)
    plt.title("State distribution in the trajectory")
    plt.xlabel("p")
    plt.ylabel("s")
    plt.ylim(-3, 3)
    plt.xlim(-1, 1)
    plt.show()


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Figure Directory 
    figure_dir = "/home/wmq1/Study/Intern/mountain-car-problem"

    # Init of the instances
    domain = Domain()
    FQI_agent = Fitted_Q_Agent(domain,figure_dir)
    paramQ_agent = Parametric_Q_Agent(domain, 210, 0.01, 0.9)

    # # Fitted Q-Iteration
    #lin_reg = LinearRegression()
    #FQI_agent.fitted_Q_iteration(5, lin_reg, "Linear_regression")

    trees = ExtraTreesRegressor(n_estimators=50)
    FQI_agent.fitted_Q_iteration(10, trees, "Extra_trees")

    #neural_net = MLPRegressor(hidden_layer_sizes=(15, 20),
    #                          activation='tanh',
    #                          solver='adam',
    #                          max_iter=2000)
    #FQI_agent.fitted_Q_iteration(50, neural_net, "Neural_nets")

    # # Parametric Q-learning
    # paramQ_agent.parametric_Q_learning(1000)
