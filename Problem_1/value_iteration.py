import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm, trange

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for _ in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        Vs = []
        for i in range(adim):
            Vs.append(tf.where(tf.cast(terminal_mask, tf.bool),
                x=reward[:, i],
                y=reward[:, i] + gam * tf.linalg.matvec(Ts[i], V)))
        V_new = tf.reduce_max(Vs, axis=0)

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition

        err = tf.linalg.norm(V_new - V)
        V = V_new

        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V

def simulate_trajectory(problem, V, R, gam, goal_idx):
    print("Start trajectory simulation")
    Ts = problem["Ts"]
    pos2idx = problem["pos2idx"]
    idx2pos = problem["idx2pos"]
    n = problem["n"]
    sdim = n ** 2
    adim = len(Ts)
    S = pos2idx[0, 0] # initial state

    Xs = [0]
    Ys = [0]
    # simulate up to 100 steps
    for _ in trange(100):
        # compute optimal action
        Qs = []
        for a in range(adim):
            Qs.append(R[S, a] + gam * tf.reduce_sum(V * Ts[a][S]))
        action = tf.argmax(Qs, axis=0)

        # sample from state transition distribution
        S_new_sample = np.random.choice(np.arange(sdim),
                                        p=Ts[action][S].numpy())

        pt = idx2pos[S_new_sample]
        Xs.append(pt[0])
        Ys.append(pt[1])

        if S_new_sample == goal_idx:
            break

        S = S_new_sample

    plt.plot(Xs, Ys, "r")

# shared function to compare with q learning
def run_value_iter(problem):
    n = problem["n"]
    sdim = n * n
    goal_idx = problem["pos2idx"][19, 9]

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[goal_idx] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[goal_idx, :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    return V_opt


# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim = n * n
    goal_idx = problem["pos2idx"][19, 9]

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[goal_idx] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[goal_idx, :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(V_opt.numpy().reshape((n, n)))
    simulate_trajectory(problem, V_opt, reward, gam, goal_idx)
    plt.title("value iteration")
    plt.show()


if __name__ == "__main__":
    main()
