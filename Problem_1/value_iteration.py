import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm, trange

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 1

    # perform value iteration
    for _ in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim] - represents the reward for each state

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        Vs = []
        for i in range(adim):
            Vs.append(tf.where(tf.cast(terminal_mask, tf.bool),
                x=reward,
                y=reward + gam * tf.linalg.matvec(Ts[i], V)))
        V_new = tf.reduce_max(Vs, axis=0)

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition

        err = tf.linalg.norm(V_new - V)
        V = V_new

        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V

def simulate_trajectory(problem, V, goal_idx):
    print("Start trajectory simulation")
    Ts = problem["Ts"]
    pos2idx = problem["pos2idx"]
    idx2pos = problem["idx2pos"]
    n = problem["n"]
    sdim = n ** 2
    pt = np.array([0, 0]) # initial state
    pt_min, pt_max = [0, 0], [n - 1, n - 1]

    Xs = [pt[0]]
    Ys = [pt[1]]
    # simulate up to 100 steps
    for i in trange(100):
        # compute optimal action
        pt_right = np.clip(pt + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(pt + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(pt + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(pt + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        action = np.argmax(Vs)

        # sample from state transition distribution
        S = pos2idx[pt[0], pt[1]]
        S_new_sample = np.random.choice(np.arange(sdim),
                                        p=Ts[action][S].numpy())
        pt = idx2pos[S_new_sample]
        Xs.append(pt[0])
        Ys.append(pt[1])

        if S_new_sample == goal_idx:
            break

    plt.plot(Xs, Ys, "r")

# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim = n * n

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    goal_idx = problem["pos2idx"][19, 9]
    reward = np.zeros([sdim])
    reward[goal_idx] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)
    V_opt = np.array(V_opt).reshape((n, n))

    plt.figure(213)
    visualize_value_function(V_opt)
    simulate_trajectory(problem, V_opt, goal_idx)
    plt.title("value iteration")
    plt.show()


if __name__ == "__main__":
    main()
