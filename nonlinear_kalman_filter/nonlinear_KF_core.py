#%%
from math import e
import os, datetime, tqdm, collections, time

import numpy as np
import matplotlib.pyplot as plt
import torch

RANDOM_SEED = 30

#%%
t_step = 500; trade_occur_rate = 0.1; ask_prop = 0.5

nonlinear_type = "square"  # "abs", "square"

if nonlinear_type == "abs":
    RANDOM_SEED = 30
    F = np.array([[1, 0], [0, 1 - 0.02]])
    Q = np.eye(2)
    R = np.array([[0.1]]).reshape((1, 1))
    x_0 = np.array([80, 0.5]).reshape((-1, 1))

if nonlinear_type == "square":
    RANDOM_SEED = 10
    F = np.array([[1, 0], [0, 1 - 0.02]])
    Q = np.diag([1, 0.2])
    R = np.array([[0.1]]).reshape((1, 1))
    x_0 = np.array([80, 0.1]).reshape((-1, 1))


np.random.seed(RANDOM_SEED)

x = x_0
t_hist = np.linspace(0, 1, t_step)
log_likelihood = 0
x_hist = []; z_hist = []
H_real_hist = []; order_type = []

# simulate Kalman filter
for _ in range(t_step):
    if np.random.uniform(0, 1) > trade_occur_rate:
        x = F @ x + np.random.multivariate_normal(np.zeros(2), Q).reshape((-1, 1))  # x: shape (2, 1)
        x_hist.append(x); z_hist.append(None)
        H_real_hist.append(None); order_type.append(0)
    else:
        x = F @ x + np.random.multivariate_normal(np.zeros(2), Q).reshape((-1, 1))  # x: shape (2, 1)
        ot = 1 if np.random.uniform(0, 1) <= ask_prop else -1
        if nonlinear_type == "abs":
            coef = np.array([1, 0.5*ot]).reshape((1, -1))  # coef: shape (1, 2)
            H_real = np.array([[1, 0.5*ot*np.sign(x[1, 0])]]).reshape((1, -1))  # H: shape (1, 2)
            z = coef @ np.abs(x) + np.random.normal(0, R).reshape((1, 1))  # z: shape (1, 1)
        if nonlinear_type == "square":
            coef = np.array([1, 0.5*ot]).reshape((1, -1))
            H_real = np.array([[1, ot*x[1, 0]]]).reshape((1, -1))
            z = coef @ np.array([x[0, 0], x[1, 0]**2]) + np.random.normal(0, R).reshape((1, 1))  # z: shape (1, 1)

        x_hist.append(x); z_hist.append(z)
        H_real_hist.append(H_real); order_type.append(ot)

x_hist = np.concatenate(x_hist, axis=1)  # x_hist: shape (2, T)
order_type = np.array(order_type)  # order_type: shape (T,)

# Kalman filter estimates with known parameters
x_prior_est_hist = []; p_prior_est_hist = []
innovation_hist = []; innovation_cov_hist = []
optimal_kalman_gain_hist = []
x_posterior_est_hist = []; p_posterior_est_hist = []

x_posterior_est = x_0; p_posterior_est = Q
for t_idx in range(t_step):
    # prediction step
    if t_idx == 0 or order_type[t_idx - 1] != 0:
        x_prior_est = F @ x_posterior_est  # x_prior_est: shape (2, 1)
        p_prior_est = F @ p_posterior_est @ (F.T) + Q  # p_prior_est: shape (2, 2)
    else:
        x_prior_est = F @ x_prior_est  # x_prior_est: shape (2, 1)
        p_prior_est = F @ p_prior_est @ (F.T) + Q  # p_prior_est: shape (2, 2)
    x_prior_est_hist.append(x_prior_est)
    p_prior_est_hist.append(p_prior_est)

    # update step
    if order_type[t_idx] == 0:
        innovation = None; innovation_cov = None
        optimal_kalman_gain = None; x_posterior_est = None; p_posterior_est = None
    else:
        H_real = H_real_hist[t_idx]  # H: shape (1, 2)
        coef = np.array([1, 0.5 * order_type[t_idx]]).reshape((1, -1))  # temp_vec: shape (1, 2)
        if nonlinear_type == "abs":
            H_est = np.array([1, 0.5 * order_type[t_idx] * np.sign(x_prior_est[1, 0])]).reshape((1, -1))  # H: shape (1, 2)
            innovation = z_hist[t_idx] - (coef @ np.array([x_prior_est[0, 0], np.abs(x_prior_est[1, 0])]).reshape((-1, 1))) # innovation: shape (1, 1)
        if nonlinear_type == "square":
            H_est = np.array([1, order_type[t_idx] * x_prior_est[1, 0]]).reshape((1, -1))  # H: shape (1, 2)
            innovation = z_hist[t_idx] - (coef @ np.array([x_prior_est[0, 0], x_prior_est[1, 0]**2]).reshape((-1, 1)))  # innovation: shape (1, 1)
        H = H_est

        innovation_cov = H @ p_prior_est @ H.T + R  # innovation_cov: shape (1, 1)
        optimal_kalman_gain = p_prior_est @ H.T @ np.linalg.inv(innovation_cov)  # shape (2, 1)
        x_posterior_est = x_prior_est + optimal_kalman_gain @ innovation  # shape (2, 1)
        p_posterior_est = (np.eye(2) - optimal_kalman_gain @ H) @ p_prior_est  # shape (2, 2)
        log_likelihood += -0.5 * innovation.T @ np.linalg.inv(innovation_cov) @ innovation - 0.5 * np.log(np.linalg.det(innovation_cov))

    innovation_hist.append(innovation); innovation_cov_hist.append(innovation_cov)
    optimal_kalman_gain_hist.append(optimal_kalman_gain)
    x_posterior_est_hist.append(x_posterior_est)
    p_posterior_est_hist.append(p_posterior_est)

x_prior_est_hist = np.concatenate(x_prior_est_hist, axis=1)  # shape (2, T)
p_prior_est_hist = np.concatenate([np.diag(p).reshape((-1, 1)) for p in p_prior_est_hist], axis=1)

idx_bid = np.where(order_type == -1)[0]
idx_ask = np.where(order_type == 1)[0]

plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(t_hist, x_hist[0, :], label=r"$M_t$")
plt.plot(t_hist, x_prior_est_hist[0, :], label=r"$\hat{M}_t$")
plt.fill_between(t_hist, x_prior_est_hist[0, :] - np.sqrt(p_prior_est_hist[0, :]), x_prior_est_hist[0, :] + np.sqrt(p_prior_est_hist[0, :]), alpha=0.3, color="C1")
y_min, y_max = plt.ylim()
plt.vlines(t_hist[idx_bid], y_min, y_max, color="red", alpha=0.15)
plt.vlines(t_hist[idx_ask], y_min, y_max, color="green", alpha=0.15)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_hist, x_hist[1, :], label=r"$S_t$", color="C0")
plt.plot(t_hist, x_prior_est_hist[1, :], label=r"$\hat{S}_t$", color="C1")
plt.fill_between(t_hist, x_prior_est_hist[1, :] - np.sqrt(p_prior_est_hist[1, :]), x_prior_est_hist[1, :] + np.sqrt(p_prior_est_hist[1, :]), alpha=0.3, color="C1")
y_min, y_max = plt.ylim()
plt.vlines(t_hist[idx_bid], y_min, y_max, color="red", alpha=0.15)
plt.vlines(t_hist[idx_ask], y_min, y_max, color="green", alpha=0.15)
plt.legend()

plt.subplot(4, 1, 3)
if nonlinear_type == "abs":
    plt.plot(t_hist, np.abs(x_hist[1, :]), label=r"$|S_t|$", color="C0")
    plt.plot(t_hist, np.abs(x_prior_est_hist[1, :]), label=r"$\hat{|S_t|}$", color="C1")
if nonlinear_type == "square":
    plt.plot(t_hist, x_hist[1, :]**2, label=r"$S_t^2$", color="C0")
    plt.plot(t_hist, x_prior_est_hist[1, :]**2, label=r"$\hat{S_t^2}$", color="C1")
y_min, y_max = plt.ylim()
plt.vlines(t_hist[idx_bid], y_min, y_max, color="red", alpha=0.15)
plt.vlines(t_hist[idx_ask], y_min, y_max, color="green", alpha=0.15)
plt.legend()

plt.subplot(4, 1, 4)
if nonlinear_type == "abs":
    plt.plot(t_hist, x_hist[0, :] + 0.5 * np.abs(x_hist[1, :]), color="green", alpha=0.3)
    plt.plot(t_hist, x_hist[0, :] - 0.5 * np.abs(x_hist[1, :]), color="red", alpha=0.3)
if nonlinear_type == "square":
    plt.plot(t_hist, x_hist[0, :] + 0.5 * x_hist[1, :]**2, color="green", alpha=0.3)
    plt.plot(t_hist, x_hist[0, :] - 0.5 * x_hist[1, :]**2, color="red", alpha=0.3)
plt.plot(t_hist[idx_ask], [z_hist[i].flatten()[0] for i in idx_ask], "o", color="green", label=r"$A_t$")
plt.plot(t_hist[idx_bid], [z_hist[i].flatten()[0] for i in idx_bid], "o", color="red", label=r"$B_t$")
plt.legend()
plt.tight_layout()
print("log likelihood: ", log_likelihood)

#%%
init_params = "random"  # "manual", "random", "underlying"
epoch_max = 5e5

if init_params == "underlying":
    F_diag_est = torch.tensor(np.diag(F), requires_grad=True).to(torch.float64)
    Q_diag_est = torch.tensor(np.diag(Q), requires_grad=True).to(torch.float64)
    R_est = torch.tensor(np.diag(R), requires_grad=True).to(torch.float64)

if init_params == "manual":
    F_diag_est = torch.tensor(0.1 * np.ones(2), requires_grad=True).to(torch.float64)
    Q_diag_est = torch.tensor(0.5 * np.ones(2), requires_grad=True).to(torch.float64)  # Q_diag_est: shape (2,)
    R_est = torch.tensor(np.ones(1), requires_grad=True).to(torch.float64)  # R_est: shape (1,)

if init_params == "random":
    np.random.seed(int(1000 * time.time()) % (2 ** 32))
    F_diag_est = torch.tensor(np.random.uniform(0, 1, 2), requires_grad=True).to(torch.float64)
    Q_diag_est = torch.tensor(np.random.uniform(0, 1, 2), requires_grad=True).to(torch.float64)  # shape (2,)
    R_est = torch.tensor(np.random.uniform(0, 1, 1), requires_grad=True).to(torch.float64)  # shape (1,)
    np.random.seed(RANDOM_SEED)

def _log_likelihood_torch(F_diag_est, Q_diag_est, R_est):
    F_est = torch.diag(F_diag_est)
    Q_est = torch.diag(Q_diag_est)
    log_likelihood_est = 0
    x_posterior_est = torch.from_numpy(x_0).to(torch.float64)  # x_0: shape (2, 1)
    p_posterior_est = Q_est  # Q: shape (2, 2)

    # simulate Kalman filter
    x = torch.from_numpy(x_0).to(torch.float64)
    for t_idx in range(t_step):
        # prediction step
        if t_idx == 0 or order_type[t_idx - 1] != 0:
            x_prior_est = F_est @ x_posterior_est  # shape (2, 1)
            p_prior_est = F_est @ p_posterior_est @ F_est.T + Q_est  # shape (2, 2)
        else:
            x_prior_est = F_est @ x_prior_est  # shape (2, 1)
            p_prior_est = F_est @ p_prior_est @ F_est.T + Q_est  # shape (2, 2)

        if order_type[t_idx] != 0:
            # true underlying H
            #H = torch.Tensor(H_hist[t_idx]).reshape((1, 2)).to(torch.float64)  # shape (1, 2)
            # estimated H
            z = torch.tensor(z_hist[t_idx]).reshape((1, 1)).to(torch.float64)  # shape (1, 1)
            if nonlinear_type == "abs":
                H = torch.cat([torch.ones(1), 0.5 * order_type[t_idx] * torch.sign(x_prior_est[1])]).reshape((1, -1)).to(torch.float64)  # shape (1, 2)
                innovation = z - H @ x_prior_est  # shape (1, 1)
            if nonlinear_type == "square":
                H = torch.cat([torch.ones(1), (order_type[t_idx] * x_prior_est[[1], 0])]).reshape((1, -1)).to(torch.float64)  # shape (1, 2)
                innovation = (z - (x_prior_est[0, 0] + 0.5*order_type[t_idx]*(x_prior_est[1, 0]**2))).reshape((1, 1))  # shape (1, 1)

            innovation_cov = H @ p_prior_est @ H.T + R_est  # shape (1, 1)
            optimal_kalman_gain = p_prior_est @ H.T @ torch.linalg.inv(innovation_cov)  # shape (2, 1)
            x_posterior_est = x_prior_est + optimal_kalman_gain @ innovation  # shape (2, 1)
            p_posterior_est = (torch.eye(2) - optimal_kalman_gain @ H) @ p_prior_est  # shape (2, 2)
            log_likelihood_est += -0.5 * innovation.T @ torch.linalg.inv(innovation_cov) @ innovation - 0.5 * torch.logdet(innovation_cov)

    return log_likelihood_est

optimizer = torch.optim.Adam([F_diag_est, Q_diag_est, R_est])
train_log = []

for epoch in tqdm.tqdm(range(int(epoch_max))):
    log_ll_torch = _log_likelihood_torch(F_diag_est, Q_diag_est, R_est)
    loss = -log_ll_torch
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_log.append([
        log_ll_torch.detach().cpu().numpy().copy(),
        F_diag_est.detach().cpu().numpy().copy(),
        Q_diag_est.detach().cpu().numpy().copy(),
        R_est.detach().cpu().numpy().copy()
    ])

print("log likelihood -- est:", train_log[-1][0], "underlying =", log_likelihood)
print("F -- est:", train_log[-1][1], "underlying =", np.diag(F))
print("Q -- est:", train_log[-1][2], "underlying =", np.diag(Q))
print("R -- est:", train_log[-1][3], "underlying =", np.diag(R))

epoch_hist = np.arange(1, epoch_max+1)
plt.figure(figsize=(6, 8))

plt.subplot(4, 1, 1)
plt.plot(epoch_hist, [float(x[0]) for x in train_log], label=r"$\hat{\theta}$")
plt.hlines(log_likelihood, epoch_hist[0], epoch_hist[-1], label=r"$\theta$", linestyle="--", color="red")
plt.xscale("log")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(epoch_hist, [np.linalg.norm(x[1] - np.diag(F), ord=1) for x in train_log],
        label=r"$||\hat{F} - F||_1$")
plt.hlines(0, epoch_hist[0], epoch_hist[-1], linestyle="--", color="black")
plt.xscale("log")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(epoch_hist, [np.linalg.norm(x[2] - np.diag(Q), ord=1) for x in train_log],
        label=r"$||\hat{Q} - Q||_1$")
plt.hlines(0, epoch_hist[0], epoch_hist[-1], linestyle="--", color="black")
plt.xscale("log")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(epoch_hist, [np.abs(float(x[3]) - float(R)) for x in train_log],
        label=r"$|\hat{R} - R|$")
plt.hlines(0, epoch_hist[0], epoch_hist[-1], linestyle="--", color="black")
plt.xscale("log")
plt.legend()
plt.tight_layout()

#%%






