import numpy as np
import pandas as pd
import random
import cvxpy as cp
import time

import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.linalg import block_diag
# from sklearn.model_selection import KFold

from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.stats import t
from scipy.stats import cauchy

import warnings
warnings.filterwarnings("ignore")

from mpmath import mp
mp.dps = 1000

import joblib
import ray
from ray.util.joblib import register_ray
import os
import multiprocessing



def generate_coef(p, num_info_aux, num_uninfo_aux, noise = 0.01):
	K = num_info_aux + num_uninfo_aux

	beta_target = np.zeros(p + 1)
	beta_target[0] = 1

	for i in range(1, p + 1):
		if i % 2 == 1:
			beta_target[i] = 0
		else:
			beta_target[i] = 0

	Beta_S = np.zeros((p + 1, K))

	for m in range(K):
		if m < num_uninfo_aux:
			Beta_S[1:, m] += np.random.normal(loc = 0, scale = noise * 5, size = p)
		else:
			Beta_S[1:, m] += np.random.normal(loc = 0, scale = noise, size = p)

	return Beta_S, beta_target


def generate_data(p, nS, nT, num_info_aux, num_uninfo_aux, num_outliers, h_parameter):
	K = num_info_aux + num_uninfo_aux

	Beta_S, beta_target = generate_coef(p, num_info_aux, num_uninfo_aux)
	Beta_all = np.column_stack([Beta_S[:, k] for k in range(K)] + [beta_target])

	ZS_list, YS_list, true_Y_list = [], [], []
	N_vec = [nS] * K + [nT]

	real_outlier_list = []

	for k in range(K + 1):
		Zk_raw = np.random.normal(0, 1, size = (N_vec[k], p))
		intercept_col = np.ones((N_vec[k], 1))
		Zk = np.hstack([intercept_col, Zk_raw])

		true_Yk = Zk @ Beta_all[:, k]
		
		noise = np.random.normal(0, 1, size = N_vec[k])
		# noise = np.random.laplace(0, 1, size = N_vec[k])
		# noise = skewnorm.rvs(a = 10, loc = 0, scale = 1, size = N_vec[k])
		# noise = t.rvs(df = 20, size = N_vec[k])
		# noise = cauchy.rvs(loc = 0, scale = 1, size = N_vec[k])
		# mix = np.random.choice([0, 1], size = N_vec[k], p = [0.9, 0.1])
		# noise = np.where(mix == 0, np.random.normal(0, 1, size = N_vec[k]), np.random.normal(0, 100, size = N_vec[k]))

		Yk = true_Yk + noise

		if k == K and num_outliers > 0:
			n_samples = N_vec[k]
			num_real_outlier = min(num_outliers, n_samples)

			# for i in range(0, num_real_outlier, 2):
			# 	real_outlier_list.append(i)
			# 	real_outlier_list.append(nT - i - 1)
			
			for i in range(0, num_real_outlier, 1):
				real_outlier_list.append(i)

			# outlier_noise = np.random.normal(loc=0, scale=1, size=actual_num_outliers) * outlier_magnitude
			Yk[real_outlier_list] += h_parameter

		ZS_list.append(Zk)
		YS_list.append(Yk)
		true_Y_list.append(true_Yk)

	Z0 = ZS_list[-1]
	Y0 = YS_list[-1]
	ZS_list = ZS_list[:-1]
	YS_list = YS_list[:-1]

	true_Y = np.concatenate(true_Y_list)
	SigmaS_list = [np.identity(nS) for _ in range(K)]
	Sigma0 = np.identity(nT)

	Z = np.vstack(ZS_list + [Z0])
	Y = np.concatenate(YS_list + [Y0])
	Sigma = block_diag(*SigmaS_list, Sigma0)

	# return Z, ZS_list, Z0, Y, YS_list, Y0, true_Y, Sigma, SigmaS_list, Sigma0, Beta_all, beta_target
	return Z, Z0, Y, Y0, true_Y, Sigma, real_outlier_list


def Construct_matrix(Z, Y, gamma):
	N, p = Z.shape

	P = np.zeros((2 * p + 2 * N, 2 * p + 2 * N))
	P[2 * p : 2 * p + N, 2 * p : 2 * p + N] = np.identity(N)

	q_0 = np.zeros(2 * p + 2 * N)
	q_0[2 * p + N : 2 * p + 2 * N] = (gamma * np.ones(N))

	K = np.zeros((2 * p + 2 * N, 2 * p + 2 * N))
	K[: p, : p] = np.identity(p)
	K[: p, p : 2 * p] = -np.identity(p)
	K[p : 2 * p, : p] = -np.identity(p)
	K[p : 2 * p, p : 2 * p] = np.identity(p)

	q_1 = np.zeros(2 * p + 2 * N)
	q_1[: p] = np.ones(p)
	q_1[p : 2 * p] = np.ones(p)

	S = np.zeros((5 * N + 2 * p, 2 * p + 2 * N))
	S[: N, : p] = Z
	S[: N, p : 2 * p] = -Z
	S[: N, 2 * p : 2 * p + N] = - np.identity(N)
	S[: N, 2 * p + N : 2 * p + 2 * N] = - np.identity(N)

	S[N : 2 * N, : p] = -Z
	S[N : 2 * N, p : 2 * p] = Z
	S[N : 2 * N, 2 * p : 2 * p + N] = - np.identity(N)
	S[N : 2 * N, 2 * p + N : 2 * p + 2 * N] = - np.identity(N)

	S[2 * N : 3 * N, 2 * p : 2 * p + N] = -np.identity(N)
	S[3 * N : 4 * N, 2 * p : 2 * p + N] = np.identity(N)
	S[4 * N : 5 * N, 2 * p + N : 2 * p + 2 * N] = -np.identity(N)
	S[5 * N : 5 * N + p, : p] = -np.identity(p)
	S[5 * N + p : 5 * N + 2 * p, p : 2 * p] = -np.identity(p)

	h_w = np.zeros(5 * N + 2 * p)
	h_w[: N] = Y
	h_w[N : 2 * N] = -Y
	h_w[3 * N : 4 * N] = gamma * np.ones(N)

	u_0 = np.zeros(5 * N + 2 * p)
	u_0[3 * N : 4 * N] = gamma * np.ones(N)

	u_1 = np.zeros(5 * N + 2 * p)

	return P, K, q_0, q_1, S, h_w, u_0, u_1


def training_qp_w(P, K, q_0, q_1, S, h_w, Z, lambda_w, alpha):
	N, p = Z.shape

	Q = (1 / N) * P + lambda_w * (1 - alpha) * K
	f = (1 / N) * q_0 + lambda_w * alpha * q_1

	t = cp.Variable(2 * p + 2 * N)

	constraints = [S @ t <= h_w]

	objective = cp.Minimize(0.5 * cp.quad_form(t, Q) + f.T @ t)
	problem = cp.Problem(objective, constraints)

	try:
		problem.solve(solver = cp.OSQP, eps_abs = 1e-10, eps_rel = 1e-10, max_iter = 100000)

		# print(f'Status training W: {problem.status}')

		w_plus = t.value[ : p]
		w_minus = t.value[p : 2 * p]
		u = t.value[2 * p : 2 * p + N]
		v = t.value[2 * p + N : 2 * p + 2 * N]

		info = {
			'w_plus' : w_plus,
			'w_minus' : w_minus,
			'u' : u,
			'v' : v,
			't_value' : t.value,
			'Q' : Q,
			'f' : f,
			'Problem' : problem
		}

		return info

	except cp.error.SolverError as e:
		print(f"[training_qp] Solver OSQP Error: {e}")

		return None


def training_qp_delta(P, K, q_0, q_1, S, h_d, Z0, lambda_d, alpha):
	nT, p = Z0.shape

	Q = (1 / nT) * P + lambda_d * (1 - alpha) * K
	f = (1 / nT) * q_0 + lambda_d * alpha * q_1

	t = cp.Variable(2 * p + 2 * nT)

	constraints = [S @ t <= h_d]

	objective = cp.Minimize(0.5 * cp.quad_form(t, Q) + f.T @ t)
	problem = cp.Problem(objective, constraints)

	try:
		problem.solve(solver = cp.OSQP, eps_abs = 1e-10, eps_rel = 1e-10, max_iter = 100000)

		# print(f'Status training DELTA: {problem.status}')

		d_plus = t.value[ : p]
		d_minus = t.value[p : 2 * p]
		u = t.value[2 * p : 2 * p + nT]
		v = t.value[2 * p + nT : 2 * p + 2 * nT]

		info = {
			'd_plus' : d_plus,
			'd_minus' : d_minus,
			'u' : u,
			'v' : v,
			't_value' : t.value,
			'Q' : Q,
			'f' : f,
			'Problem' : problem
		}
		return info

	except cp.error.SolverError as e:
		print(f"[training_qp] Solver OSQP Error: {e}")

		return None


def check_KKT(Q, f, S, h, t, v):
	KKT = True

	# STATIONARY
	Stationary = Q @ t + f + S.T @ v        # (N,)
	for s in Stationary:
		if abs(s) >= 1e-5:
			print(f'Unsatisfy Stationary: {s}')
			KKT = False
			return KKT

	# PRIMAL FEASIBILITY
	Primal_Feas = S @ t - h         # (2 * N + 2 * p,)
	for p in Primal_Feas:
		if p >= 1e-5:
			print(f'Unsatisfy Primal Feasibility: {p}')
			KKT = False
			return KKT

	# DUAL FEASIBILITY
	Dual_Feas = v               # (2 * N + 2 * p,)
	for d in Dual_Feas:
		if d < -1e-5:
			print(f'Unsatisfy Dual Feasibility: {d}')
			KKT = False
			return KKT

	# COMPLEMENTARY SLACKNESS
	Comp_Slack = Dual_Feas * Primal_Feas        # (2 * N + 2 * p,)
	for c in Comp_Slack:
		if abs(c) >= 1e-5:
			print(f'Unsatisfy Complementary Slackness: {c}')
			KKT = False
			return KKT

	return KKT


def construct_Y0_mask(nT, N):
	Y0_mask = np.zeros((nT, N))
	Y0_mask[:, N - nT: ] = np.eye(nT)

	return Y0_mask


def construct_Active_lagrane(u_qp, S):
	A = []
	Ac = []

	for i in range(len(u_qp)):
		u = abs(u_qp[i])
		if u >= 1e-12:
			A.append(i)
		else:
			Ac.append(i)

	S_A = S[A]
	S_Ac = S[Ac]

	u_A = u_qp[A]
	u_Ac = u_qp[Ac]

	return A, Ac, u_A, u_Ac, S_A, S_Ac


def construct_test_statistic(j_Outlier, Z, Z0, Y, outliers_obs):
	N, p = Z.shape
	nT = Z0.shape[0]

	# construct eta
	ej = np.zeros((nT, 1))
	ej[j_Outlier][0] = 1
	zj = Z0[j_Outlier].reshape((p, 1))

	I_minusOobs = np.zeros((nT, nT))

	for i in range(nT):
		if i not in outliers_obs:
			I_minusOobs[i][i] = 1

	X_minusOobs = np.dot(I_minusOobs, Z0)

	pinv = np.linalg.pinv(X_minusOobs)
	eta = (np.identity(nT) - np.dot(np.dot(zj.T, pinv), I_minusOobs))

	etaj = np.zeros((N, 1))
	etaj[N - nT : , :] = eta.T @ ej

	etaT_yobs = np.dot((etaj.T), Y)[0]

	return etaj, etaT_yobs


def calculate_a_b(etaj, Y, Sigma):
	N = Y.shape[0]

	e1 = etaj.T @ Sigma @ etaj
	a = (Sigma @ etaj) / e1

	a = a.reshape(-1, 1)

	e2 = np.identity(N) - a @ etaj.T
	b = e2 @ Y

	return a, b.reshape(-1, 1), e1[0][0]


def clean_Matrix_c_d(A, B):
	A_clean = []
	b_clean = []

	for a in A:
		if abs(a) >= 1e-5:
			A_clean.append(a)
		else:
			A_clean.append(0)

	for b in B:
		if abs(b) >= 1e-5:
			b_clean.append(b)
		else:
			b_clean.append(0)

	A_clean = np.array(A_clean, dtype = np.float64)
	b_clean = np.array(b_clean, dtype = np.float64)

	return A_clean, b_clean


def compute_quotient(numerator, denominator):
	if abs(denominator) <= 1e-15:
		denominator = 0

	if abs(numerator) <= 1e-15:
		numerator = 0

	if denominator == 0:
		return np.inf

	quotient = numerator / denominator

	if quotient <= 0:
		return np.inf

	return quotient


def compute_interval(Q_z, f_z, S_Az, S_Azc, h_z, u0_z, u1_z, Z, t_z, v_Az, Az, Azc):
	N, p = Z.shape

	u0_A = u0_z[Az]
	u1_A = u1_z[Az]
	h_Ac = h_z[Azc]

	dim_Q = Q_z.shape[0]
	dim_mat = dim_Q + len(v_Az)

	Matrix = np.zeros((dim_mat, dim_mat))
	Matrix[: dim_Q, : dim_Q] = Q_z
	Matrix[: dim_Q, dim_Q : dim_mat] = S_Az.T
	Matrix[dim_Q : dim_mat, : dim_Q] = S_Az

	Mat_inv = np.linalg.inv(Matrix)

	product_matrix = np.zeros(dim_mat)
	product_matrix[: dim_Q] = -f_z
	product_matrix[dim_Q : dim_mat] = u0_A

	product_matrix_z = np.zeros(dim_mat)
	product_matrix_z[dim_Q : dim_mat] = u1_A

	A_t_nu = Mat_inv @ product_matrix_z
	b_t_nu = Mat_inv @ product_matrix

	A_p = A_t_nu[: p]
	A_m = A_t_nu[p : 2 * p]
	A_mat = A_p - A_m

	b_p = b_t_nu[: p]
	b_m = b_t_nu[p : 2 * p]
	b_mat = b_p - b_m

	psi = A_t_nu[: dim_Q]
	gamma = A_t_nu[dim_Q : dim_mat]

	t_1 = np.inf
	t_2 = np.inf

	num = S_Azc @ t_z - h_Ac
	dem = S_Azc @ psi

	for j in range(len(Azc)):
		numerator = - num[j]
		denominator = dem[j]

		quotient = compute_quotient(numerator, denominator)

		if quotient < t_1:
			t_1 = quotient

	for j in range(len(Az)):
		numerator = - v_Az[j]
		denominator = gamma[j]

		quotient = compute_quotient(numerator, denominator)

		if quotient < t_2:
			t_2 = quotient

	return t_1, t_2, A_mat, b_mat


def merge_intervals(intervals):
	intervals_sorted = sorted(intervals, key = lambda x : x[0])
	merged = []

	for l, r in intervals_sorted:
		if r <= l:
			continue

		if not merged or l > merged[-1][1]:
			merged.append([l, r])
		else:
			merged[-1][1] = max(merged[-1][1], r)

	result = []
	for a, b in merged:
		result.append((a, b))

	return result


def intersect_two_lists(A, B):
	if not A or not B:
		return []

	result = []
	i = 0
	j = 0

	while i < len(A) and j < len(B):
		lo = max(A[i][0], B[j][0])
		hi = min(A[i][1], B[j][1])

		if hi > lo:
			result.append((lo, hi))

		if A[i][1] < B[j][1]:
			i += 1
		else:
			j += 1

	return merge_intervals(result)


def intersect_many(lists, base):
	current = [base]

	for lst in lists:
		current = intersect_two_lists(current, lst)
		if not current:
			break

	return current


def subtract_intervals(base, subs):
	bl, br = base

	if bl > br:
		return []

	subs_merged = merge_intervals(subs)

	result = []
	left = bl

	for l, r in subs_merged:
		if r <= bl or l >= br:
			continue

		if l > left:
			result.append((left, l))

		left = r

	if left < br:
		result.append((left, br))

	return result


def compute_V_t_i(t_idx, i_idx, list_zk, xi, f_matrix, g_matrix):
	z_prev = float(list_zk[t_idx])
	z_curr = float(list_zk[t_idx + 1])

	f_ti = float(f_matrix[i_idx])
	g_ti = float(g_matrix[i_idx])

	base = (z_prev, z_curr)

	if abs(g_ti) <= 0:
		if abs(f_ti) >= xi:
			return [base]
		else:
			return []

	if g_ti > 0:
		a = (xi - f_ti) / g_ti
		b = (-xi - f_ti) / g_ti
	else:  # g_ti < 0
		a = (-xi - f_ti) / g_ti
		b = (xi - f_ti) / g_ti

	part_left  = (z_prev, min(z_curr, b))
	part_right = (max(z_prev, a), z_curr)

	parts = []

	for l, r in (part_left, part_right):
		L = max(l, z_prev)
		R = min(r, z_curr)

		if R > L:
			parts.append((L, R))

	V_t_i = merge_intervals(parts)

	return V_t_i


def Lemma_3(list_zk, list_f_mat, list_g_mat, xi, Outlier_obs):
	T = len(list_zk) - 1

	n = len(list_f_mat[0])
	all_ct = []

	t = 0
	while t < T:
		base = (list_zk[t], list_zk[t + 1])

		V_list = []
		i_var = 0
		while i_var < n:
			V_t_i = compute_V_t_i(t, i_var, list_zk, xi, list_f_mat[t], list_g_mat[t])
			V_list.append(V_t_i)
			i_var += 1

		# ------------------------ Vti outliers ------------------------
		list_for_A = []
		idx_o = 0
		while idx_o < len(Outlier_obs):
			list_for_A.append(V_list[Outlier_obs[idx_o]])
			idx_o += 1

		# print(f'V List outliers: {list_for_A}')

		A_t = intersect_many(list_for_A, base)
		if not A_t:
			t += 1
			continue

		# print(f'A_t: {A_t}')

		# ------------------------ Complement for inliers ------------------------
		non_out = []
		idx = 0
		while idx < n:
			if idx not in Outlier_obs:
				non_out.append(idx)
			idx += 1

		complements = []
		for i_non in non_out:
			comp = subtract_intervals(base, V_list[i_non])
			complements.append(comp)
			# print(f'V_list i non: {V_list[i_non]}, complement: {comp}')

		has_empty = False

		for comp in complements:
			if not comp:
				has_empty = True
				break

		if has_empty:
			t += 1
			continue

		B_t = intersect_many(complements, base)
		# print(f'B_t: {B_t}')

		if not B_t:
			t += 1
			continue

		# ------------------------ C_t = A_t ∩ B_t ------------------------
		C_t = intersect_two_lists(A_t, B_t)
		for interval in C_t:
			all_ct.append(interval)

		t += 1

		# print()
		# print('---------------------------------------------------------------------------')

	# print(all_ct)

	return all_ct


def triple_parametric(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, zk_threshold, etajT_Yobs, verbose = True):
	N, p = Z.shape
	nT = Z0.shape[0]
	zk = -zk_threshold

	list_zk = [zk]
	list_g_beta = []
	list_l_beta = []
	
	list_zk_OC = []
	list_g_OC = []
	list_l_OC = []

	while zk < zk_threshold:
		w_interval_z = [zk]
		d_interval_z = [zk]
		intervals_wdz = [zk]

		Yz = a_y * zk + b_y
		Yz = Yz.ravel()
		Y0z = Y0_mask @ Yz
		Pw_z, Kw_z, q0w_z, q1w_z, Sw_z, hw_z, u0w_z, u1w_z = Construct_matrix(Z, Yz, gamma)

			# --------------------------------- TRAINING W ---------------------------------
		u0w_z[: N] = b_y.reshape(N)
		u0w_z[N : 2 * N] = -b_y.reshape(N)
		u1w_z[: N] = a_y.reshape(N)
		u1w_z[N : 2 * N] = -a_y.reshape(N)

		info_w = training_qp_w(Pw_z, Kw_z, q0w_z, q1w_z, Sw_z, hw_z, Z, lambda_w, alpha)
		if info_w == None:
			return None, None, None

		w_z = info_w['w_plus'] - info_w['w_minus']
		Q_wz = info_w['Q']
		f_wz = info_w['f']
		t_wz = info_w['t_value']
		prob_w = info_w['Problem']
		v_zw = prob_w.constraints[0].dual_value

			# --------------------------------- CHECKING KKT CONDITION ---------------------------------
		KKT_w = check_KKT(Q_wz, f_wz, Sw_z, hw_z, t_wz, v_zw)
		if not KKT_w:
			print(' | W Unsatisfy KKT Condition')
			return None, None, None
		
			# --------------------------------- TRAINING DELTA ---------------------------------
		Pd_z, Kd_z, q0d_z, q1d_z, Sd_z, hd_z, u0d_z, u1d_z = Construct_matrix(Z0, Y0z, gamma)

		hd_z[: nT] = Y0z - Z0 @ w_z
		hd_z[nT : 2 * nT] = - (Y0z - Z0 @ w_z)
		hd_z[3 * nT : 4 * nT] = gamma * np.ones(nT)

		info_d = training_qp_delta(Pd_z, Kd_z, q0d_z, q1d_z, Sd_z, hd_z, Z0, lambda_d, alpha)
		if info_d == None:
			return None, None, None

		d_z = info_d['d_plus'] - info_d['d_minus']
		Q_dz = info_d['Q']
		f_dz = info_d['f']
		t_dz = info_d['t_value']

		prob_d = info_d['Problem']
		v_dz = prob_d.constraints[0].dual_value

			# CHECKING KKT CONDITION
		KKT_d = check_KKT(Q_dz, f_dz, Sd_z, hd_z, t_dz, v_dz)
		if not KKT_d:
			print(' | DELTA Unsatisfy KKT Condition')
			return None, None, None

		Beta_z = w_z + d_z

			# --------------------------------- Calculating Outliers  ---------------------------------
		res_target_z = Y0z - Z0 @ Beta_z

		outlier_mask_z = np.abs(res_target_z) > xi_threshold
		outlier_z = []
		for i, o in enumerate(outlier_mask_z):
			if o:
				outlier_z.append(i)

			# --------------------------------- INTERVAL FOR W ---------------------------------
		Az, Azc, vw_Az, vw_Azc, S_Az, S_Azc = construct_Active_lagrane(v_zw, Sw_z)
		tr_w_1, tr_w_2, A_w, b_w = compute_interval(Q_wz, f_wz, S_Az, S_Azc, hw_z, u0w_z, u1w_z, Z, t_wz, vw_Az, Az, Azc)

		# A_w_clean, b_w_clean = utils.clean_Matrix_A_b(A_w, b_w)

		tr_w = min(tr_w_1, tr_w_2)

		if zk + tr_w < zk_threshold:
			w_interval_z.append(zk + tr_w)
		else:
			w_interval_z.append(zk_threshold)

			# --------------------------------- INTERVAL FOR DELTA ---------------------------------
		b_yT = b_y[N - nT : ,].reshape(nT)
		a_yT = a_y[N - nT : ,].reshape(nT)

		u0d_z[: nT] = b_yT - Z0 @ b_w
		u0d_z[nT : 2 * nT] = - (b_yT - Z0 @ b_w)
		u1d_z[: nT] = a_yT - Z0 @ A_w
		u1d_z[nT : 2 * nT] = - (a_yT - Z0 @ A_w)

		Oz, Ozc, vd_Oz, vd_Ozc, S_Oz, S_Ozc = construct_Active_lagrane(v_dz, Sd_z)
		tr_d_1, tr_d_2, A_d, b_d = compute_interval(Q_dz, f_dz, S_Oz, S_Ozc, hd_z, u0d_z, u1d_z, Z0, t_dz, vd_Oz, Oz, Ozc)

		# A_d_clean, b_d_clean = utils.clean_Matrix_A_b(A_d, b_d)

		tr_d = min(tr_d_1, tr_d_2)

		if zk + tr_d < zk_threshold:
			d_interval_z.append(zk + tr_d)
		else:
			d_interval_z.append(zk_threshold)

		intervals_wdz.append(min(w_interval_z[1], d_interval_z[1]))

				# --------------------------------- INTERVAL FOR BETA ---------------------------------
		g_beta = a_y.reshape(N) - Z @ (A_w + A_d)
		l_beta = b_y.reshape(N) - Z @ (b_w + b_d)
		g_beta = g_beta[N - nT :]
		l_beta = l_beta[N - nT :]

		if verbose:
			print(f'zk: {zk}')
			print('---------------------------------')
			print(f'Intervals for w = {w_interval_z} with transition point: {tr_w}')
			print(f'Intervals for d = {d_interval_z} with transition point: {tr_d}')
			print(f'Final inetvals d and w: {intervals_wdz}')
			print(f'Outlier_z: {outlier_z}, check obs: {np.array_equal(outliers_obs, outlier_z)}')
			print('---------------------------------')
			print()

		if zk < etajT_Yobs < intervals_wdz[1]:
			list_zk_OC.append(zk)
			list_zk_OC.append(intervals_wdz[1] + 1e-5)
			list_g_OC.append(g_beta)
			list_l_OC.append(l_beta)

		zk = intervals_wdz[1] + 1e-3
		# list_zk.append(intervals_wdz)
		list_zk.append(zk)
		list_g_beta.append(g_beta)
		list_l_beta.append(l_beta)

	return list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC


def compute_pivot(intervals_lemma_w, etaj, etaT_yobs, Cov_matrix, tn_mu = 0):
	tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, Cov_matrix), etaj))[0][0]

		# ---------------------------------------- Intervals Lemma 3 ----------------------------------------
	intervals_lemma = []

	for inter in intervals_lemma_w:
		intervals_lemma.append([inter[0], inter[1]])

	new_interval_lemma = []

	for each_interval in intervals_lemma:
		if len(new_interval_lemma) == 0:
			new_interval_lemma.append(each_interval)
		else:
			sub = each_interval[0] - new_interval_lemma[-1][1]
			if abs(sub) < 0.01:
				new_interval_lemma[-1][1] = each_interval[1]
			else:
				new_interval_lemma.append(each_interval)

	intervals_lemma = new_interval_lemma
	print(f'New Interval Lemma: {intervals_lemma}')

	numerator = 0
	denominator = 0

	for each_interval in intervals_lemma:
		al = each_interval[0]
		ar = each_interval[1]

		denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

		if etaT_yobs >= ar:
			numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
		elif (etaT_yobs >= al) and (etaT_yobs < ar):
			numerator = numerator + mp.ncdf((etaT_yobs - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

	if denominator != 0:
		return float(numerator/denominator)

	else:
		return None


def plot_2_hist(list_pivot, list_p_value):
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 hàng, 2 cột

	# Subplot 1: Pivot
	axes[0].hist(list_pivot, bins = 10, edgecolor = 'black', color = 'black', alpha = 0.68)
	axes[0].axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2, label = 'alpha = 0.05')
	axes[0].set_title("Pivot")
	axes[0].set_xlabel("Pivot")
	axes[0].set_ylabel("Frequency")
	axes[0].legend()

	# Subplot 2: P_value
	axes[1].hist(list_p_value, bins = 10, edgecolor = 'black', color = 'black', alpha = 0.68)
	axes[1].axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2, label = 'alpha = 0.05')
	axes[1].set_title("P-value")
	axes[1].set_xlabel("P-value")
	axes[1].set_ylabel("Frequency")
	axes[1].legend()

	plt.tight_layout()
	plt.show()


def plot_2_ecdf(list_pivot, list_p_value_2, label1 = "Pivot Huber", label2 = "P_value"):
	plt.rcParams.update({'font.size': 18})
	grid = np.linspace(0, 1, 101)

	fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 hàng, 2 cột

	# --- Subplot 1 ---
	ecdf1 = sm.distributions.ECDF(np.array(list_pivot))
	axes[0].plot(grid, ecdf1(grid), color = 'cyan', linestyle = '-', linewidth = 3, label = label1)
	axes[0].plot([0, 1], [0, 1], 'k--')
	axes[0].set_title(label1)
	axes[0].legend()

	# --- Subplot 2 ---
	ecdf2 = sm.distributions.ECDF(np.array(list_p_value_2))
	axes[1].plot(grid, ecdf2(grid), color = 'magenta', linestyle='-', linewidth = 3, label = label2)
	axes[1].plot([0, 1], [0, 1], 'k--')
	axes[1].set_title(label2)
	axes[1].legend()

	plt.tight_layout()
	plt.show()
	

def naive_p_value(T, eta, tn_mu):
	val = float(eta.T @ eta)
	scale = mp.sqrt(mp.mpf(val))
	z = (T - tn_mu) / scale
	cdf = 0.5 * (1 + mp.erf(mp.mpf(z) / mp.sqrt(2)))

	P_value_naive = 2 * min(cdf, 1 - cdf)

	return P_value_naive


def bonferroni(naive_p, n):
	P_value_bon = naive_p * (2 ** n)
	
	return min(P_value_bon, 1)


def generate_real_data(file_path, seed, nS = 300, nT = 50):
	df = pd.read_csv(file_path, header=None)

	X_df = df.iloc[:, :280]
	y_df = df.iloc[:, 280]

	median_len = X_df.iloc[:, 61].median()
	source_df = X_df[X_df.iloc[:, 61] < median_len]
	target_df = X_df[X_df.iloc[:, 61] >= median_len]

	y_source = y_df.loc[source_df.index]
	y_target = y_df.loc[target_df.index]

	source_sample = source_df.sample(n = nS, random_state = seed)
	target_sample = target_df.sample(n = nT, random_state = seed)

	y_source_sample = y_source.loc[source_sample.index]
	y_target_sample = y_target.loc[target_sample.index]

	X_S = source_sample.to_numpy()
	X_T = target_sample.to_numpy()
	Y_S = y_source_sample.to_numpy()
	Y_T = y_target_sample.to_numpy()

	X_full = np.vstack([X_S, X_T])
	Y_full = np.concatenate([Y_S, Y_T])

	mean = X_full.mean(axis = 0)
	std = X_full.std(axis = 0)

	std[std == 0] = 1.0

	X_full = (X_full - mean) / std
	X_T = (X_T - mean) / std

	Sigma = np.cov(X_full.T, rowvar = False)

	return X_full, X_T, Y_full, Y_T, Sigma


def segment_worker(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, etajT_Yobs, z_start, z_end):
	pid = os.getpid()
	proc_name = multiprocessing.current_process().name
	print(f"Segment [{z_start:.2f} -> {z_end:.2f}] đang chạy trên Process ID: {pid}")

	os.environ["OMP_NUM_THREADS"] = "1"
	os.environ["MKL_NUM_THREADS"] = "1"

	N, p = Z.shape
	nT = Z0.shape[0]
	zk = z_start

	list_zk = []
	list_g_beta = []
	list_l_beta = []
	
	list_zk_OC = []
	list_g_OC = []
	list_l_OC = []

	while zk < z_end:
		w_interval_z = [zk]
		d_interval_z = [zk]
		intervals_wdz = [zk]

		Yz = a_y * zk + b_y
		Yz = Yz.ravel()
		Y0z = Y0_mask @ Yz
		Pw_z, Kw_z, q0w_z, q1w_z, Sw_z, hw_z, u0w_z, u1w_z = Construct_matrix(Z, Yz, gamma)

			# --------------------------------- TRAINING W ---------------------------------
		u0w_z[: N] = b_y.reshape(N)
		u0w_z[N : 2 * N] = -b_y.reshape(N)
		u1w_z[: N] = a_y.reshape(N)
		u1w_z[N : 2 * N] = -a_y.reshape(N)

		info_w = training_qp_w(Pw_z, Kw_z, q0w_z, q1w_z, Sw_z, hw_z, Z, lambda_w, alpha)
		if info_w == None:
			return None, None, None, None, None, None

		w_z = info_w['w_plus'] - info_w['w_minus']
		Q_wz = info_w['Q']
		f_wz = info_w['f']
		t_wz = info_w['t_value']
		prob_w = info_w['Problem']
		v_zw = prob_w.constraints[0].dual_value

			# --------------------------------- CHECKING KKT CONDITION ---------------------------------
		KKT_w = check_KKT(Q_wz, f_wz, Sw_z, hw_z, t_wz, v_zw)
		if not KKT_w:
			print(' | W Unsatisfy KKT Condition')
			return None, None, None, None, None, None
		
			# --------------------------------- TRAINING DELTA ---------------------------------
		Pd_z, Kd_z, q0d_z, q1d_z, Sd_z, hd_z, u0d_z, u1d_z = Construct_matrix(Z0, Y0z, gamma)

		hd_z[: nT] = Y0z - Z0 @ w_z
		hd_z[nT : 2 * nT] = - (Y0z - Z0 @ w_z)
		hd_z[3 * nT : 4 * nT] = gamma * np.ones(nT)

		info_d = training_qp_delta(Pd_z, Kd_z, q0d_z, q1d_z, Sd_z, hd_z, Z0, lambda_d, alpha)
		if info_d == None:
			return None, None, None, None, None, None

		d_z = info_d['d_plus'] - info_d['d_minus']
		Q_dz = info_d['Q']
		f_dz = info_d['f']
		t_dz = info_d['t_value']

		prob_d = info_d['Problem']
		v_dz = prob_d.constraints[0].dual_value

			# CHECKING KKT CONDITION
		KKT_d = check_KKT(Q_dz, f_dz, Sd_z, hd_z, t_dz, v_dz)
		if not KKT_d:
			print(' | DELTA Unsatisfy KKT Condition')
			return None, None, None, None, None, None

		Beta_z = w_z + d_z

			# --------------------------------- Calculating Outliers  ---------------------------------
		res_target_z = Y0z - Z0 @ Beta_z

		outlier_mask_z = np.abs(res_target_z) > xi_threshold
		outlier_z = []
		for i, o in enumerate(outlier_mask_z):
			if o:
				outlier_z.append(i)

			# --------------------------------- INTERVAL FOR W ---------------------------------
		Az, Azc, vw_Az, vw_Azc, S_Az, S_Azc = construct_Active_lagrane(v_zw, Sw_z)
		tr_w_1, tr_w_2, A_w, b_w = compute_interval(Q_wz, f_wz, S_Az, S_Azc, hw_z, u0w_z, u1w_z, Z, t_wz, vw_Az, Az, Azc)

		# A_w_clean, b_w_clean = utils.clean_Matrix_A_b(A_w, b_w)

		tr_w = min(tr_w_1, tr_w_2)

		if zk + tr_w < z_end:
			w_interval_z.append(zk + tr_w)
		else:
			w_interval_z.append(z_end)

			# --------------------------------- INTERVAL FOR DELTA ---------------------------------
		b_yT = b_y[N - nT : ,].reshape(nT)
		a_yT = a_y[N - nT : ,].reshape(nT)

		u0d_z[: nT] = b_yT - Z0 @ b_w
		u0d_z[nT : 2 * nT] = - (b_yT - Z0 @ b_w)
		u1d_z[: nT] = a_yT - Z0 @ A_w
		u1d_z[nT : 2 * nT] = - (a_yT - Z0 @ A_w)

		Oz, Ozc, vd_Oz, vd_Ozc, S_Oz, S_Ozc = construct_Active_lagrane(v_dz, Sd_z)
		tr_d_1, tr_d_2, A_d, b_d = compute_interval(Q_dz, f_dz, S_Oz, S_Ozc, hd_z, u0d_z, u1d_z, Z0, t_dz, vd_Oz, Oz, Ozc)

		# A_d_clean, b_d_clean = utils.clean_Matrix_A_b(A_d, b_d)

		tr_d = min(tr_d_1, tr_d_2)

		if zk + tr_d < z_end:
			d_interval_z.append(zk + tr_d)
		else:
			d_interval_z.append(z_end)

		intervals_wdz.append(min(w_interval_z[1], d_interval_z[1]))
		
		current_z_end = intervals_wdz[1]

		# --------------------------------- INTERVAL FOR BETA ---------------------------------
		g_beta = a_y.reshape(N) - Z @ (A_w + A_d)
		l_beta = b_y.reshape(N) - Z @ (b_w + b_d)
		g_beta = g_beta[N - nT :]
		l_beta = l_beta[N - nT :]

		if zk < etajT_Yobs < current_z_end:
			list_zk_OC.append(zk)            
			list_zk_OC.append(current_z_end) 
			
			list_g_OC.append(g_beta)
			list_l_OC.append(l_beta)

		zk = current_z_end + 1e-5
		list_zk.append(zk)
		list_g_beta.append(g_beta)
		list_l_beta.append(l_beta)

	return list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC

# 1. Kết nối cluster (chỉ làm 1 lần ở đầu chương trình)
# address='auto' sẽ nối vào cụm 3 máy của bạn
if not ray.is_initialized():
    ray.init(address = 'auto') 

# 2. Đăng ký Ray làm backend cho Joblib
register_ray()

def divide_and_conquer(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, etajT_Yobs, z_min, z_max, num_segments = None):
    
    # --- BƯỚC 1: CẤU HÌNH RAY CLUSTER ---
    # Kiểm tra xem đã kết nối Ray chưa, nếu chưa thì kết nối
    if not ray.is_initialized():
        print("Đang kết nối tới Ray Cluster...")
        ray.init(address = 'auto')
        
    # Đăng ký Ray làm backend cho Joblib
    register_ray()
    
    # Lấy tổng số core thực tế của cả cụm (Hy vọng là 48)
    total_cores_cluster = int(ray.cluster_resources().get("CPU", 1))
    print(f"Đang chạy trên Cluster với tổng {total_cores_cluster} Cores.")

    # --- BƯỚC 2: TỐI ƯU HÓA SỐ ĐOẠN CHIA (QUAN TRỌNG) ---
    # Nếu không truyền num_segments, tự động tính toán.
    # Chiến thuật: Chia số task gấp 4 lần số core. 
    # Ví dụ: 48 core -> chia thành 192 đoạn nhỏ.
    # Lý do: Để core nào làm nhanh xong trước thì lấy việc làm tiếp (Load Balancing).
    if num_segments is None:
        num_segments = total_cores_cluster * 4
    
    print(f"Đang chia Line Z thành {num_segments} đoạn nhỏ để phân phối...")

    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]

    # --- BƯỚC 3: CHẠY SONG SONG TRÊN CLUSTER ---
    # with joblib.parallel_backend('ray'): Là câu lệnh thần thánh để đẩy việc sang máy khác
    with joblib.parallel_backend('ray'):
        results = joblib.Parallel(n_jobs=-1, verbose=5)( # n_jobs=-1 để bung lụa hết 48 core
            joblib.delayed(segment_worker)(
                Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, 
                alpha, gamma, a_y, b_y, xi_threshold, etajT_Yobs, 
                seg[0], seg[1]
            ) for seg in segments
        )

    # --- BƯỚC 4: GỘP KẾT QUẢ (GIỮ NGUYÊN CODE CŨ) ---
    print("Đã tính xong, đang gộp kết quả...")
    all_list_zk = [z_min]
    all_list_g_beta = []
    all_list_l_beta = []

    all_list_zk_OC = []
    all_list_g_OC = []
    all_list_l_OC = []

    for r in results:
        if r is None or r[0] is None: # Kiểm tra kỹ hơn trường hợp trả về None
            continue

        list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC = r
        
        all_list_zk.extend(list_zk)
        all_list_g_beta.extend(list_g_beta)
        all_list_l_beta.extend(list_l_beta)

        all_list_zk_OC.extend(list_zk_OC)
        all_list_g_OC.extend(list_g_OC)
        all_list_l_OC.extend(list_l_OC)

    return all_list_zk, all_list_g_beta, all_list_l_beta, all_list_zk_OC, all_list_g_OC, all_list_l_OC


def run(test_instances, seed):
	nS = test_instances
	nT = 50
	num_info_aux = 1
	num_uninfo_aux = 0
	p = 400

	rho = 0
	h_parameter = 0
	K = num_info_aux + num_uninfo_aux
	N = nS * K + nT
	num_outliers = 0

	Z, Z0, Y, Y0, true_Y, Sigma = generate_data(p, nS, nT, num_info_aux, num_uninfo_aux, num_outliers, h_parameter, rho)

		# ------------------ Blog Feedback Real Data ------------------
	# file_path = r"Real_Data\Blog_Feedback\blogData_train.csv"
	# Z, Z0, Y, Y0, Sigma, true_outliers = get_Blog_Feedback_data_for_inference(file_path, seed = seed, nS = 150, nT = 50)

		# ------------------ Communities and Crime Real Data ------------------
	# DATA_PATH = 'Real_Data\Communities_and_Crime\communities_clean.csv'
	# NAMES_PATH = 'Real_Data\Communities_and_Crime\communities.names'
	# Z, Z0, Y, Y0, Sigma, true_outliers = get_communities_data_for_inference(data_file_path = DATA_PATH, names_file_path = NAMES_PATH, seed = seed, n_source = 60, n_target = 40, p = 120)
	
		# ------------------ Superconductivity Real Data ------------------
	# TRAIN_PATH = r'Real_Data\Superconductivity_Data\train.csv'
	# UNIQUE_PATH = r'Real_Data\Superconductivity_Data\unique_m.csv'
	# Z, Z0, Y, Y0, Sigma, true_outliers = get_superconductivity_data_for_inference(train_file_path = TRAIN_PATH, unique_m_path = UNIQUE_PATH, seed = seed, n_source = 200, n_target = 50, p = 81)

		# ------------------ RNA-Seq V2 Real Data (Source: BRCA, Target: OV) ------------------
	# GENE_PATH = 'Real_Data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz'
	# PHENO_PATH = 'Real_Data/Survival_SupplementalTable_S1_20171025_xena_sp'
	# Z, Z0, Y, Y0, Sigma, true_outliers = get_tcga_data_for_inference(gene_file_path = GENE_PATH, pheno_file_path = PHENO_PATH, seed = seed, n_source = 100, n_target = 50, p = 250)

		# ------------------ Fred MD Dataset ------------------
	# data_file_path = r"Real_Data\Fred_MD\fred_md_current.csv"
	# Z, Z0, Y, Y0, Sigma, true_outliers = get_fred_md_data_for_inference(data_file_path, seed = seed, n_source = 40, n_target = 35, p = 100)
	
	N, p = Z.shape
	nT = Z0.shape[0]

	gamma = 1.0
	alpha = 0.5
	xi_threshold = 2.8
	lambda_w = np.sqrt(np.log(p) / N) * 9.0
	lambda_d = np.sqrt(np.log(p) / nT) * 4.5
	threshold = 20

	P_w, K_w, q0_w, q1_w, S_w, h_w, u0_w, u1_w = Construct_matrix(Z, Y, gamma)

	info_w = training_qp_w(P_w, K_w, q0_w, q1_w, S_w, h_w, Z, lambda_w, alpha)
	if info_w == None:
		return None, None, None, None, None, None

	w_hat_qp = info_w['w_plus'] - info_w['w_minus']
	Q_w = info_w['Q']
	f_w = info_w['f']
	t_hat_w = info_w['t_value']
	prob_w = info_w['Problem']
	v_lag_w = prob_w.constraints[0].dual_value

			# CHECKING KKT CONDITION
	KKT_w = check_KKT(Q_w, f_w, S_w, h_w, t_hat_w, v_lag_w)
	if not KKT_w:
		print(' | W Unsatisfy KKT Condition')
		return None, None, None, None, None, None

		# --------------------------------------- TRAINING DELTA ---------------------------------------
	P_d, K_d, q0_d, q1_d, S_d, h_d, u0_d, u1_d = Construct_matrix(Z0, Y0, gamma)

	h_d[: nT] = Y0 - Z0 @ w_hat_qp
	h_d[nT : 2 * nT] = - (Y0 - Z0 @ w_hat_qp)
	h_d[3 * nT : 4 * nT] = gamma * np.ones(nT)

	info_d = training_qp_delta(P_d, K_d, q0_d, q1_d, S_d, h_d, Z0, lambda_d, alpha)
	if info_d == None:
		return None, None, None, None, None, None

	d_hat_qp = info_d['d_plus'] - info_d['d_minus']
	Q_d = info_d['Q']
	f_d = info_d['f']
	t_hat_d = info_d['t_value']

	prob_d = info_d['Problem']
	v_lag_d = prob_d.constraints[0].dual_value

		# CHECKING KKT CONDITION
	KKT_d = check_KKT(Q_d, f_d, S_d, h_d, t_hat_d, v_lag_d)
	if not KKT_d:
		print(' | DELTA Unsatisfy KKT Condition')
		return None, None, None, None, None, None

		# --------------------------------------- COMBINING BETA HAT ---------------------------------------
	beta_hat_Or = w_hat_qp + d_hat_qp

	res_target = Y0 - Z0 @ beta_hat_Or

	outlier_mask = np.abs(res_target) >= xi_threshold
	outliers_obs = []
	for i, o in enumerate(outlier_mask):
		if o:
			outliers_obs.append(i)

	if len(outliers_obs) == 0:
		print('No Outliers Obs')
		return None, None, None, None, None, None
	
	# print(f"True outlier: {true_outliers}")
	# print(f"Outlier obs: {outliers_obs}, Len Outlier Obs: {len(outliers_obs)}")

	# for j in outliers_obs:
	# 	print(f"j: {j}")
	# 	Y0_mask = construct_Y0_mask(nT, N)

	# 	etaj, etajT_Yobs = construct_test_statistic(j, Z, Z0, Y, outliers_obs)
	# 	a_y, b_y, eta = calculate_a_b(etaj, Y, Sigma)

	# 	new_threshold = threshold + (np.sqrt(eta))

	# 	if abs(etajT_Yobs) > threshold:
	# 		print(f'etajT_Yobs out range threshold: {abs(etajT_Yobs)} > {new_threshold}')
	# 		continue
	# 	else:
	# 		print(f'etajT_Yobs in range threshold: {etajT_Yobs} <= {new_threshold}')

	# 		# ------------------------------------------------ RoSI - TL ------------------------------------------------
	# 	# list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC = triple_parametric(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, threshold, etajT_Yobs, verbose = False)
	# 	list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC = divide_and_conquer(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, etajT_Yobs, z_min = -new_threshold, z_max = new_threshold, num_segments = 6)

	# 	if list_zk == None or list_zk == []:
	# 		if list_zk == None:
	# 			print('List_zk is NONE')
	# 		else:
	# 			print(f'List_zk len: {len(list_zk)}')

	# 		continue

	# 	intervals_lemma = Lemma_3(list_zk, list_l_beta, list_g_beta, xi_threshold, outliers_obs)

	# 	if intervals_lemma == []:
	# 		print("Interval lemma = []")
	# 		continue

	# 	tn_mu = 0
	# 	pivot = compute_pivot(intervals_lemma, etaj, etajT_Yobs, Sigma, tn_mu)

	# 	if list_zk_OC == None or list_zk_OC == []:
	# 		if list_zk_OC == None:
	# 			print('list_zk_OC is NONE')
	# 		else:
	# 			print(f'list_zk_OC len: {len(list_zk_OC)}')

	# 		continue

	# 	pivot_OC = compute_pivot([list_zk_OC], etaj, etajT_Yobs, Sigma, tn_mu)

	# 	P_value_naive = naive_p_value(etajT_Yobs, etaj, tn_mu)
	# 	P_value_bon = bonferroni(P_value_naive, nT)

	# 	p_value = 2 * min(1 - pivot, pivot)
	# 	P_value_OC = 2 * min(pivot_OC, 1 - pivot_OC)
		
	# 	if P_value_bon > 1:
	# 		P_value_bon = 1

	# 	print(f'P_value_naive: {P_value_naive}, P_value_bon: {P_value_bon}, P_value_RoSI: {p_value}, P_value_RoSI_OC: {P_value_OC}')
	# 	print('--------------------------------------------------------------------------------------------------------')
	# 	print()

	j_Outlier = np.random.choice(outliers_obs)

	# true_outliers = set(range(5))  # {0, 1, 2, 3, 4}
	# intersect = list(set(outliers_obs) & true_outliers)

	# if len(intersect) > 0:
	#     j_Outlier = np.random.choice(intersect)
	# else:
	#     return None, None,None, None, None,None

	T = res_target[j_Outlier]

	Y0_mask = construct_Y0_mask(nT, N)

	etaj, etajT_Yobs = construct_test_statistic(j_Outlier, Z, Z0, Y, outliers_obs)
	a_y, b_y, eta = calculate_a_b(etaj, Y, Sigma)

	threshold = threshold * (np.sqrt(eta))

	# print(f"etaT_Yobs: {etajT_Yobs}, T: {T}")

	if abs(etajT_Yobs) > threshold:
		print(f'etajT_Yobs out range threshold: {abs(etajT_Yobs)} > {threshold}')
		return None, None, None, None, None, None
	else:
		print(f'etajT_Yobs in range threshold: {etajT_Yobs} <= {threshold}')

		# ------------------------------------------------ RoSI - TL ------------------------------------------------
	# list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC = triple_parametric(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, threshold, etajT_Yobs, verbose = False)
	list_zk, list_g_beta, list_l_beta, list_zk_OC, list_g_OC, list_l_OC = divide_and_conquer(Z, Z0, Y0_mask, outliers_obs, lambda_w, lambda_d, alpha, gamma, a_y, b_y, xi_threshold, etajT_Yobs, z_min = -threshold, z_max = threshold, num_segments = 16)

	if list_zk == None or list_zk == []:
		if list_zk == None:
			print('List_zk is NONE')
		else:
			print(f'List_zk len: {len(list_zk)}')

		return None, None, None, None, None, None

	intervals_lemma = Lemma_3(list_zk, list_l_beta, list_g_beta, xi_threshold, outliers_obs)

	if intervals_lemma == []:
		print("Interval lemma = []")
		return None, None, None, None, None, None

	# tn_mu = np.dot(etaj.T, true_Y)[0]
	tn_mu = 0
	# print(f"tn_mu: {tn_mu}")

	pivot = compute_pivot(intervals_lemma, etaj, etajT_Yobs, Sigma, tn_mu)

	if list_zk_OC == None or list_zk_OC == []:
		if list_zk_OC == None:
			print('list_zk_OC is NONE')
		else:
			print(f'list_zk_OC len: {len(list_zk_OC)}')

		return None, None, None, None, None, None
	
	intervals_lemma_OC = Lemma_3(list_zk_OC, list_l_OC, list_g_OC, xi_threshold, outliers_obs)

	if intervals_lemma_OC == []:
		print("Interval lemma OC = []")
		return None, None, None, None, None, None

	pivot_OC = compute_pivot(intervals_lemma_OC, etaj, etajT_Yobs, Sigma, tn_mu)

	P_value_naive = naive_p_value(etajT_Yobs, etaj, tn_mu)
	P_value_bon = bonferroni(P_value_naive, nT)

	# return 0, P_value_naive, P_value_bon, 0
	return pivot, P_value_naive, P_value_bon, pivot_OC, len(list_zk), len(outliers_obs)
	# return 0, P_value_naive, P_value_bon, 0, 0, len(outliers_obs)


def run_synthetic(test_instances):
	# pivot, P_value_naive, P_value_bon, pivot_OC = run(test_instances)
	# p_value = 2 * min(pivot, 1 - pivot)
	# P_value_OC = 2 * min(pivot_OC, 1 - pivot_OC)
	# print(f'P_value_naive: {P_value_naive}, P_value_bon: {P_value_bon}, P_value_RoSI: {p_value}, P_value_RoSI_OC: {P_value_OC}')

	Iteration = 100
	alpha = 0.05
	count_naive, count_bon, count_RoSI, count_OC = 0, 0, 0, 0

	list_len_interval = []
	list_time = []
	list_len_o_obs = []

	iter, seed = 1, 1
	start = time.time()

	while iter <= Iteration:
		np.random.seed(seed)
		print(f'Iter: {iter}, Seed: {seed}')

		start_iter = time.time()
		pivot, P_value_naive, P_value_bon, pivot_OC, intervals, len_outlier_obs = run(test_instances, seed)
		end_iter = time.time()
		
		runtime_iter = ((end_iter - start_iter) / 60)

		if pivot is not None and pivot_OC is not None:
			p_value = 2 * min(1 - pivot, pivot)
			P_value_OC = 2 * min(pivot_OC, 1 - pivot_OC)
			
			if P_value_bon > 1:
				P_value_bon = 1
			
			if p_value <= alpha:
				count_RoSI += 1
			
			if P_value_naive <= alpha:
				count_naive += 1
			
			if P_value_bon <= alpha:
				count_bon += 1
			
			if P_value_OC <= alpha:
				count_OC += 1

			list_len_interval.append(intervals)
			list_time.append(runtime_iter)
			list_len_o_obs.append(len_outlier_obs)
			iter += 1

			print('--------------------------------------------------------------------------------------------------------')
			print(f'P_value_naive: {P_value_naive}, P_value_bon: {P_value_bon}, P_value_RoSI: {p_value}, P_value_RoSI_OC: {P_value_OC}, time (1 iter): {runtime_iter:.6f}')
			print()

		seed += 1

	end = time.time()
	runtime = ((end - start) / 60)
	print(f'Runtime for {Iteration} iterations: {runtime:.6f} minutes')

	FPR_naive = count_naive / Iteration
	FPR_bon = count_bon / Iteration
	FPR_RoSI = count_RoSI / Iteration
	FPR_OC = count_OC / Iteration

	# return list_len_interval, list_time, list_len_o_obs
	return FPR_naive, FPR_bon, FPR_RoSI, FPR_OC, list_len_interval, list_time, list_len_o_obs
	# return FPR_naive, FPR_bon


if __name__ == '__main__':
	list_test = [125, 150, 175, 200]
	for test_instances in list_test:
		print(f"Source Instance: {test_instances}")
		FPR_naive, FPR_bon, FPR_RoSI, FPR_OC, list_len_interval, list_time, list_len_o_obs = run_synthetic(test_instances)

		print(f"FPR_naive: {FPR_naive}, FPR_bon: {FPR_bon}, FPR_RoSI: {FPR_RoSI}, FPR_OC: {FPR_OC}")
		print(f"list_len_interval: {list_len_interval}, list_time: {list_time}, list_len_o_obs: {list_len_o_obs}")
	
	# print(f"Real Data")
	# # run_synthetic(0)
	# FPR_naive, FPR_bon, FPR_RoSI, FPR_OC, list_pvalue_Rosi, list_pvalue_Rosi_OC = run_synthetic(0)

	# print(f"FPR_naive: {FPR_naive}, FPR_bon: {FPR_bon}, FPR_RoSI: {FPR_RoSI}, FPR_OC: {FPR_OC}")
	# print(f"list_pvalue_Rosi: {list_pvalue_Rosi}")

	# print(f"list_pvalue_Rosi_OC: {list_pvalue_Rosi_OC}")
