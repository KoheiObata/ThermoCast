import numpy as np

#forgetting factor
FF=0.1


def _update_parameter(i, X, V_fl, model):
	model.a = update_param(i, X, V_fl, model, 'a')
	model.b1 = update_param(i, X, V_fl, model, 'b1')
	model.b2 = update_param(i, X, V_fl, model, 'b2')
	model.b3 = 0 if model.position==-1 else update_param(i, X, V_fl, model, 'b3')
	model.b4 = 0 if model.position==1 else update_param(i, X, V_fl, model, 'b4')
	model.f1 = update_param(i, X, V_fl, model, 'f1')
	model.f2 = update_param(i, X, V_fl, model, 'f2')
	model.f3 = update_param(i, X, V_fl, model, 'f3')
	model.f4 = 0 if model.position==-1 else update_param(i, X, V_fl, model, 'f4')

def update_param(i, X, V_fl, model, param):
	T_w = len(X)
	denominator, numerator = 0, 0
	for t in range(1,T_w):
		coeff, other = other_param(t-1, i, X, V_fl, model, param)
		denominator += np.exp(FF*t)*coeff
		numerator += np.exp(FF*t)*coeff*other
	return numerator/denominator


def get_a(t, i, X, model):
	#a x T_fi(t)
	coeff = X[t, 0, i]
	return coeff, model.a*coeff

def get_b1(t, i, X, model):
	#b1 x U_i(t)
	coeff = X[t, 2, i]
	return coeff, model.b1*coeff

def get_b2(t, i, X, model):
	#b2 x (1 - U_i(t)) x T_bi(t)
	coeff = (1-X[t, 2, i])*X[t, 1, i]
	return coeff, model.b2*coeff

def get_b3(t, i, X, V_fl, model):
	#b3 x V_fl x T_fi-1(t)
	coeff = V_fl[t, 0]*X[t, 0, i-1]
	return coeff, model.b3*coeff

def get_b4(t, i, X, V_fl, model):
	#b4 x V_fl x T_fi+1(t)
	coeff = V_fl[t, 0]*X[t, 0, i+1]
	return coeff, model.b4*coeff

def get_f1(t, i, X, model):
	#f1 x T_bi(t)
	coeff = X[t, 1, i]
	return coeff, model.f1*coeff

def get_f2(t, i, X, model):
	#f2 x T_fi(t)
	coeff = X[t, 0, i]
	return coeff, model.f2*coeff

def get_f3(t, i, X, model):
	#f3 x U_i(t) x W_i(t)
	coeff = X[t, 2, i] * X[t, 3, i]
	return coeff, model.f3*coeff

def get_f4(t, i, X, model):
	#f4 x T_bi-1(t)
	coeff = X[t, 1, i-1]
	return coeff, model.f4*coeff


def other_param(t, i, X, V_fl, model, param):
	#front temprature
	if param in ['a', 'b1', 'b2', 'b3', 'b4']:
		a_coeff, a_x = get_a(t, i, X, model)
		b1_coeff, b1_x = get_b1(t, i, X, model)
		b2_coeff, b2_x = get_b2(t, i, X, model)
		b3_coeff, b3_x = (0,0) if model.position==-1 else get_b3(t, i, X, V_fl, model)
		b4_coeff, b4_x = (0,0) if model.position==1 else get_b4(t, i, X, V_fl, model)

		if param=='a':
			return a_coeff, X[t+1, 0, i] - b1_x - b2_x - b3_x - b4_x
		if param=='b1':
			return b1_coeff, X[t+1, 0, i] - a_x - b2_x - b3_x - b4_x
		if param=='b2':
			return b2_coeff, X[t+1, 0, i] - a_x - b1_x - b3_x - b4_x
		if param=='b3':
			return b3_coeff, X[t+1, 0, i] - a_x - b1_x - b2_x - b4_x
		if param=='b4':
			return b4_coeff, X[t+1, 0, i] - a_x - b1_x - b2_x - b3_x

	#back temprature
	if param in ['f1', 'f2', 'f3', 'f4']:
		f1_coeff, f1_x = get_f1(t, i, X, model)
		f2_coeff, f2_x = get_f2(t, i, X, model)
		f3_coeff, f3_x = get_f3(t, i, X, model)
		f4_coeff, f4_x = (0,0) if model.position==-1 else get_f4(t, i, X, model)

		if param=='f1':
			return f1_coeff, X[t+1, 1, i] - f2_x - f3_x - f4_x
		if param=='f2':
			return f2_coeff, X[t+1, 1, i] - f1_x - f3_x - f4_x
		if param=='f3':
			return f3_coeff, X[t+1, 1, i] - f1_x - f2_x - f4_x
		if param=='f4':
			return f4_coeff, X[t+1, 1, i] - f1_x - f2_x - f3_x
