import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def _predict(model, X, V_fl, T_p):
	i = model.theta0.sever_index
	for t in range(T_p):
		X_predict = np.zeros((1, X.shape[1], X.shape[2]))
		V_fl_predict = np.zeros((1, V_fl.shape[1]))

		X_predict[0, 0, i] = predict_T_f(model, X, V_fl)
		X_predict[0, 1, i] = predict_T_f(model, X, V_fl)
		X_predict[0, 2, i] = predict_U(model, X)
		X_predict[0, 3, i] = predict_W(model, X)
		V_fl_predict[0, 0] = predict_V_fl(model, V_fl)

		X_predict[0, 0, i-1] = predict_T_f_down(model, X, V_fl)
		X_predict[0, 1, i-1] = predict_T_b_down(model, X, V_fl)
		X_predict[0, 0, i+1] = predict_T_f_top(model, X, V_fl)
		X_predict[0, 1, i+1] = predict_T_b_top(model, X, V_fl)

		X = np.concatenate([X, X_predict], axis=0)
		V_fl = np.concatenate([V_fl, V_fl_predict], axis=0)

	return X


def predict_T_f(model, X, V_fl):
	i = model.theta0.sever_index
	#T_fi(t+1) = a x T_fi + b1 x U_i x T_bi + b2 x (1-U_i) x T_bi + b3 x V_fl x T_fi-1 + b4 x V_fl x T_fi+1
	return model.theta0.a*X[-1, 0, i] + model.theta0.b1*X[-1, 2, i]*X[-1, 1, i] + model.theta0.b2*(1-X[-1, 2, i])*X[-1, 1, i] + model.theta0.b3*V_fl[-1, 0]*X[-1, 0, i-1] + model.theta0.b4*V_fl[-1, 0]*X[-1, 0, i+1]

def predict_T_b(model, X, V_fl):
	#T_bi(t+1) = f1 x T_bi + f2 x T_fi + f3 x U_i x W_i + f4 x T_bi-1
	i = model.theta0.sever_index
	return model.theta0.f1*X[-1, 1, i] + model.theta0.f2*X[-1, 0, i] + model.theta0.f3*X[-1, 2, i]*X[-1, 3, i] + model.theta0.f4*X[-1, 1, i-1]

def predict_V_fl(model, V_fl):
	#V_fl(t+1) = n + n0 x V_fl(t) + n1 x V_fl(t-1)
	pred=0
	for i, coeff in enumerate(model.ar_V_fl):
		if i==0:
			pred += coeff
		else:
			pred += coeff*V_fl[-i][0]
	return pred

def predict_U(model, X):
	#U(t+1) = U(t)
	i = model.theta0.sever_index
	return X[-1, 2, i]

def predict_W(model, X):
	#W(t+1) = W(t)
	i = model.theta0.sever_index
	return X[-1, 3, i]

def predict_T_f_down(model, X, V_fl):
	i = model.theta0.sever_index - 1
	#T_fi(t+1) = a x T_fi + b1 x U_i x T_bi + b2 x (1-U_i) x T_bi + b4 x V_fl x T_fi+1
	return model.theta_down.a*X[-1, 0, i] + model.theta_down.b1*X[-1, 2, i]*X[-1, 1, i] + model.theta_down.b2*(1-X[-1, 2, i])*X[-1, 1, i] + model.theta_down.b4*V_fl[-1, 0]*X[-1, 0, i+1]

def predict_T_b_down(model, X, V_fl):
	#T_bi(t+1) = f1 x T_bi + f2 x T_fi + f3 x U_i x W_i
	i = model.theta0.sever_index - 1
	return model.theta_down.f1*X[-1, 1, i] + model.theta_down.f2*X[-1, 0, i] + model.theta_down.f3*X[-1, 2, i]*X[-1, 3, i]

def predict_T_f_top(model, X, V_fl):
	i = model.theta0.sever_index + 1
	#T_fi(t+1) = a x T_fi + b1 x U_i x T_bi + b2 x (1-U_i) x T_bi + b3 x V_fl x T_fi-1
	return model.theta_top.a*X[-1, 0, i] + model.theta_top.b1*X[-1, 2, i]*X[-1, 1, i] + model.theta_top.b2*(1-X[-1, 2, i])*X[-1, 1, i] + model.theta_top.b3*V_fl[-1, 0]*X[-1, 0, i-1]

def predict_T_b_top(model, X, V_fl):
	#T_bi(t+1) = f1 x T_bi + f2 x T_fi + f3 x U_i x W_i + f4 x T_bi-1
	i = model.theta0.sever_index + 1
	return model.theta_top.f1*X[-1, 1, i] + model.theta_top.f2*X[-1, 0, i] + model.theta_top.f3*X[-1, 2, i]*X[-1, 3, i] + model.theta_top.f4*X[-1, 1, i-1]

#uncompleted yet, need to write ar model by myown
def predict_T_bottom(X, V_fl, model):
	i = model.theta0.sever_index - 1


	ar_X = np.concatenate([X[:, 0, 0].reshape(-1), V_fl.reshape(-1)])
	ar_param = AR(ar_X.reshape(-1), p=4)

	pred=0
	for i, coeff in enumerate(ar_param):
		if i==0:
			pred += coeff
		elif i==len(ar_param)-1:
			pred += coeff*V_fl[-1, 0]
		else:
			pred += coeff*X[-i, 0, 0]

	return pred

def AR(X, p=2):
	ar_model = AutoReg(X, lags=p).fit()
	return ar_model.params
