import numpy as np
import pandas as pd
from random import uniform

def not_scale(f):
	return df[f]

def inverse_not_scale(pred, f):
	return pred

def mean_scale(f):
	return (df[f] - df[f].mean()) / (df[f].max() - df[f].min())

def inverse_mean_scale(pred, f):
	return pred * (df[f].max() - df[f].min()) + df[f].mean()

def standard_scale(f):
	return (df[f] - df[f].mean()) / df[f].std()

def inverse_standard_scale(pred, f):
	return pred * df[f].std() + df[f].mean()

function = standard_scale

inverse_function = inverse_not_scale if function == not_scale else inverse_mean_scale if function == mean_scale else inverse_standard_scale

print(function)

df = pd.DataFrame({
	"meters":[ 67,  74,  93,  96, 102, 114, 127,  182, 193],
	"rooms": [  2,   3,   3,   3,   4,   3,   3,    5,   4],
	"prices":[380, 450, 630, 640, 730, 720, 850, 1020, 980]})

new_df = pd.DataFrame({
	"meters":function("meters"),
	"rooms": function("rooms"),
	"prices":function("prices")
	})

x = np.array([[m, r, 1] for m, r in new_df[["meters", "rooms"]].values])
y = np.array(new_df["prices"]).reshape(-1, 1)
w = np.array([uniform(0,1) for i in range(x.shape[1])]).astype(float).reshape(-1,1)
N = x.shape[0]
lr = 1e-4 if function == not_scale else 1e-0

def cost_and_std():
	y_pred = inverse_function(np.dot(x, w), "prices").flatten()
	print("Coste:", np.sum(np.power(y_pred - df["prices"].to_numpy(), 2)) / (2 * N))
	print("Std:", np.mean(np.absolute(y_pred - df["prices"].to_numpy())))

cost_and_std()

for i in range(1000):
	hx = np.dot(x, w)
	diff = hx - y
	res = np.dot(diff.T[0], x)
	gradient = np.divide(res, N)
	w -= lr * gradient.reshape(-1, 1)

y_pred = np.dot(x, w).flatten()
y_pred = inverse_function(y_pred, "prices")

cost_and_std()

print("w =", w.transpose())
print("Real prices:", df["prices"].to_numpy())
print("Calculated prices:", y_pred.astype(int))