import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Functions
def binary_call_option_price(S,K,r,sigma,delta_t):
    d_1 = (np.log(S/K)+(r+sigma**2/2)*delta_t)/(sigma*np.sqrt(delta_t))
    d_2 = d_1 - sigma*np.sqrt(delta_t)
    C = np.exp(-r*delta_t)*norm.cdf(d_2)
    return C

def binary_put_option_price(S,K,r,sigma,delta_t):
    d_1 = (np.log(S/K)+(r+sigma**2/2)*delta_t)/(sigma*np.sqrt(delta_t))
    d_2 = d_1 - sigma*np.sqrt(delta_t)
    P = np.exp(-r*delta_t)*norm.cdf(-d_2)
    return P

def binary_call_option_payoff(S,K,X=1):
    diff = S - K
    payoff = np.where(diff>0,1,0)
    return payoff

def binary_put_option_payoff(S,K,X=1):
    diff = S - K
    payoff = np.where(diff<0,1,0)
    return payoff

def binary_call_option_delta_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_call_option_price(S+change,K,r,sigma,delta_t) - binary_call_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_put_option_delta_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_put_option_price(S+change,K,r,sigma,delta_t) - binary_put_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_call_option_gamma_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_call_option_delta_approx(S+change,K,r,delta_t,sigma,change) - binary_call_option_delta_approx(S,K,r,delta_t,sigma,change))/change
    return approx_slope

def binary_put_option_gamma_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_put_option_delta_approx(S+change,K,r,delta_t,sigma,change) - binary_put_option_delta_approx(S,K,r,delta_t,sigma,change))/change
    return approx_slope

def binary_call_option_vega_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_call_option_price(S,K,r,sigma+change,delta_t) - binary_call_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_put_option_vega_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_put_option_price(S,K,r,sigma+change,delta_t) - binary_put_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_call_option_theta_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_call_option_price(S+change,K,r,sigma,delta_t+change) - binary_call_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_put_option_theta_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_put_option_price(S+change,K,r,sigma,delta_t+change) - binary_put_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_call_option_rho_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_call_option_price(S,K,r+change,sigma,delta_t) - binary_call_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

def binary_put_option_rho_approx(S,K,r,delta_t,sigma,change):
    approx_slope = (binary_put_option_price(S,K,r+change,sigma,delta_t) - binary_put_option_price(S,K,r,sigma,delta_t))/change
    return approx_slope

# a. [1 point] Draw a payoff diagram (at maturity) of the binary call and of the binary put
S_vector = np.arange(0,200,0.1)
K_vector = np.ones(2000)*100
plt.plot(S_vector,binary_call_option_payoff(S_vector,K_vector), color = 'g', label='Binary Call Option')
plt.plot(S_vector,binary_put_option_payoff(S_vector,K_vector), color = 'r', label='Binary Put Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.axhline(y = 1, color = 'k', linestyle = ':')
plt.title('Payoff diagram (at maturity) of the binary call and binary put')
plt.xlabel('Stock price')
plt.ylabel('Payoff')
plt.legend(loc='center right')
plt.savefig('payoff_diagram.png')
plt.clf()

# b. [1 point] Compute the value of the binary call and of the binary put
C = binary_call_option_price(100,100,0.025,0.2,0.25)
P = binary_put_option_price(100,100,0.025,0.2,0.25)

print(f'Value of the binary call: {C}')
print(f'Value of the binary put: {P}')

# c. [2 points] Compute the greeks delta, gamma, vega, theta and rho for the binary call and for the binary put. (Approximate these)
delta_call = binary_call_option_delta_approx(100,100,0.025,0.25,0.2,0.0001)
delta_put = binary_put_option_delta_approx(100,100,0.025,0.25,0.2,0.0001)
print(f'Delta Binary Call Option: {delta_call}')
print(f'Delta Binary Put Option: {delta_put}')

gamma_call = binary_call_option_gamma_approx(100,100,0.025,0.25,0.2,0.0001)
gamma_put = binary_put_option_gamma_approx(100,100,0.025,0.25,0.2,0.0001)
print(f'Gamma Binary Call Option: {gamma_call}')
print(f'Gamma Binary Put Option: {gamma_put}')

vega_call = binary_call_option_vega_approx(100,100,0.025,0.25,0.2,0.0001)
vega_put = binary_put_option_vega_approx(100,100,0.025,0.25,0.2,0.0001)
print(f'Vega Binary Call Option: {vega_call}')
print(f'Vega Binary Put Option: {vega_put}')

theta_call = binary_call_option_theta_approx(100,100,0.025,0.25,0.2,0.0001)
theta_put = binary_put_option_theta_approx(100,100,0.025,0.25,0.2,0.0001)
print(f'Theta Binary Call Option: {theta_call}')
print(f'Theta Binary Put Option: {theta_put}')

rho_call = binary_call_option_rho_approx(100,100,0.025,0.25,0.2,0.0001)
rho_put = binary_put_option_rho_approx(100,100,0.025,0.25,0.2,0.0001)
print(f'Rho Binary Call Option: {rho_call}')
print(f'Rho Binary Put Option: {rho_put}')

# e. [2 points] Create a grid of Stock prices. Compute each of the five greeks {delta, gamma, vega, rho, theta} at these gridpoints.
S_vector = np.arange(5,200,1)
K_vector = np.ones(len(S_vector))*100
plt.plot(S_vector,binary_call_option_delta_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'g', label='Binary Call Option')
plt.plot(S_vector,binary_put_option_delta_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'r', label='Binary Put Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.title('Approximation of delta of the binary call and binary put')
plt.xlabel('Stock price')
plt.ylabel('Delta')
plt.legend(loc='lower right')
plt.savefig('delta.png')
plt.clf()

S_vector = np.arange(5,200,1)
K_vector = np.ones(len(S_vector))*100
plt.plot(S_vector,binary_call_option_gamma_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'g', label='Binary Call Option')
plt.plot(S_vector,binary_put_option_gamma_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'r', label='Binary Put Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.title('Approximation of gamma of the binary call and binary put')
plt.xlabel('Stock price')
plt.ylabel('Gamma')
plt.legend(loc='lower right')
plt.savefig('gamma.png')
plt.clf()

S_vector = np.arange(5,200,1)
K_vector = np.ones(len(S_vector))*100
plt.plot(S_vector,binary_call_option_vega_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'g', label='Binary Call Option')
plt.plot(S_vector,binary_put_option_vega_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'r', label='Binary Put Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.title('Approximation of vega of the binary call and binary put')
plt.xlabel('Stock price')
plt.ylabel('Vega')
plt.legend(loc='lower right')
plt.savefig('vega.png')
plt.clf()

S_vector = np.arange(5,200,1)
K_vector = np.ones(len(S_vector))*100
plt.plot(S_vector,binary_call_option_rho_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'g', label='Binary Call Option')
plt.plot(S_vector,binary_put_option_rho_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'r', label='Binary Put Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.axhline(y = -0.25, color = 'k', linestyle = ':')
plt.title('Approximation of rho of the binary call and binary put')
plt.xlabel('Stock price')
plt.ylabel('Rho')
plt.legend(loc='lower right')
plt.savefig('rho.png')
plt.clf()

S_vector = np.arange(5,200,1)
K_vector = np.ones(len(S_vector))*100
plt.plot(S_vector,binary_call_option_theta_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'g', label='Binary Call Option')
plt.plot(S_vector,binary_put_option_theta_approx(S_vector, K_vector, 0.025,0.25,0.2,0.0001), color = 'r', label='Binary Put Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.title('Approximation of theta of the binary call and binary put')
plt.xlabel('Stock price')
plt.ylabel('Theta')
plt.legend(loc='lower right')
plt.savefig('theta.png')
plt.clf()

# f. [2 points] Make a graph of delta and gamma of the binary call with S = K = 100 over a grid of T - t in {1, 0.25, 0.1, 0.025, 0.01, 0.0025, 0.001, 0.00025, 0.0001}.
delta_t_vector = np.array([1, 0.25, 0.1, 0.025, 0.01, 0.0025, 0.001, 0.00025, 0.0001])
plt.plot(delta_t_vector,binary_call_option_delta_approx(100, 100, 0.025,delta_t_vector,0.2,0.0001), color = 'b', label='Delta Binary Call Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.title('Approximation of Delta of the binary call')
plt.xlabel('Time')
plt.ylabel('Delta')
plt.legend(loc='upper right')
plt.savefig('delta_in_time.png')
plt.clf()

delta_t_vector = np.array([1, 0.25, 0.1, 0.025, 0.01, 0.0025, 0.001, 0.00025, 0.0001])
delta_t_vector = np.array([1, 0.25, 0.1, 0.025, 0.01, 0.0025, 0.001, 0.00025, 0.0001])
plt.plot(delta_t_vector,binary_call_option_gamma_approx(100, 100, 0.025,delta_t_vector,0.2,0.0001), color = 'b', label='Gamma Binary Call Option')
plt.axhline(y = 0, color = 'k', linestyle = ':')
plt.title('Approximation of Gamma of the binary call')
plt.xlabel('Time')
plt.ylabel('Gamma')
plt.legend(loc='lower right')
plt.savefig('gamma_in_time.png')
plt.clf()