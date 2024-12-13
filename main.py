import numpy as np

# Dados fornecidos (usando a terceira coluna)
data = [
    102, 138, 190, 122, 128, 112, 128, 116, 134, 104, 116, 152, 134, 132, 130, 118,
    136, 108, 108, 128, 118, 134, 178, 134, 162, 162, 120, 98, 144, 118, 118, 138,
    134, 108, 96, 142, 122, 146, 126, 176, 104, 112, 140, 102, 142, 146, 92, 112,
    152, 116, 118, 128, 116, 134, 108, 134, 124, 124, 114, 154, 114, 114, 98, 128,
    130, 122, 112, 106, 128, 128, 116, 154, 126, 140, 122, 154, 140, 120, 140, 114,
    122, 94, 122, 172, 100, 150, 154, 170, 140, 144, 156, 132, 140, 150, 130, 118,
    162, 128, 130, 208
]

def h(sigma2, data):
    n = len(data)
    term1 = -n / (2 * sigma2)
    term2 = sum((x - np.mean(data)) ** 2 for x in data) / (2 * sigma2 ** 2)
    return term1 + term2

def h_prime(sigma2, data):
    n = len(data)
    term1 = n / (2 * sigma2 ** 2)
    term2 = -2 * sum((x - np.mean(data)) ** 2 for x in data) / (2 * sigma2 ** 3)
    return term1 + term2

def newton_raphson(data, tol=1e-6, max_iter=100):
    sigma2 = np.var(data)  # Estimativa inicial
    for _ in range(max_iter):
        h_val = h(sigma2, data)
        h_prime_val = h_prime(sigma2, data)
        
        # Atualização de Newton-Raphson
        sigma2_new = sigma2 - h_val / h_prime_val
        
        # Verifica a convergência
        if abs(sigma2_new - sigma2) < tol:
            return sigma2_new
        sigma2 = sigma2_new
    
    raise ValueError("O método de Newton-Raphson não convergiu")

# Calculando a variância estimada
sigma2_est = newton_raphson(data)
print(f"Estimativa da variância usando Newton-Raphson: {sigma2_est:.6f}")
