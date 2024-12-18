import numpy as np
import matplotlib.pyplot as plt

# Dados da terceira coluna fornecida
data = [102, 138, 190, 122, 128, 112, 128, 116, 134, 104, 116, 152, 134, 132, 130, 118,
        136, 108, 108, 128, 118, 134, 178, 134, 162, 162, 120, 98, 144, 118, 118, 138,
        134, 108, 96, 142, 122, 146, 126, 176, 104, 112, 140, 102, 142, 146, 92, 112,
        152, 116, 118, 128, 116, 134, 108, 134, 124, 124, 114, 154, 114, 114, 98, 128,
        130, 122, 112, 106, 128, 128, 128, 116, 154, 126, 140, 122, 154, 140, 120, 140,
        114, 122, 94, 122, 172, 100, 150, 154, 170, 140, 144, 156, 132, 140, 150, 130,
        118, 162, 128, 130, 208]

# Função h(sigma^2)
def h(sigma2, n, soma_quadrados):
    return -n / (2 * sigma2) + soma_quadrados / (2 * sigma2**2)

# Derivada de h(sigma^2)
def h_prime(sigma2, n, soma_quadrados):
    return n / (2 * sigma2**2) - soma_quadrados / (sigma2**3)

# Estimativa de Máxima Verossimilhança (EMV)
n = len(data)
media = np.mean(data)
soma_quadrados = sum((x - media)**2 for x in data)
sigma2_emv = soma_quadrados / n

sigma2_nr = sigma2_emv / 2  # Chute inicial
iterations = [sigma2_nr]  # Para registrar as iterações

epsilon = 1e-6  # Critério de convergência
max_iter = 100  # Máximo de iterações
for _ in range(max_iter):
    h_value = h(sigma2_nr, n, soma_quadrados)
    h_prime_value = h_prime(sigma2_nr, n, soma_quadrados)
    sigma2_next = sigma2_nr - h_value / h_prime_value
    iterations.append(sigma2_next)
    if abs(sigma2_next - sigma2_nr) < epsilon:
        break
    sigma2_nr = sigma2_next

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(iterations, label="Newton-Raphson", marker="o")
plt.axhline(y=sigma2_emv, color="r", linestyle="--", label="EMV (Estimativa Direta)")
plt.title(r"Estimativa de $\sigma^2$ usando Newton-Raphson e EMV")
plt.xlabel("Iterações")
plt.ylabel(r"$\sigma^2$")
plt.legend()
plt.grid()
plt.show()

# Resultados finais
print(f"Estimativa de \u03c32 usando Newton-Raphson: {sigma2_nr}")
print(f"Estimativa de \u03c32 usando EMV: {sigma2_emv}")
