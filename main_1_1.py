import numpy as np
import matplotlib.pyplot as plt

# Dados: selecionamos a primeira variável (primeira coluna)
dados = [
    107, 145, 237, 91, 185, 106, 177, 120, 116, 105,
    109, 186, 257, 218, 164, 158, 117, 130, 132, 138,
    131,  88, 161, 145, 128, 231,  78, 113, 134, 104,
    122, 442, 237, 148, 231, 161, 119, 185, 118,  98,
    218, 147, 176, 106, 109, 138,  84, 137, 139,  97,
    169, 160, 123, 130, 198, 215, 177, 100,  91, 141,
    139, 176, 218, 146, 128, 127,  76, 126, 184,  58,
     95, 144, 124, 167, 150, 156, 193, 194,  73,  98,
    127, 153, 161, 194,  87, 188, 149, 215, 163, 111,
    198, 265, 143, 136, 298, 173, 148, 110, 188, 208
]

# Função g(\mu) e sua derivada g'(\mu)
def g(mu, data):
    return -np.sum(-2 * np.array(data) + 2 * mu) / 2

def g_prime(mu, data):
    return len(data)  # Derivada é constante: n

# Método de Newton-Raphson
def newton_raphson(data, tol=1e-6, max_iter=100):
    mu_k = np.mean(data)  # Chute inicial: média dos dados
    iteracoes = [mu_k]

    for _ in range(max_iter):
        g_value = g(mu_k, data)
        g_prime_value = g_prime(mu_k, data)

        if abs(g_prime_value) < 1e-10:
            raise ValueError("Derivada muito próxima de zero. Newton-Raphson não converge.")

        mu_k1 = mu_k - g_value / g_prime_value
        iteracoes.append(mu_k1)

        if abs(mu_k1 - mu_k) < tol:
            break

        mu_k = mu_k1

    return mu_k, iteracoes

# Cálculo do EMV (\bar{x})
emv = np.mean(dados)

# Estimativa pelo método de Newton-Raphson
estimativa_mu, iteracoes = newton_raphson(dados)

# Gráfico de convergência
plt.figure(figsize=(10, 6))
plt.plot(iteracoes, label="Newton-Raphson", marker="o")
plt.axhline(y=emv, color="r", linestyle="--", label="EMV (\u03BC = {:.2f})".format(emv))
plt.title("Convergência do Método de Newton-Raphson")
plt.xlabel("Iterações")
plt.ylabel("\u03BC")
plt.legend()
plt.grid()
plt.show()

# Exibição dos resultados
print(f"EMV (Média dos dados): {emv:.6f}")
print(f"Estimativa de \u03BC pelo Newton-Raphson: {estimativa_mu:.6f}")
print(f"Iterações do Newton-Raphson: {len(iteracoes)}")