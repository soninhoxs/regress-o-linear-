import copy, math
import numpy as np
import matplotlib.pyplot as plt
 
plt.style.use('ggplot')
np.set_printoptions(precision=2)
 
 
# Dados de treino: [tamanho, quartos, andares, idade] e preços
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
 
print(f"Formato de X: {X_train.shape}, Tipo de X:{type(X_train)})")
print(X_train)
print(f"Formato de y: {y_train.shape}, Tipo de y:{type(y_train)})")
print(y_train)
 
 
# Parâmetros iniciais pré-definidos (valores ótimos conhecidos)
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"Formato de w_init: {w_init.shape}, Tipo de b_init: {type(b_init)}")
 
 
def predict_single_loop(x, w, b): 
    """
    Previsão usando regressão linear com loop explícito.
    
    Argumentos:
      x (ndarray): Formato (n,) — exemplo com múltiplas features
      w (ndarray): Formato (n,) — parâmetros do modelo
      b (scalar):  parâmetro do modelo (viés)
      
    Retorna:
      p (scalar):  previsão
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]   # multiplica cada feature pelo seu peso
        p = p + p_i          # acumula o resultado
    p = p + b                # adiciona o viés
    return p
 
 
# Pega uma linha dos dados de treino
x_vec = X_train[0,:]
print(f"Formato de x_vec: {x_vec.shape}, Valor de x_vec: {x_vec}")
 
# Faz uma previsão com loop
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"Previsão f_wb: {f_wb}")
 
 
def predict(x, w, b): 
    """
    Previsão usando regressão linear com produto escalar (mais eficiente).
 
    Argumentos:
      x (ndarray): Formato (n,) — exemplo com múltiplas features
      w (ndarray): Formato (n,) — parâmetros do modelo
      b (scalar):  parâmetro do modelo (viés)
      
    Retorna:
      p (scalar):  previsão
    """
    p = np.dot(x, w) + b     # produto escalar vetorizado
    return p    
 
 
# Pega uma linha dos dados de treino
x_vec = X_train[0,:]
print(f"Formato de x_vec: {x_vec.shape}, Valor de x_vec: {x_vec}")
 
# Faz uma previsão com produto escalar
f_wb = predict(x_vec, w_init, b_init)
print(f"Previsão f_wb: {f_wb}")
 
 
def compute_cost(X, y, w, b): 
    """
    Calcula o custo (erro quadrático médio) da regressão linear.
 
    Argumentos:
      X (ndarray (m,n)): Dados com m exemplos e n features
      y (ndarray (m,)) : valores alvo
      w (ndarray (n,)) : parâmetros do modelo
      b (scalar)       : parâmetro do modelo (viés)
      
    Retorna:
      cost (scalar): custo calculado
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b        # previsão para o exemplo i
        cost = cost + (f_wb_i - y[i])**2    # erro ao quadrado
    cost = cost / (2 * m)                   # média do erro quadrático
    return cost
 
 
# Calcula e exibe o custo com os parâmetros ótimos pré-definidos
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Custo com w ótimo: {cost}')
 
 
def compute_gradient(X, y, w, b): 
    """
    Calcula o gradiente do custo em relação a w e b.
 
    Argumentos:
      X (ndarray (m,n)): Dados com m exemplos e n features
      y (ndarray (m,)) : valores alvo
      w (ndarray (n,)) : parâmetros do modelo
      b (scalar)       : parâmetro do modelo (viés)
      
    Retorna:
      dj_dw (ndarray (n,)): gradiente do custo em relação a w
      dj_db (scalar):       gradiente do custo em relação a b
    """
    m, n = X.shape           # m = número de exemplos, n = número de features
    dj_dw = np.zeros((n,))
    dj_db = 0.
 
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]    # erro da previsão
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]  # gradiente em relação a w[j]
        dj_db = dj_db + err                        # gradiente em relação a b
    dj_dw = dj_dw / m   # média dos gradientes de w
    dj_db = dj_db / m   # média do gradiente de b
        
    return dj_db, dj_dw
 
 
# Calcula e exibe o gradiente nos parâmetros iniciais
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db nos parâmetros iniciais: {tmp_dj_db}')
print(f'dj_dw nos parâmetros iniciais: \n {tmp_dj_dw}')
 
 
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Executa o gradiente descendente em lote para aprender w e b.
    Atualiza os parâmetros a cada iteração usando a taxa de aprendizado alpha.
    
    Argumentos:
      X (ndarray (m,n))   : Dados com m exemplos e n features
      y (ndarray (m,))    : valores alvo
      w_in (ndarray (n,)) : parâmetros iniciais do modelo
      b_in (scalar)       : parâmetro inicial do modelo (viés)
      cost_function       : função para calcular o custo
      gradient_function   : função para calcular o gradiente
      alpha (float)       : taxa de aprendizado
      num_iters (int)     : número de iterações
      
    Retorna:
      w (ndarray (n,)) : parâmetros w atualizados
      b (scalar)       : parâmetro b atualizado
    """
    
    J_history = []  # histórico do custo para plotar depois
    w = copy.deepcopy(w_in)  # evita modificar o w global
    b = b_in
    
    for i in range(num_iters):
 
        # Calcula o gradiente e atualiza os parâmetros
        dj_db, dj_dw = gradient_function(X, y, w, b)
 
        # Atualiza w e b com o gradiente e a taxa de aprendizado
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
      
        # Salva o custo a cada iteração (limite para evitar uso excessivo de memória)
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))
 
        # Exibe o custo a cada 10% das iterações
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4d}: Custo {J_history[-1]:8.2f}   ")
        
    return w, b, J_history
 
 
# Inicializa os parâmetros com zero
initial_w = np.zeros_like(w_init)
initial_b = 0.
 
# Configurações do gradiente descendente
iterations = 1000
alpha = 5.0e-7
 
# Executa o gradiente descendente
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b e w encontrados pelo gradiente descendente: {b_final:0.2f}, {w_final} ")
 
# Exibe as previsões finais comparadas com os valores reais
m, _ = X_train.shape
for i in range(m):
    print(f"Previsão: {np.dot(X_train[i], w_final) + b_final:0.2f}, Valor real: {y_train[i]}")
 
 
# Plota o custo ao longo das iterações
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Custo vs. Iteração")       ;  ax2.set_title("Custo vs. Iteração (final)")
ax1.set_ylabel('Custo')                   ;  ax2.set_ylabel('Custo')
ax1.set_xlabel('Passo da iteração')       ;  ax2.set_xlabel('Passo da iteração')
plt.show()