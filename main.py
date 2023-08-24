# IMPORTANDO BIBLIOTECAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# CARREGANDO DADOS
data_frame=pd.read_csv('dataset.csv')
'''# Verificando shape do dataset
print(data_frame.shape)
# Colunas do dataset
print(data_frame.columns)
# Amostra dos dados
print(data_frame.head())
# Informações sobre o conjunto de dataset carregado.
print(data_frame.info)'''

# ANÁLISE EXPLORATÓRIA - RESUMO ESTATÍSTICO
'''# Verificar se há valores ausentes
# A função isnull verifica os valores ausentes e a função sum soma e retorna a quantia
print(data_frame.isnull().sum())
# Imprimendo o coeficiente de correlação, que proxima de de 1 forte relação positiva, proxima de menos -1 for correlação negativa, em zero nenhuma relação.abs
print(data_frame.corr())
# Resumo estatistico dp dataset
print(data_frame.describe())
# Resumo estatistico da variavel preditora
print(data_frame["horas_estudo_mes"].describe())
# Histograma da variavel preditora
sns.histplot(data=data_frame, x="horas_estudo_mes", kde=True)
plt.show()'''

#PREPARAR VARIÁVEL DE ENTRADA X
x = np.array(data_frame['horas_estudo_mes'])
'''print(type(x))
print(x.shape)'''
# Ajusta o shape de x
x = x.reshape(-1,1)
'''print(type(x))
print(x.shape)'''
# Prepara a variável alvo
y = data_frame['salario']

'''# Gráfico de dispeersão entre x e y #Mostra a relação entre duas variáveis
plt.scatter(x,y, color = "blue", label = "DADOS REAIS HISTÒRICOS")
plt.xlabel("Horas de Estudo")
plt.ylabel("Salario")
plt.legend()
plt.show()'''

#Dividir dados em treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.2,random_state=42)
'''print(x_teste.shape)
print(x_treino.shape)
print(y_teste.shape)
print(y_treino.shape)'''

#MODELAGEM PREDITIVA (Machine Learning)
#Cria o modelo de regressão linear simples
modelo = LinearRegression()
#Treina o modelo
modelo.fit(x_treino, y_treino)
'''# Visualiza a reta de regreção linear (previsões) e os dados reais usados no treinamento
plt.scatter(x, y, color = 'blue', label = 'DADOS REAIS HISTÓRICOS' )
plt.plot(x, modelo.predict(x), color = 'red', label = 'Reta de Regressão com as Previsões do Modelo')
plt.xlabel("Horas de Estudo")
plt.ylabel("Salario")
plt.legend()
plt.show()
'''

#avaliar o modelo nos dados de teste
score = modelo.score(x_teste,y_teste)
print(f"Coeficiente R^2: {score:.2f}")

#Intercepto - parâmetro w0
print(modelo.intercept_)
# Slope - parâmetro w1
print(modelo.coef_)


#DEPLOY DO MODELO
#Define umnovo valor para horas de estudo
#Exemplo 1
horas_de_estudo_novo = np.array([[48]])
#Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_de_estudo_novo)
print(f"Se você estudar {horas_de_estudo_novo[0][0]} horas por mês, seu salário pode ser igual a {salario_previsto[0]:.2f}")
#Exemplo 2
#Define umnovo valor para horas de estudo
horas_de_estudo_novo = np.array([[60]])
#Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_de_estudo_novo)
print(f"Se você estudar {horas_de_estudo_novo[0][0]} horas por mês, seu salário pode ser igual a {salario_previsto[0]:.2f}")
