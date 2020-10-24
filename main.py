import Adaline as adl
import Perceptron as pcp
import visualizador_gráfico
import pandas as pd

################################# LENDO BASES DE DADOS ############################################

#teste com base aleatória da internet
basebb = pd.read_csv("base-adaline/baseex.txt",',',header=None)
#transformação da base para lista de listas
listabb = basebb.values.tolist()

##Base de treino real
conjunto_de_treino = pd.read_csv("base-adaline/base1.txt","\t",header=None)
lista_para_treino = conjunto_de_treino.values.tolist()

##Base para teste
conjunto_de_teste = pd.read_csv("base-adaline/base2.txt","\t",header=None)
lista_para_teste = conjunto_de_teste.values.tolist()
###############################################################################################

a = adl.adaline(amostras=listabb,taxa_de_aprendizagem=0.1)
a.treinar()
print(a.__str__())
print(a.pegar_lista_de_pesos())
print(a.testar_amostra([0.4329, -1.3719, 0.7022, -0.8535])) #Para a base bb, deve retornar 1
print(a.testar_amostra([0.3024, 0.2286, 0.8630, 2.7909])) #Para a base bb, deve retornar -1