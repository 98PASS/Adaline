import Adaline as adl
import Perceptron as pcp
import visualizador_gráfico
import pandas as pd

################################# LENDO BASES DE DADOS ############################################

#teste com base aleatória da internet
basebb = pd.read_csv("base-adaline/baseex.txt",',',header=None)
basebb = pd.read_csv("base-adaline/elogico.txt",',',header=None)
#transformação da base para lista de listas
listabb = basebb.values.tolist()

##Base 1
conjunto_1 = pd.read_csv("base-adaline/base1.txt", "\t", header=None)
lista_1 = conjunto_1.values.tolist()
##Base 2
conjunto_2 = pd.read_csv("base-adaline/base2.txt", "\t", header=None)
lista_2 = conjunto_2.values.tolist()
##Bases Combinadas
bases = [conjunto_1, conjunto_2]
bases_combinadas = pd.concat(bases)
lista_combinada = bases_combinadas.values.tolist()
###############################################################################################

a = adl.Adaline(amostras=lista_2, taxa_de_aprendizagem=0.001)
eqm_lista=a.treinar(precisão=0.00000005,lim_épocas=1000)
Y,gu,acertos=a.testar_uma_base(lista_2)

print(a.__str__())

yey = 0
nay = 0
for valor in acertos:
    if valor:
        yey +=1
    else:
        nay +=1
print("Acertos  "+str(yey))
print("Erros: "+str(nay))
pesos = a.pegar_lista_de_pesos()
vg = visualizador_gráfico.plotador_de_grafico(lista_2,Y,pesos)
vg.plot_eqm(eqm_lista)
vg.plot_data()


#listaex_testes
# print("Ada ->\t1\t=\t" + str(a.testar_entrada([0.4329, -1.3719, 0.7022, -0.8535]))) #Para a base bb, deve retornar 1
# print("Ada ->\t-1\t=\t"+ str(a.testar_entrada([0.3024, 0.2286, 0.8630, 2.7909]))) #Para a base bb, deve retornar -1