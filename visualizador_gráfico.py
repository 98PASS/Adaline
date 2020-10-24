import matplotlib.pyplot as plt

def plotar_dados(dados, coeficiente_de_reta_a, coeficiente_de_reta_b):
    for amostra in dados:
        plt.plot(amostra, '.')
    plotar_reta(coeficiente_de_reta_a, coeficiente_de_reta_b)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def plotar_reta(coeficiente_de_reta_a, coeficiente_de_reta_b):
    pass