import matplotlib.pyplot as plt
## SÓ FUNCIONARÁ COM BASES DE DADOS BIDIMENSIONAIS


class plotador_de_grafico():
    def __init__(self, dados, classes, pesos):
        self.dados = dados
        self.classes = classes
        self.pesos = pesos

    def plot_data(self):
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Base de Dados Classificada")
        for i in range(len(self.dados)):
            dado =self.dados[i]
            classe = dado[-1]
            if classe == 1:
                plt.plot(dado[0],dado[1],marker='*',color='blue', label="+1")
            elif classe == -1:
                plt.plot(dado[0], dado[1], marker='o',color='red', label="-1")
        plt.show()

    def plot_reta(self,bias):
        pass
        pesoA = self.pesos[0]
        pesoB = self.pesos[1]
        # inX = -(bias - pesoB*)
        plt.axline((0, ), (1, ))
        pass

    def plot_eqm(self,lista_eqm):
        plt.title("EQM / Época")
        plt.xlabel("Época")
        plt.ylabel("EQM(W)")
        plt.plot(range(len(lista_eqm)),lista_eqm)
        plt.show()