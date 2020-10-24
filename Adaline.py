## Critério de parada -> Certo para todas as entradas | Pesos não se alteram mais
class adaline:
    #   Construtor
    ## A entrada deve ser passada com uma Lista de Amostras, onde cada Amostra é uma Lista de características, sendo o último elemento da lista a classificação desejada
    def __init__(self, amostras=[[]], taxa_de_aprendizagem=0.1, épocas_de_treino=100,teta=0, bias=1):
        ## atribuição de constantes
        self.amostras = amostras
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.teta = teta
        self.bias = bias
        self.épocas_de_treino = épocas_de_treino
        # apesar de a última posição ser a característica esperada, vou adicionar o bias, então a contagem continua correta
        self.número_de_caracteristicas = len(amostras[0])
        # INSERÇÃO DO BIAS
        if teta == 0:
            #se não está sendo usado, corrija o número de características
            self.usando_bias = False
            self.número_de_caracteristicas -= 1
        else:
            #se está, insira-o no conjunto de treino
            self.usando_bias = True
            self.inserir_bias_nas_amostras()
        #INICIAÇÃO DOS PESOS
        self.pesos = self.criar_lista_de_pesos_zerada()

    # To_String
    def __str__(self) -> str:
        contador_de_pesos = 0
        if self.usando_bias:
            string = ("Adaline\n" +
                      "Taxa de Aprendizagem = " + str(self.taxa_de_aprendizagem) + "\n" +
                      "Quantidade de Características = " + str(self.número_de_caracteristicas) + "\n" +
                      "Bias de Vetor = (X0 = " + str(self.bias) + ") | (W0 = " + str(self.teta) + ")\n" +
                      "Pesos = Vetor W")
        else:
            string = ("Adaline\n" +
                      "Taxa de Aprendizagem = " + str(self.taxa_de_aprendizagem) + "\n" +
                      "Quantidade de Características = " + str(self.número_de_caracteristicas) + "\n" +
                      "Sem uso de Bias\n" +
                      "Pesos = Vetor W")
            contador_de_pesos+=1

        for peso in self.pesos:
            string += "\nW" + str(contador_de_pesos) + " = " + str(peso)
            contador_de_pesos += 1
        return string

    ## Cria lista de pesos com zeros, onde o primeiro elemento recebe Teta
    def criar_lista_de_pesos_zerada(self) -> list:
        lista_de_pesos = [0] * self.número_de_caracteristicas
        lista_de_pesos[0] = self.teta
        return lista_de_pesos

    # Retorna a lista de pesos do Adaline
    def pegar_lista_de_pesos(self) -> list:
        return self.pesos

    #insere o bias nas amostras de aprendizado
    def inserir_bias_nas_amostras(self):
        #insere o bias nas amostras
        for exemplo in self.amostras:
            exemplo.insert(0,self.bias)

    #imprime base de dados de aprendizado
    def imprimir_amostras(self):
        print("AMOSTRAS")
        print(10 * "_")
        contador_de_exemplo = 1
        for exemplo in self.amostras:
            print("AMOSTRA "+str(contador_de_exemplo))
            self.imprimir_uma_amostra(exemplo)
            contador_de_exemplo +=1
            print(10 * "_")

    #imprime um único exemplo
    def imprimir_uma_amostra(self,amostra):
        contador = 0
        for característica in amostra:
            if contador == 0:
                print("X" + str(contador) + " = " + str(característica) + " -> (Bias)")
            elif contador == self.número_de_caracteristicas:
                print("Classificação = " + str(característica))
            else:
                print("X" + str(contador) + " = " + str(característica))
            contador += 1

    # calcula indução do adaline para uma amostra
    def calculo_de_u(self, amostra) -> float:
        u = 0
        # u = pesoi * amostrai para todos
        for i in range(self.número_de_caracteristicas):
            u += self.pesos[i] * amostra[i]
        return u - self.teta

    # compara U ao Teta para retornar -1 ou 1
    def função_de_ativação(self, u) -> int:
        if u >= self.teta:
            return 1
        else:
            return -1

    # testa e dá saída
    def testar_amostra(self, amostra)->int:
        if self.usando_bias:
            amostra.insert(0,self.bias)
        return self.função_de_ativação(self.calculo_de_u(amostra))

    #calcula o erro para o adaline
    def calculo_de_erro(self, amostra)->float:
        saída_desejada = amostra[-1]
        u = self.calculo_de_u(amostra)
        return saída_desejada - u

    #treina o adaline
    def treinar(self):
        pronto = False
        contador_de_épocas = 1
        print(50 * "______" + "\tTREINAMENTO INICIADO" + 50 * "_")
        while contador_de_épocas < self.épocas_de_treino:
            novos_pesos = self.criar_lista_de_pesos_zerada()
            for exemplo in self.amostras:
                saída_desejada = exemplo[-1]
                erro = self.calculo_de_erro(exemplo)
                #Regra Delta (LMS) -> Wj+1 = Wj + (taxa_de_aprendizagem * erro) * Xi
                for i in range(self.número_de_caracteristicas):
                    novos_pesos[i] = self.pesos[i] + (self.taxa_de_aprendizagem * erro) * exemplo[i]
                self.pesos = novos_pesos
            contador_de_épocas += 1
        print("Treinamento Completo em ["+str(contador_de_épocas)+"] Épocas.\n")