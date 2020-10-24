class Perceptron:
    #   Construtor
    ## A entrada deve ser passada com uma Lista de Amostras, onde cada Amostra é uma Lista de características, sendo o último elemento da lista a classificação desejada
    def __init__(self, amostras=[[]], taxa_de_aprendizagem=0.1, teta=0, bias=0):
        ## atribuição de constantes
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.teta = teta
        self.bias = bias
        self.número_de_caracteristicas = len(amostras[0])
        # INSERÇÃO DO BIAS
        if bias == 0:
            # se não está sendo usado, corrija o número de características
            self.usando_bias = False
            self.amostras = amostras
            self.número_de_caracteristicas -= 1
        else:
            # se está usando o bias, insira-o em cada elemento
            self.usando_bias = True
            self.amostras = self.inserir_bias_em_lista(amostras)
        # INICIAÇÃO DOS PESOS
        self.pesos = self.criar_lista_de_pesos_zerada()

    # To_String
    def __str__(self) -> str:
        contador_de_pesos = 0
        if self.usando_bias:
            string = ("Perceptron\n" +
                      "Taxa de Aprendizagem = " + str(self.taxa_de_aprendizagem) + "\n" +
                      "Quantidade de Características = " + str(self.número_de_caracteristicas) + "\n" +
                      "Bias de Vetor = (X0 = " + str(self.bias) + ") | (W0 = " + str(self.teta) + ")\n" +
                      "Pesos = Vetor W")
        else:
            string = ("Perceptron\n" +
                      "Taxa de Aprendizagem = " + str(self.taxa_de_aprendizagem) + "\n" +
                      "Quantidade de Características = " + str(self.número_de_caracteristicas) + "\n" +
                      "Sem uso de Bias\n" +
                      "Pesos = Vetor W")
            contador_de_pesos += 1
        for peso in self.pesos:
            string += "\nW" + str(contador_de_pesos) + " = " + str(peso)
            contador_de_pesos += 1
        return string

    # insere o bias nas amostras de aprendizado
    def inserir_bias_em_lista(self, lista_de_amostras):
        nova_lista = []
        # insere o bias nas amostras (na primeira posição -> X0)
        for exemplo in lista_de_amostras:
            exemplo.insert(0, self.bias)
            nova_lista.append(exemplo)
        return nova_lista

    ## Cria lista de pesos com zeros, onde o primeiro elemento recebe Teta
    def criar_lista_de_pesos_zerada(self) -> list:
        lista_de_pesos = [0] * self.número_de_caracteristicas
        lista_de_pesos[0] = self.teta
        return lista_de_pesos

    #Dá a lista de pesos do Perceptron
    def pegar_lista_de_pesos(self) -> list:
        return self.pesos

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

    #calcula indução do perceptron para uma amostra
    def calculo_de_u(self,amostra) -> float:
        u=0
        # u = pesoi * amostrai para todos
        for i in range(self.número_de_caracteristicas):
           u += self.pesos[i] * amostra[i]
        return u

    #compara U ao Teta para retornar 0 ou 1
    def função_de_ativação(self,u) -> int:
        if u >= self.teta:
            return 1
        else:
            return 0

    #calcula o erro (saída desejada - saída obtida)
    def cálculo_de_erro(self, amostra):
        saída_obtida = self.função_de_ativação(self.calculo_de_u(amostra))
        saída_desejada = amostra[-1]
        return saída_desejada - saída_obtida

    #testa e dá saída
    def testar_entrada(self, amostra)->int:
        if self.usando_bias:
            amostra.insert(0,self.bias)
        return self.função_de_ativação(self.calculo_de_u(amostra))

    #Treina o perceptron com critério de parada de 100% de acerto para base de treino
    def treinar(self):
        contador_de_época = 1
        pronto = False
        print(50*"______"+"\tTREINAMENTO INICIADO"+50*"_")
        while not pronto:
            print("\n\tÉpoca : "+str(contador_de_época)+"\n")
            n_erros = 0 # Contagem do número de erros de cada época
            indice = 1
            pronto = True
            for exemplo in self.amostras:
                erro = self.cálculo_de_erro(exemplo)

                #O critério de parada é passar por TODA UMA ÉPOCA de testes sem presenciar nenhum erro
                if erro != 0:
                    ##Se ele errou um único exemplo sequer, o perceptron ainda não está pronto
                    n_erros += 1 #conta mais um erro na época atual
                    pronto = False
                    # print(15 * "_")
                    print("Errou no Exemplo " + str(indice))
                    print(15 * "_")
                else:
                    # print(15 * "_")
                    print("Acertou no Exemplo " + str(indice))
                    print(15 * "_")
                indice += 1
                #Loop da função Wj+1 = Wj + ( alfa * Xj * erro)
                novos_pesos = self.criar_lista_de_pesos_zerada()
                for i in range(self.número_de_caracteristicas):
                    novo_peso = self.pesos[i] + ( self.taxa_de_aprendizagem * exemplo[i] * erro)
                    novos_pesos[i]=novo_peso
                self.pesos = novos_pesos
            contador_de_época += 1
        print("Treinamento Finalizado em ["+str(contador_de_época-1)+"] Épocas.")
