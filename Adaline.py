## Critério de parada -> Certo para todas as entradas | Pesos não se alteram mais
class Adaline:
    #   Construtor
    ## A entrada deve ser passada com uma Lista de Amostras, onde cada Amostra é uma Lista de características, sendo o último elemento da lista a classificação desejada
    def __init__(self, amostras=[[]], taxa_de_aprendizagem=0.1, teta=0, bias=0):
        ## atribuição de constantes
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.teta = teta
        self.bias = bias
        # apesar de a última posição ser a característica esperada, vou adicionar o bias, então a contagem continua correta
        self.número_de_caracteristicas = len(amostras[0])
        # INSERÇÃO DO BIAS
        if bias == 0:
            #se não está sendo usado, corrija o número de características
            self.usando_bias = False
            self.amostras = amostras
            self.número_de_caracteristicas -= 1
        else:
            #se está usando o bias, insira-o em cada elemento
            self.usando_bias = True
            self.amostras = self.inserir_bias_em_lista(amostras)
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
    def inserir_bias_em_lista(self,lista_de_amostras):
        nova_lista = []
        #insere o bias nas amostras (na primeira posição -> X0)
        for exemplo in lista_de_amostras:
            exemplo.insert(0,self.bias)
            nova_lista.append(exemplo)
        return nova_lista

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
        gg = 1 if not self.usando_bias else 0
        for característica in amostra:
            if contador == 0 and self.usando_bias:
                print("X" + str(contador) + " = " + str(característica) + " -> (Bias)")
            elif contador == self.número_de_caracteristicas:
                print("Classificação = " + str(característica))
            else:
                print("X" + str(contador+gg) + " = " + str(característica))
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

    # testa e dá saída para uma amostra de fora da base de treino
    def testar_entrada(self, entrada)->int:
        if self.usando_bias:
            entrada.insert(0, self.bias)
        return self.função_de_ativação(self.calculo_de_u(entrada))

    #Testa uma base de dados e retorna um trio de listas:(Saídas_Esperadas, Saídas_Obtidas, Acertos)
    def testar_uma_base(self,base)->(list,list,list):
        saídas_esperadas = []
        saídas_obtidas = []
        lista_de_acertos = []
        for exemplo in base:
            saída_esperada = exemplo[-1]
            saída_do_ada = self.testar_entrada(exemplo)
            saídas_esperadas.append(saída_esperada)
            saídas_obtidas.append(saída_do_ada)
            lista_de_acertos.append(saída_esperada==saída_do_ada)
        return saída_esperada,saídas_obtidas,lista_de_acertos

    # testa e dá a saída para uma amostra de dentro da base de treino
    def testar_exemplo(self,exemplo)->int:
        return self.função_de_ativação(self.calculo_de_u(exemplo))

    #calcula o erro para o adaline
    def calculo_de_erro(self, amostra)->float:
        saída_desejada = amostra[-1]
        u = self.calculo_de_u(amostra)
        return saída_desejada - u

    #calcula o Erro quadratico
    def calculo_de_EQ(self,lista_de_quadrados_de_erros):
        erro_quadrático = 1/2 * sum(lista_de_quadrados_de_erros)
        return erro_quadrático

    #calcula o Erro Quadratico Medio
    def calculo_de_EQM(self,lista_de_quadrados_de_erros,numero_de_exemplos):
        return 1/numero_de_exemplos * sum(lista_de_quadrados_de_erros)

    #treina o adaline
    def treinar(self,precisão=0,lim_épocas = 100):
        contador_de_épocas = 1
        lista_eqm=[]
        sinal_de_convergencia = False
        #loop de treinamento
        print(50 * "______" + "\tTREINAMENTO INICIADO" + 50 * "_")
        eqm_atual = 0
        while contador_de_épocas < lim_épocas and not sinal_de_convergencia:
            #inicia lista de pesos vazia para ser preenchida pedo lms
            novos_pesos = self.pesos
            erros_da_época = []
            #inicia uma nova época de Treinamento
            print("\n\tÉpoca : " + str(contador_de_épocas) + "\n")
            for exemplo in self.amostras:
                # self.testar_exemplo(exemplo)
                erro = self.calculo_de_erro(exemplo)
                self.imprimir_uma_amostra(exemplo)
                print("Ada_Saída -> " + str(self.testar_exemplo(exemplo)))
                print(25*"_")
                ##adiciona um quadrado de erro na lista de erros da época
                erros_da_época.append(pow(erro,2))
                ###### Regra Delta (LMS) -> Wj+1 = Wj + (taxa_de_aprendizagem * erro) * Xi ########
                for i in range(self.número_de_caracteristicas):
                    novos_pesos[i] += self.taxa_de_aprendizagem * erro * exemplo[i]
                    ################################################################################
            #Atualiza os pesos do adaline após seu cálculo (atualizando ao final da época)
            self.pesos = novos_pesos
            eqm_anterior = eqm_atual
            eqm_atual = self.calculo_de_EQM(erros_da_época,len(self.amostras))
            lista_eqm.append(eqm_atual)
            print("Época ["+str(contador_de_épocas)+"] EQM -> "+str(eqm_atual))
            #Contar final da época de treinamento
            #Compara se o módulo de EQM atual - Anterior <= Precisão, se sim, atingiu a precisão desejada
            if abs(eqm_atual - eqm_anterior) <= precisão and len(lista_eqm) >=2:
                #Se sim, informa que convergiu, e para o treinamento
                sinal_de_convergencia = True
                print(50*"_")
                print("Convergência na época ["+str(contador_de_épocas)+"]")
            else:
                contador_de_épocas += 1
        #Fim do treinamento
        print("Treinamento Completo em ["+str(contador_de_épocas)+"] Épocas.")
        print(50 * "_")
        return lista_eqm