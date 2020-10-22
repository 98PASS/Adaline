## Critério de parada -> Certo para todas as entradas | Pesos não se alteram mais
class perceptron:

    #################################### FUNÇÕES PARA INICIALIZAÇÃO ####################################

    #   Construtor
    ## A entrada deve ser passada com uma Lista de Listas, onde as listas dentro da Lista_de_Features é um EXEMPLO com sua classificação
    def __init__(self, taxa_de_aprendizagem, teta, exemplos, bias=0):
        ## atribuição de constantes
        self.bias = bias  # a extensão
        self.lista_de_exemplos = exemplos
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.teta = teta
        ##
        if bias != 0:
            self.colocar_bias()
        ##
        self.dimensoes = len(exemplos[0]) - 1
        self.lista_de_pesos = self.iniciar_nova_lista_de_pesos()




    #  Método ToString que retorna os pesos e número de dimensões do perceptron
    def __str__(self):
        return ("Perceptron\n" +
                "Quantia de Dimensões:\t[" + str(self.dimensoes) +"]\n" +
                "Lista de Pesos:\t" + str(self.lista_de_pesos))

    #   Iniciar todos os pesos como 0
    def iniciar_nova_lista_de_pesos(self):
        lista = []
        # tem pesos para cada característica + bias , que fica no final
        for i in range(self.dimensoes):
            lista.append(0)
        # se existir um bias, adicione o teta a lista de pesos
        if self.bias != 0:
            lista[0]=self.teta
        return lista

    #   Adiciona o bias (vetor extendido)
    def colocar_bias(self):
        for i in range(len(self.lista_de_exemplos)):
            # print("antes: "+str(self.lista_de_exemplos[i]))
            #insere na posição 0 a característica do bias
            self.lista_de_exemplos[i].insert(0,self.bias)
            # print("depois: "+str(self.lista_de_exemplos[i]))

    #   Imprime base de dados
    def imprimir_base(self):
        contador = 1
        for exemplo in self.lista_de_exemplos:
            print("Exemplo: " + str(contador))
            contador += 1
            for i in range(len(exemplo)):
                if i == 0:
                    print("Bias: \t" + str(exemplo[i]))
                elif i < self.dimensoes:
                    print("X" + str(i) + " : \t" + str(exemplo[i]))
                else:
                    print("Classe:\t" + str(exemplo[i]))
            print(50*"-")
    #####################################################################################################

    ################################### FUNÇÕES PARA PERCEPTRON COMUM ###################################

    #   Junção somadora retorna u = somatório (i ate n) de Feature[i] * Peso[i]
    ## Entrada -> Um exemplo (que é uma lista)
    def junção_somadora(self, exemplo):
        u = 0
        saída_desejada = exemplo[-1]  # extração da saída desejada
        for i in range(len(exemplo) - 1):
            u += exemplo[i] * self.lista_de_pesos[i]
        return u, saída_desejada

    #   Função de ativação retorna 0 ou 1 - Compara U >= Teta ? 1 , 0
    def função_de_ativação(self, u):
        if u >= self.teta:
            return 1
        else:
            return 0

    #   Calcula o erro ->(Saída desejada - Saída Obtida da função_de_ativação)
    def calcular_erro_perceptron(self, saída_desejada, saída_obtida):
        erro = saída_desejada - saída_obtida
        return erro

    #   Retorna LISTA DE PESOS atualizada, pedindo como entrada da função o Erro obtido
    def atualizar_lista_de_pesos_perceptron(self, erro, exemplo):
        a = self.taxa_de_aprendizagem
        pesos_atualizados = []
        for i in range(len(self.lista_de_pesos)):
            peso = self.lista_de_pesos[i]
            entrada = exemplo[i]
            pesos_atualizados.append(peso + (a * entrada * erro))
        return pesos_atualizados

    #   Compara uma lista de pesos com a lista de pesos atual do perceptron
    ##  Se as listas forem iguais, retorna True, senão, atualiza a lista de pesos para a nova lista e retorna False
    def comparar_lista_de_pesos(self, novos_pesos):
        for i in range(len(self.lista_de_pesos)):
            if (novos_pesos[i] != self.lista_de_pesos[i]):
                self.lista_de_pesos = novos_pesos
                return False
        return True

    #   Treina o perceptron
    def treinar_perceptron(self):
        pronto = False
        contador = 0  # Conta quantas vezes a lista permaneceu inalterada
        número_de_exemplos = len(self.lista_de_exemplos)
        while not pronto:  # Repita até que esteja pronto -> os pesos parem de se alterar
            for exemplo in self.lista_de_exemplos:
                u, saída_desejada = self.junção_somadora(exemplo)
                saída_obtida = self.função_de_ativação(u)
                erro = self.calcular_erro_perceptron(saída_desejada, saída_obtida)
                novos_pesos = self.atualizar_lista_de_pesos_perceptron(erro, exemplo)
                listas_iguais = self.comparar_lista_de_pesos(novos_pesos)
                if listas_iguais:
                    contador += 1
                    listas_iguais=False
                    print("Iterações sem Mudança de Pesos : "+ str(contador))
                    # Se o número de vezes que a lista permaneceu inalterada for igual ao número de exemplos, então o Perceptron  atingiu seu ponto de parada
                    if contador == número_de_exemplos:
                        print("Treinamento completo, Perceptron Pronto")
                        pronto = True
                        break
                else:
                    print("Novos Pesos: " + str(self.lista_de_pesos))
                    contador = 0
    #####################################################################################################

    ####################################### FUNÇÕES PARA ADALINE ########################################

    #   (ADALINE) Regra Delta para atualização de Pesos com erro mínimo
    def regra_delta(self):
        gradiente = self.calculo_gradiente()

    # calcula o erro para o vetor de pesos (que é uma lista)
    def erro_adaline(self,exemplo):
        erro =0
        u, saída_desejada = self.junção_somadora(exemplo)
        # X é uma característica do vetor de características
        erro += (saída_desejada - u)
        #este é o erro quadrático -> 1/2 erro^2
        erro_quadrático = 0.5 * pow(erro,2)
        return erro, erro_quadrático

    # Calcula o gradiente decrescente do erro para o exemplo passado
    def calculo_gradiente(self, exemplo):
        gradiente = 0
        erro, erro_quadratico = self.erro_adaline(exemplo)
        #para cada característica do exemplo, multiplique erro * característica
        for x in exemplo:
            gradiente += erro*x
        return gradiente

    # Treina o Adaline
    def treinar_adaline(self):
        self.lista_de_pesos = self.iniciar_nova_lista_de_pesos()

        a = self.taxa_de_aprendizagem
        pronto = False
        contador =0
        limite_do_contador=len(self.lista_de_exemplos)

        # Loop de treino que só para quando os pesos não mudarem por 1 periodo
        while not pronto:
            for exemplo in self.lista_de_exemplos:
                gradiente = self.calculo_gradiente(exemplo)
                if gradiente != 0:
                    #Inicia-se uma nova lista para os pesos
                    novos_pesos = []
                    #Preenche-se a lista de pesos com o cálculo peso * calculo_gradiente
                    for peso in self.lista_de_pesos:
                        novo_peso = peso + a*gradiente
                        novos_pesos.append(novo_peso)
                #Comparação da lista de pesos
                iguais = self.comparar_lista_de_pesos(novos_pesos)
                if iguais:
                    contador += 1
                    print("Iterações sem mudança de pesos: "+ str(contador))
                    if contador == limite_do_contador:
                        pronto = True
                else:
                    print(self.lista_de_pesos)
                    contador=0



    #####################################################################################################


##Exemplo "porta N_And de 3 entradas" - Este exemplo já possui o bias aplicado
porta_nand3 = [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
## porta nand -> (x0, x1, x2, saída desejada)

##Exemplo "porta N_And de 2 entradas" - Este exemplo precisa da aplicação do Bias
porta_nand2 = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
## porta nand -> (x1, x2, saída desejada)

##Inicialização
# p = perceptron(taxa_de_aprendizagem=0.1, teta=0.5, exemplos=porta_nand3, bias=0)
p = perceptron(taxa_de_aprendizagem=0.1, teta=0.5, exemplos=porta_nand2, bias=1)
##
##Cabeçalho
print(30*"_"+"Perceptron Inicial"+31*"_")
print(p.__str__())
print(80*"_")
##
##Treinamento
p.treinar_perceptron()
# print(porta_nand)
# p.imprimir_base()
##
##Rodapé
print("\n\n"+30*"_"+"Perceptron Após Treino"+30*"_")
print(p.__str__())
print(82*"_")




