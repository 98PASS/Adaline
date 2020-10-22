## Critério de parada -> Certo para todas as entradas | Pesos não se alteram mais
class adaline:

    #################################### FUNÇÕES PARA INICIALIZAÇÃO ####################################

    #   Construtor
    ## A entrada deve ser passada com uma Lista de Listas, onde as listas dentro da Lista_de_Features é um EXEMPLO com sua classificação
    def __init__(self, taxa_de_aprendizagem, teta=0, amostras=[], bias=0):
        ## atribuição de constantes
        self.bias = bias  # a extensão
        self.lista_de_exemplos = amostras
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.teta = teta
        ##
        if bias != 0:
            self.colocar_bias()
        ##
        self.dimensoes = len(amostras[0]) - 1
        self.lista_de_pesos = self.iniciar_nova_lista_de_pesos()

    #  Método ToString que retorna os pesos e número de dimensões do perceptron
    def __str__(self):
        return ("Perceptron/Adaline\n" +
                "Quantia de Dimensões:\t[" + str(self.dimensoes) + "]\n" +
                "Lista de Pesos:\t" + str(self.lista_de_pesos))

    #   Iniciar todos os pesos como 0
    def iniciar_nova_lista_de_pesos(self):
        lista = []
        # tem pesos para cada característica + bias , que fica no final
        for i in range(self.dimensoes):
            lista.append(0)
        # se existir um bias, adicione o teta a lista de pesos
        if self.bias != 0:
            lista[0] = self.teta
        return lista

    #   Adiciona o bias (vetor extendido)
    def colocar_bias(self):
        for i in range(len(self.lista_de_exemplos)):
            # print("antes: "+str(self.lista_de_exemplos[i]))
            # insere na posição 0 a característica do bias
            self.lista_de_exemplos[i].insert(0, self.bias)
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
            print(50 * "-")

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
                    listas_iguais = False
                    print("Iterações sem Mudança de Pesos : " + str(contador))
                    # Se o número de vezes que a lista permaneceu inalterada for igual ao número de exemplos, então o Perceptron  atingiu seu ponto de parada
                    if contador == número_de_exemplos:
                        print("Treinamento completo, Perceptron Pronto")
                        pronto = True
                        break
                else:
                    print("Novos Pesos: " + str(self.lista_de_pesos))
                    contador = 0

    #####################################################################################################

    #### Critério de parada do ADALINE = Erro começa a subir
    ##### Significa que por limitação de hardware já passamos do erro mínimo (normalmente impossível de se obter)

    ####################################### FUNÇÕES PARA ADALINE ########################################

    #   (ADALINE) Regra Delta para atualização de Pesos com erro mínimo -> RETORNA LISTA DE PESOS e ERRO
    def lms(self, exemplo):
        novos_pesos = []
        erro, erro_quadratico = self.erro_adaline(exemplo)
        a = self.taxa_de_aprendizagem
        exp = 2 * a * erro
        somatorio = 0
        for feature in exemplo:
            somatorio += exp * feature
        for i in range(len(self.lista_de_pesos)):
            novo_peso = self.lista_de_pesos[i] + somatorio
            novos_pesos.append(novo_peso)
        return novos_pesos, erro

    # calcula o erro para o vetor de pesos (que é uma lista)
    def erro_adaline(self, exemplo):
        erro = 0
        u, saída_desejada = self.junção_somadora(exemplo)
        # X é uma característica do vetor de características
        erro += (saída_desejada - u)
        # este é o erro quadrático -> 1/2 erro^2
        erro_quadrático = 0.5 * pow(erro, 2)
        return erro, erro_quadrático

    # Treina o Adaline
    def treinar_adaline(self):
        contador =0
        limite_contador = len(self.lista_de_pesos)
        erro_minimo = None ## o erro mínimo armazena o menor erro encontrado
        pronto = False
        while not pronto:
            for exemplo in self.lista_de_exemplos:
                novos_pesos, erro = self.lms(exemplo)
                if erro_minimo is None or erro_minimo > abs(erro):
                    ## o erro mínimo só é atualizado se for encontrado um módulo de erro menor ou ele ainda estiver vazio
                    erro_minimo = abs(erro)
                    ## Se a lista for diferente, ela é atualizada e o contador é zerado
                    if self.comparar_lista_de_pesos(novos_pesos):
                        print("Novos Pesos: " + str(novos_pesos))
                    contador = 0
                else:
                    ## O CRITÉRIO DE PARADA É -> encontrar o erro perfeito = 0 OU percorrer TODA A BASE DE TREINO sem mudar a lista de pesos
                    contador+=1
                    print("Iterações sem mudanças de peso: " + str(contador) + " - Erro calculado: " + str(erro))
                    if contador == limite_contador or erro_minimo == 0:
                        pronto = True
                        break
    #####################################################################################################


##Exemplo "porta N_And de 3 entradas" - Este exemplo já possui o bias aplicado
porta_nand3 = [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
## porta nand -> (x0, x1, x2, saída desejada)

##Exemplo "porta N_And de 2 entradas" - Este exemplo precisa da aplicação do Bias
porta_nand2 = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
## porta nand -> (x1, x2, saída desejada)

##Inicialização
# p = perceptron(taxa_de_aprendizagem=0.1, teta=0.5, exemplos=porta_nand3, bias=0)
p = adaline(taxa_de_aprendizagem=0.1, teta=0.5, amostras=porta_nand3, bias=0)
##
##Cabeçalho
print(30 * "_" + "Perceptron Inicial" + 31 * "_")
print(p.__str__())
print(80 * "_")
##
##Treinamento
p.treinar_adaline()
# print(porta_nand)
# p.imprimir_base()
##
##Rodapé
print("\n\n" + 30 * "_" + "Perceptron Após Treino" + 30 * "_")
print(p.__str__())
print(82 * "_")
