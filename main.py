## Critério de parada -> Certo para todas as entradas | Pesos não se alteram mais
class perceptron:
    #   Construtor
    ## A entrada deve ser passada com uma Lista de Listas, onde as listas dentro da Lista_de_Features é um EXEMPLO com sua classificação
    def __init__(self, taxa_de_aprendizagem, teta, exemplos):
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.exemplos = exemplos
        self.lista_pesos = self.iniciar_pesos()
        self.teta = teta

    #   ToString
    def __str__(self):
        return ("Perceptron\n" +
                "Features:\t" + str(self.exemplos) + "\n" +
                "Pesos:\t" + str(self.lista_pesos))

    #   Iniciar todos os pesos como 0
    def iniciar_pesos(self):
        lista = []
        for i in range(len(self.exemplos)-1):
            lista.append(0)
        return lista

    #   Junção somadora retorna u = somatório (i ate n) de Feature[i] * Peso[i]
    ## Entrada -> Um exemplo (que é uma lista)
    def junção_somadora(self, exemplo):
        u = 0
        saída_desejada = exemplo[-1]  # extração da saída desejada
        for i in range(len(exemplo) - 1):
            u += exemplo[i] * self.lista_pesos[i]
        return u, saída_desejada

    #   Função de ativação retorna 0 ou 1 - Compara U >= Teta ? 1 , 0
    def função_de_ativação(self, u):
        if u >= self.teta:
            return 1
        else:
            return 0

    #   Calcula o erro ->(Saída desejada - Saída Obtida da função_de_ativação)
    def calcular_erro(self, saída_desejada, saída_obtida):
        erro = saída_desejada - saída_obtida
        return erro

    #   Retorna LISTA DE PESOS atualizada, pedindo como entrada da função o Erro obtido
    def atualização_dos_pesos(self, erro, exemplo):
        a = self.taxa_de_aprendizagem
        pesos_atualizados = []
        for i in range(len(self.lista_pesos)):
            peso = self.lista_pesos[i]
            entrada = exemplo[i]
            pesos_atualizados.append(peso + (a * entrada * erro))
        return pesos_atualizados

    #   Compara uma lista de pesos com a lista de pesos atual do perceptron
    ##  Se as listas forem iguais, retorna True, senão, atualiza a lista de pesos para a nova lista e retorna False
    def comparar_pesos(self, novos_pesos):
        for i in range(len(self.lista_pesos)):
            if (novos_pesos[i] != self.lista_pesos[i]):
                self.lista_pesos = novos_pesos
                return False
        return True

    #   Treina o perceptron
    def treinar(self):
        pronto = False
        contador = 0  # Conta quantas vezes a lista permaneceu inalterada
        número_de_exemplos = len(self.exemplos)
        while not pronto:  # Repita até que esteja pronto -> os pesos parem de se alterar
            for exemplo in self.exemplos:
                u, saída_desejada = self.junção_somadora(exemplo)
                saída_obtida = self.função_de_ativação(u)
                erro = self.calcular_erro(saída_desejada,saída_obtida)
                novos_pesos = self.atualização_dos_pesos(erro,exemplo)
                listas_iguais = self.comparar_pesos(novos_pesos)
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
                    print("Novos Pesos: "+str(self.lista_pesos))
                    contador = 0


##Exemplo "porta N_And de 3 entradas"
porta_nand = [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
## porta nand -> (x0, x1, x2, saída desejada)

##Inicialização
p = perceptron(0.1, 0.5, porta_nand)
##Cabeçalho
print(30*"_"+"Perceptron Inicial"+31*"_")
print(p.__str__())
print(80*"_")
##Treinamento
p.treinar()
##Rodapé
print("\n\n"+30*"_"+"Perceptron Após Treino"+30*"_")
print(p.__str__())
print(82*"_")
