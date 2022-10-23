'''
Rodrigo Amador Coelho 17/05/2021

QT por densidade
Esta classe cria a quadtree.
A quantidade máxima de dados em cada node é determinada por uma variável
Isso para que as divisões aconteçam por densidade.

'''

class Node:
    # QuadTree estrutura
    # Acredita se que todos os dados vão está presentes dentro do raio ou diametro.
    def __init__(self, mid, aresta, alturamax, qtdmax):
        self.data = None             # Lista com os pontos n-dimensionais
        self.mid = mid               # centro do hypercubo
        self.aresta = aresta         # Aresta do hypercubo
        self.pai = None
        self.filhos = None
        self.alturamax = alturamax   # Altura maxima que arvore pode alcancar
        self.ndatamax = qtdmax       # Quantidade de maxima de dados no node


    # Inserir novo dado a arvore.
    def insert(self, novo):

        # Verifico condicao de altura
        if self.alturamax == 0:
            # Caso não tenha nenhum dado, cria a lista antes de inserir
            if self.data is None:
                self.data = []
                self.data.append(novo)
            else:
                self.data.append(novo)

        else:  # Caso a altura maxima não foi alcancada
            if self.filhos is not None:
                # Crio os filhos necessarios
                position = []
                newmid = []
                # Calculo a localizacao para onde o ponto sera enviado
                for k, value in enumerate(novo):
                    if value > self.mid[k]:
                        position.append('1')
                        newmid.append(self.mid[k] + (self.aresta / 2))  # Maior que mid
                    else:
                        position.append('0')
                        newmid.append(self.mid[k] - (self.aresta / 2))  # Menor que mid
                # Criar o filho e inserir o valor e os valores de mid e raio
                position = ''.join(position)  # Juntar as strings de uma lista em uma unica string
                if position not in self.filhos:
                    self.filhos.update({position: Node(newmid, self.aresta / 2, self.alturamax - 1, self.ndatamax)})
                    self.filhos[position].pai = self
                    self.filhos[position].insert(novo)  # Após criar, inserir
                else:
                    self.filhos[position].insert(novo)  # Caso já exista, inserir

            # Verifico se existe filho, caso não insiro o dado
            if self.filhos is None:
                if self.data is None:  # Caso não tenha nenhum dado, cria a lista antes de inserir
                    self.data = []
                    self.data.append(novo)
                else:
                    self.data.append(novo)

                # Criar estrutura para os filhos
                if (len(self.data) > self.ndatamax):  # Número maximo de dados por node
                    self.filhos = {}

                    # Crio os filhos necessarios
                    for i, j in enumerate(self.data):
                        position = []
                        newmid = []
                        for k, value in enumerate(j):
                            if value > self.mid[k]:
                                position.append('1')
                                newmid.append(self.mid[k] + (self.aresta / 2))  # Maior que mide
                            else:
                                position.append('0')
                                newmid.append(self.mid[k] - (self.aresta / 2))  # Menor que mid
                        # Criar o filho e inserir o valor e os valores de mid e raio
                        position = ''.join(position)  # Juntar as stringas de uma lista em uma unica string
                        #### NOVO
                        if position not in self.filhos:
                            self.filhos.update({position: Node(newmid, self.aresta / 2, self.alturamax - 1, self.ndatamax)})
                            self.filhos[position].pai = self
                            self.filhos[position].insert(j)  # Após criar, inserir
                        else:
                            self.filhos[position].insert(j)  # Caso já exista, inserir

                    self.data = None  # Os dados deste nó são apagados.


