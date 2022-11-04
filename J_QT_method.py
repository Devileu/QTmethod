## Foram implementadas e testadas melhorias na arvore.
## Árvore suporta dados de alta dimensionalidade.
## Esse codigo está sendo usado para rodar as bases da literatura SEA_Normal e elecNormNew2
## O classificador é reinicializado no momento do Drift.
## Corrigido questão do dado de verificação ser restirado da arvore.
## Implementado Quantificação e outros classificadores.
## Regra de atualização com dados da janela

'''
Rodrigo Amador Coelho 07/07/2022

Está implementado um sistema para altura dinâmica.
Utiliza os dados da janela para estimar a altuda da QT
Classe para estimar altura
"from NovaQTreeHeight import Node as NDTheight"

Implementado QT com dados proximos 100 com 50/50 classes
Altura sendo calculada depois de 100 do noco conceito.

'''

import numpy as np
from skmultiflow.data import SEAGenerator
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
# from my_eddm import EDDM
from skmultiflow.drift_detection import PageHinkley
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNN
import os
import psutil
import time
from NovaQTree import Node as NDT
from NovaQTreeHeight import Node as NDTheight
from sklearn import metrics

# Plot Model Classes
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Dinamic height
from scipy.stats import kde  # Densidade
import math
import random

# Save results
import pickle




def MyModelPlot(X, y, myclf, model):

    h = .01  # step size in the mesh # Definicao da linha de separacao

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # we create an instance of Neighbours Classifier and fit the data.
    # clf = NearestCentroid(shrink_threshold=shrinkage)
    clf = myclf
    clf.fit(X, y)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.title(model)
    plt.axis('tight')
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 1.25)
    # plt.axvline(0.5)  # Plota linha vertical
    # plt.axhline(0.5)  # Plota linha horizontal
    plt.show()


#  cria arvore
def update_tree(ndta, ndtb, X_train, y_train):
    for i, j in enumerate(X_train):
        if y_train[i] < 0.5:
            ndta.insert(j)
        else:
            ndtb.insert(j)

    return ndta, ndtb


#  pego todos os valores da arvore
def data_tree(ndta, ndtb):
    resultadoXa = catchallNEW(ndta)
    resultadoya = [0] * len(resultadoXa)
    resultadoXb = catchallNEW(ndtb)
    resultadoyb = [1] * len(resultadoXb)
    X = resultadoXa + resultadoXb
    y = resultadoya + resultadoyb
    X = np.array(X)
    y = np.array(y)
    return X, y


# Ver todos valores para NovaQTree
def catchallNEW(tree, level = 0):
    catchdata = []
    if tree.data is None:
        level = level + 1
        if tree.filhos is not None:  # NOVO NOVO NOVO
            for i, j in enumerate(tree.filhos):
                catchdata += catchallNEW(tree.filhos[j], level)  # Todos os pontos
    else:
        # print(tree.data, level)
        datafolha = np.mean(tree.data, axis=0)
        datafolha = datafolha.tolist()
        datafolha = [datafolha]
        # print(datafolha, level, 'NEW')
        return datafolha  # Retorna o ponto medio das folhas
        # return tree.data  # Retorna os pontos de cada folha

    return catchdata


# Esta função retorna os dados da quadtree incluindo a altura da folha onde o dado está
def catchallMIDdeep(tree, level = 0):
    catchmid = []
    if tree.data is None:
        # print(level, tree.mid)
        catchmid += [[tree.mid, level]]
        level = level + 1
        for i, j in enumerate(tree.filhos):
            catchmid += catchallMIDdeep(tree.filhos[j], level)  # Todos os pontos
    elif (tree.data is not None) and (tree.filhos is None):
        catchmid += [[tree.mid, level]]
    # else:
    #     print(level, tree.mid)
    #     mid = [[tree.mid, level]]
    #     return mid  # Retorna o ponto medio das folhas
    return catchmid


# Esta função faz o cálculo da altura da QT de forma dinâmica
def myKDE_height(X, mode):
    # número de dimensões da base
    dimension = X[0].size
    val = []

    # preparando os dados para o calculo do KDE
    for i in range(dimension):
        val.append(X[:, i])

    #     # Verificar se todos dados de uma dimensão possui o mesmo valor
    #     # Importante para evitar o error "numpy.linalg.LinAlgError: singular matrix"
    #     if all(element == val[i][0] for element in val[i]) is True:
    #         val[i][0] = val[i][0] + 0.00000000000000001
    #         print(">>> SAME VALUE <<<")  # Resulta em densidade extremamente alta

    # Remove dimensão se os dados possuem o mesmo valor
    # print("Dimensão início: ", len(val))
    flag = True

    while flag:
        start = len(val)
        for i, j in enumerate(val):
            if all(element == val[i][0] for element in val[i]) is True:
                del val[i]

        if start == len(val):
            flag = False

    # print("Dimensão fim: ", len(val))
    dimension = len(val)


    # KDE
    k = kde.gaussian_kde(val)
    # KDE-PDF
    Xpdf = k.pdf(val)

    # Densidade usada para os cálculos
    densidade = max(Xpdf)

    # Calculo do S de referencia
    S = (0.16 * dimension) - 0.21
    # Calculo da densidade de referência
    pdf_ref = S * ((1 - 0.66)**dimension) * 53 + 0.21 * (0.99**dimension)
    # Cálculo da densidade
    s = (pdf_ref / densidade) ** (1 / dimension) * S
    # s = (dimension * pdf_ref / densidade)**(1/dimension) * S  ## OLD

    # Formula da altura para arvores
    D = 1
    h = math.log((math.sqrt(dimension) * D) / s, 2)

    if mode == 'f_teto':
        h = math.ceil(h)  # teto do valor de h
    else:
        h = math.floor(h)  # piso do valor de h

    return h




def QT(switch, file, detectiondelay, driftposition, fullsize):

    # Escolher qual auto_altura vai ser usado
    # switch = ['naive', 'fixo', 'arvore3', 'f_piso', 'f_teto']
    # switch = ['fixo', 'naive', 'arvore2', 'arvore3', 'f_piso', 'f_teto']


    auto_h = 'arvore2'


    detec_acc = []
    detec_acc_drift = []
    total_drift = []
    height_list = []

    for ind, detec in enumerate(switch):

        print('>> Detector ', detec, ' <<')

        # stream = FileStream("C:/Users/Devileu/Desktop/Dataset/SINE1.csv")
        stream = FileStream(file)


        # Detection delay
        # detectiondelay = [500]

        # Drift position
        # driftposition = [5000]

        # Tamanho total da base
        # fullsize = 10000



        # Adaptive Windowing method for concept drift detection
        adwin = ADWIN()
        # Drift Detection Method
        ddm = DDM()
        # Early Drift Detection Method
        eddm = EDDM()
        # Page-Hinkley method for concept drift detection
        ph = PageHinkley()

        # teste gasto de memória e tempo
        pid = os.getpid()
        ps = psutil.Process(pid)
        start = time.time()

        # Variables to control loop and track performance
        pre_training_n_samples = 200
        n_samples = 0
        correct_cnt = 0
        max_samples = stream.n_samples
        detector_window_size = 100  # Sensibilidade a mudanças
        X_win = []
        y_win = []

        # Classifier
        estimator = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=0)


        # Pre training the classifier with 200 samples
        X_train, y_train = stream.next_sample(pre_training_n_samples)
        estimator.fit(X_train, y_train)
        # estimator.partial_fit(X_train, y_train)


        # # Plot model + partial data
        # model = 'Estimador'
        # MyModelPlot(X_train, y_train, estimator, model)



        if detec == 'qt':

            # ### Altura dinâmica via arvore ###
            # Aqui implemente detecção de "alturamax"
            if auto_h == 'arvore2' or auto_h == 'arvore3':
                # print('>> Método da ', auto_h, " <<")
                height_list = []
                mid = [0.5] * X_train[0].size  # Self adapt to first data dimension
                if auto_h == 'arvore2':
                    ndtaH = NDTheight(mid, 0.5, 20, 2)  # NDTheight(meio, raio, altura maxima da arvore, quantidade dados no nó folha)
                if auto_h == 'arvore3':
                    ndtaH = NDTheight(mid, 0.5, 20, 3)  # NDTheight(meio, raio, altura maxima da arvore, quantidade dados no nó folha)
                # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                for i, j in enumerate(X_train[-100:]):
                    ndtaH.insert(j)

                midXadeep = catchallMIDdeep(ndtaH)
                alturamax = max([row[1] for row in midXadeep])
                # print(' >>> Altura determinada pelo método da Árvore: ', alturamax, '<<<')
                height_list.append(alturamax)
                # Apagar a QT criada
                del(ndtaH)
            # ### Fim Altura dinâmica via arvore ###





            # Inserting pre training data
            # Criar arvore para cada classe e inserir os dados
            # alturamax = 5
            # mid = [0.5, 0.5, 0.5]  # Para 2 dimensões
            mid = [0.5] * X_train[0].size  # Self adapt to first data dimension
            ndta = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
            ndtb = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
            # inserir os mesmos dados de treino do modeo na arvore
            for i, j in enumerate(X_train):
                if y_train[i] < 0.5:
                    ndta.insert(j)
                else:
                    ndtb.insert(j)
                # inserindo os dados na janela
                X_win.append(j)
                y_win.append(y_train[i])
                if len(X_win) > detector_window_size:
                    X_win = X_win[1:]
                    y_win = y_win[1:]

        # Convert
        X_win = np.array(X_win)
        y_win = np.array(y_win)



        # Acurácia
        acc = []
        acc_drift = []
        cont_drift = 0
        correct_cnt_drift = 0
        # Quantitativos
        drift = [10000]  # Posição em que os drifts estão localizados na base
        drift_found = []  # Posição em que os drifts foram localizados
        y_roc = []
        y_pred_roc = []

        # NOVA atualização QT após drift
        qt_cont = 0
        # Arvores das classes equilibrada 50/50
        qt_cont_a = 0
        qt_cont_b = 0
        # Atualiza altura sem drift
        falg_altura = 0

        # Run test-then-train loop for max_samples or while there is data in the stream
        while n_samples < 539383 and stream.has_more_samples():  # 59800
           n_samples += 1
           cont_drift += 1
           X, y = stream.next_sample()
           y_pred = estimator.predict(X)
           y_roc.append(y[0])
           y_pred_roc.append(y_pred[0])
           if y[0] == y_pred[0]:
               correct_cnt += 1
               correct_cnt_drift += 1
               pred_result = 0
           else:
               pred_result = 1  # miss classification
               # print('^y ERROR: ' + str(n_samples))
               X_train = np.concatenate((X_train, X), axis=0)
               y_train = np.concatenate((y_train, y), axis=0)
               estimator.fit(X_train, y_train)


           # Andamento do processo
           print('.', end="")
           if not n_samples % 100:
              print(end="\r")
              print('Streaming :', n_samples, ' ', end="")




           # Plot resultados
           acc.append(correct_cnt / n_samples)
           acc_drift.append(correct_cnt_drift / cont_drift)


           ### drift detectors ###

           # adwin results
           if detec == 'adwin':
               adwin.add_element(pred_result)
               # adwin.add_element(X)
               if adwin.detected_change():
                   # print('(adwin)Change detected in data: ' + str(n_samples))
                   # anota a posição em que aconteceu o drift
                   drift_found.append(n_samples + pre_training_n_samples)
                   # tecnica usa os dados da janela
                   cont_drift = 0
                   correct_cnt_drift = 0
                   adwin.reset()
                   X_train = np.concatenate((X_win, X), axis=0)
                   y_train = np.concatenate((y_win, y), axis=0)
                   estimator.fit(X_train, y_train)

           # ddm results
           if detec == 'ddm':
               ddm.add_element(pred_result)

               if ddm.detected_change():
                   # print('(ddm)Change has been detected in data: ' + str(n_samples), '**********')
                   # anota a posição em que aconteceu o drift
                   drift_found.append(n_samples + pre_training_n_samples)
                   # tecnica usa os dados da janela
                   cont_drift = 0
                   correct_cnt_drift = 0
                   ddm.reset()
                   X_train = np.concatenate((X_win, X), axis=0)
                   y_train = np.concatenate((y_win, y), axis=0)
                   estimator.fit(X_train, y_train)

           # eddm results
           if detec == 'eddm':
               eddm.add_element(pred_result)

               if eddm.detected_change():
                   # print('(eddm)Change has been detected in data: ' + str(n_samples), '**********')
                   # anota a posição em que aconteceu o drift
                   drift_found.append(n_samples + pre_training_n_samples)
                   # tecnica usa os dados da janela
                   cont_drift = 0
                   correct_cnt_drift = 0
                   eddm.reset()
                   X_train = np.concatenate((X_win, X), axis=0)
                   y_train = np.concatenate((y_win, y), axis=0)
                   estimator.fit(X_train, y_train)

           # Page-Hinkley results
           if detec == 'ph':
               ph.add_element(pred_result)
               # ph.add_element(X)
               if ph.detected_change():
                   # print('(PH)Change has been detected in data: ' + str(n_samples))
                   # anota a posição em que aconteceu o drift
                   drift_found.append(n_samples + pre_training_n_samples)
                   # tecnica usa os dados da janela
                   cont_drift = 0
                   correct_cnt_drift = 0
                   ph.reset()
                   X_train = np.concatenate((X_win, X), axis=0)
                   y_train = np.concatenate((y_win, y), axis=0)
                   estimator.fit(X_train, y_train)





           # Metodo da arvore
           if detec == 'qt':
               if pred_result == 1:  # miss classification
                   # Decrementar contador regra NOVA atualização QT após drift
                   if qt_cont > 0:
                       qt_cont = qt_cont - 1
                   # outlier_detector.fit(X_win, y_win)
                   # outlier_result = outlier_detector.predict(X)
                   if y[0] == y[0]:  # outlier_result == y[0]:
                       # Inserir na arvore da classe oposta e verificar o lugar que ficou
                       X_tree, y_tree = data_tree(ndta, ndtb)  # Salvando os dados da arvore
                       if y > 0.5:  # Aqui foi invertida para a arvore
                           qt_cont_a = qt_cont_a - 1  # Decremento contador do 50/50
                           ndta.insert(X[0])
                           resultadoXa = catchallNEW(ndta)
                           if resultadoXa.count(X[0].tolist()) == 0:
                               # print('(Arvore)Change has been detected in data: ' + str(n_samples), '**********')
                               # anota a posição em que aconteceu o drift
                               drift_found.append(n_samples + pre_training_n_samples)
                               #
                               cont_drift = 0
                               correct_cnt_drift = 0
                               X_train = np.concatenate((X_win, X), axis=0)
                               y_train = np.concatenate((y_win, y), axis=0)
                               estimator.fit(X_train, y_train)
                               del(ndta, ndtb)

                               # ### Altura dinâmica naive ###
                               if auto_h == 'naive':
                                   ndtaN = NDT(mid, 0.5, 20)  # NDTheight(meio, raio, altura maxima da arvore)
                                   # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                                   for i, j in enumerate(X_train[-100:]):
                                       ndtaN.insert(j)

                                   midXadeep = catchallMIDdeep(ndtaN)
                                   alturamax = max([row[1] for row in midXadeep]) - 0  ## Alterado o '-1'  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                   # print(' >>> Altura Naive: ', alturamax, '<<<')
                                   height_list.append(alturamax)
                                   # Apagar a QT criada
                                   del (ndtaN)
                               # ### Fim Altura naive ###

                               # ###### Metodo da ARVORE #####
                               if auto_h == 'arvore2' or auto_h == 'arvore3':
                                   if auto_h == 'arvore2':
                                       ndtaH = NDTheight(mid, 0.5, 20, 2)  # NDTheight(meio, raio, altura maxima da arvore)
                                   if auto_h == 'arvore3':
                                       ndtaH = NDTheight(mid, 0.5, 20, 3)  # NDTheight(meio, raio, altura maxima da arvore)
                                   # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                                   for i, j in enumerate(X_train[-100:]):
                                       ndtaH.insert(j)

                                   midXadeep = catchallMIDdeep(ndtaH)
                                   alturamax = max([row[1] for row in midXadeep]) + 0   ## Foi alterado com o '+ 1' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                   # print(' >>> Altura determinada na mudança de conceito: ', alturamax, '<<<')
                                   height_list.append(alturamax)
                                   # Apagar a QT criada
                                   del (ndtaH)
                               # ####### FIM ####

                               ##### Altura dinamica FORMULA #####
                               if auto_h == 'f_piso':
                                   alturamax = myKDE_height(X_train[-100:], auto_h)
                                   height_list.append(alturamax)
                                   # print('Altura dinâmica determinada pela função: ', alturamax, '<<<')
                               ##### FIM #####

                               ##### Altura dinamica FORMULA #####
                               if auto_h == 'f_teto':
                                   alturamax = myKDE_height(X_train[-100:], auto_h)
                                   height_list.append(alturamax)
                                   # print('Altura dinâmica determinada pela função: ', alturamax, '<<<')
                               ##### FIM #####

                               ndta = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               ndtb = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               # # inserir os mesmos dados de treino do modelo na arvore
                               # ndta, ndtb = update_tree(ndta, ndtb, X_train, y_train)
                               # NOVA regra para atualizar QT
                               qt_cont = 100
                               qt_cont_a = 50
                               qt_cont_b = 50
                               falg_altura = 1
                               ndta, ndtb = update_tree(ndta, ndtb, X, y)

                           else:
                               # print('(Arvore)Need to update estimator: ' + str(n_samples))
                               del(ndta, ndtb)
                               ndta = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               ndtb = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               ndta, ndtb = update_tree(ndta, ndtb, X_tree, y_tree)
                               # NOVA regra para atualizar QT
                               ndta, ndtb = update_tree(ndta, ndtb, X, y)
                       else:
                           qt_cont_b = qt_cont_b - 1  # Decremento contador do 50/50
                           ndtb.insert(X[0])
                           resultadoXb = catchallNEW(ndtb)
                           if resultadoXb.count(X[0].tolist()) == 0:
                               # print('(Arvore)Change has been detected in data: ' + str(n_samples), '**********')
                               # anota a posição em que aconteceu o drift
                               drift_found.append(n_samples + pre_training_n_samples)
                               #
                               cont_drift = 0
                               correct_cnt_drift = 0
                               X_train = np.concatenate((X_win, X), axis=0)
                               y_train = np.concatenate((y_win, y), axis=0)
                               estimator.fit(X_train, y_train)
                               del(ndta, ndtb)

                               # ### Altura dinâmica naive ###
                               if auto_h == 'naive':
                                   ndtaN = NDT(mid, 0.5, 20)  # NDTheight(meio, raio, altura maxima da arvore)
                                   # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                                   for i, j in enumerate(X_train[-100:]):
                                       ndtaN.insert(j)

                                   midXadeep = catchallMIDdeep(ndtaN)
                                   alturamax = max([row[1] for row in midXadeep]) - 0  ## Alterado o '-1' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                   # print(' >>> Altura Naive: ', alturamax, '<<<')
                                   height_list.append(alturamax)
                                   # Apagar a QT criada
                                   del (ndtaN)
                               # ### Fim Altura naive ###

                               ###### Metodo da ARVORE #####
                               if auto_h == 'arvore2' or auto_h == 'arvore3':
                                   if auto_h == 'arvore2':
                                       ndtaH = NDTheight(mid, 0.5, 20, 2)  # NDTheight(meio, raio, altura maxima da arvore)
                                   if auto_h == 'arvore3':
                                       ndtaH = NDTheight(mid, 0.5, 20, 3)  # NDTheight(meio, raio, altura maxima da arvore)
                                   # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                                   for i, j in enumerate(X_train[-100:]):
                                       ndtaH.insert(j)

                                   midXadeep = catchallMIDdeep(ndtaH)
                                   alturamax = max([row[1] for row in midXadeep]) + 0  # ## Alterado o '+1' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                   # print(' >>> Altura determinada na mudança de conceito: ', alturamax, '<<<')
                                   height_list.append(alturamax)
                                   # Apagar a QT criada
                                   del (ndtaH)
                               ####### FIM #####

                               ##### Altura dinamica FORMULA #####
                               if auto_h == 'f_piso':
                                   alturamax = myKDE_height(X_train[-100:], auto_h)
                                   height_list.append(alturamax)
                                   # print('Altura dinâmica determinada pela função: ', alturamax, '<<<')
                               ##### FIM #####

                               ##### Altura dinamica FORMULA #####
                               if auto_h == 'f_teto':
                                   alturamax = myKDE_height(X_train[-100:], auto_h)
                                   height_list.append(alturamax)
                                   # print('Altura dinâmica determinada pela função: ', alturamax, '<<<')
                               ##### FIM #####

                               ndta = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               ndtb = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               # # inserir os mesmos dados de treino do modelo na arvore
                               # ndta, ndtb = update_tree(ndta, ndtb, X_train, y_train)
                               # NOVA regra para atualizar QT
                               qt_cont = 100
                               qt_cont_a = 50
                               qt_cont_b = 50
                               falg_altura = 1
                               ndta, ndtb = update_tree(ndta, ndtb, X, y)

                           else:
                               # print('(Arvore)Need to update estimator: ' + str(n_samples))
                               del(ndta, ndtb)
                               ndta = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               ndtb = NDT(mid, 0.5, alturamax)  # NDT(meio, raio, altura maxima da arvore)
                               ndta, ndtb = update_tree(ndta, ndtb, X_tree, y_tree)
                               # Atualiza
                               # NOVA regra para atualizar QT
                               ndta, ndtb = update_tree(ndta, ndtb, X, y)




           # Regra para colocar os 100 dados após mudança de conceito  ATUAL +
           if qt_cont > 0:
               qt_cont = qt_cont - 1
               if (qt_cont_a > 0) and (y > 0.5):
                   qt_cont_a = qt_cont_a - 1  # Decrementa arvore 50/50
                   ndta, ndtb = update_tree(ndta, ndtb, X, y)
               if (qt_cont_b > 0) and (y < 0.5):
                   qt_cont_b = qt_cont_b - 1  # Decrementa arvore 50/50
                   ndta, ndtb = update_tree(ndta, ndtb, X, y)






           # inserindo os dados na janela
           X_win = X_win.tolist()
           y_win = y_win.tolist()
           X_win.append(X[0])
           y_win.append(y[0])
           if len(X_win) > detector_window_size:
               X_win = X_win[1:]
               y_win = y_win[1:]
           # Convert
           X_win = np.array(X_win)
           y_win = np.array(y_win)


        # Acurácia
        print(' ')
        print('Método da altura dinâmica', auto_h)
        print('{} samples analyzed.'.format(n_samples))
        print('Estimator accuracy: {}'.format(correct_cnt / n_samples))
        print('Valor da AUC: ', metrics.roc_auc_score(y_roc, y_pred_roc))

        # Qualitativos
        # print('Posições com drift:', drift)
        print('Posições que ocorreram drift:', drift_found)
        print('Quantidade de drift:', len(drift_found))

        total_drift.append(drift_found)  # Guarto dodos os drifts encontrados





        # time and memory spend
        memoryUse = ps.memory_info()
        print('Memória usada: ', memoryUse.rss/1000000, "MB usados no PID")
        end = time.time()
        print('Tempo gasto: ', end - start)

        if detec == 'qt':
            print('Lista de alturas durando o fluxo: ', height_list)

        print('    ')



        detec_acc.append(acc)
        detec_acc_drift.append(acc_drift)






    ##### Recall Precisione and F1-score #####


    ## Importante - a seguencia dos resultados segue a sequencia dos detectores que estão no início.
    _drift_found = total_drift



    # # Recall = TP/(TP + FN)
    # # Precision = TP/(TP + FP)
    _recall = []  # lista com todos os valores de reacall
    _precision = []  # lista com todos os valores de precision
    _F1 = []  # Lista com todos os valores de f1-score


    # Loop para cada detector:
    for a, ddetec in enumerate(_drift_found):
        drift_found = ddetec

        recall_l = []  # lista com os valores de reacall
        precision_l = []  # lista com os valores de precision
        F1_l = []  # Lista com os valores de f1-score


        # Para cada atraso na detecção faça
        for a, d_delay in enumerate(detectiondelay):

            TP = 0
            FN = 0
            FP = 0
            TN = 0

            inicio = 0
            fim = 0

            # Para cada posição em que existe o drift
            for i, fim in enumerate(driftposition):
                flag = 0  # Para entrar apenas uma vez nos cálculos de TP e FN
                for k, val in enumerate(drift_found):
                    if inicio <= val < fim:  # Contabilizar FN
                        FP += 1
                    if flag == 0:
                        if fim <= val < (fim + d_delay):  # Contabilizar TP e FN
                            TP += (fim + d_delay) - val
                            FN += val - fim
                            flag = 1  # Para que seja acionado apenas uma vez por loop
                    if (flag == 1) and (fim <= val < (fim + d_delay)):  #  FP no espaço de delay
                        FP += 1
                        TP -= 1
                inicio = fim + d_delay  # Atualizar posição do início
                # Caso não seja encontrado nenhum drift no intervalo definido
                if flag == 0:
                    FN += d_delay

            # Para contabilizar os últimos falos positivos após o ultimo atraso.
            fim = fullsize
            for l, val in enumerate(drift_found):
                if inicio <= val < fim:
                    FP += 1



            ## Calculo do Recall e Precision
            recall = TP/(TP + FN)

            if TP == 0:
                precision = 0
            else:
                #  Novo do artigo
                #  https://classeval.wordpress.com/simulation-analysis/roc-and-precision-recall-with-imbalanced-datasets/
                #  f(x) = (1 – exp(-αx)) / (1 -exp(-α)) where α = 7.
                TN = fullsize - (d_delay * len(driftposition)) - FP
                specificity = FP / (FP + TN)  # o nome correto é  - False Positive Rate (FPR) -
                alpha = (fullsize - (d_delay * len(driftposition))) / (d_delay * len(driftposition))  # 7 original
                fFPR = (1 - math.exp(-alpha * specificity)) / (1 - math.exp(-alpha))  # Original
                fFP = (fFPR*TN)/(1-fFPR)
                precision = TP / (TP + fFP)


            recall_l.append(recall)
            precision_l.append(precision)


            if (precision and recall) == 0:
                F1 = 0
            else:
                F1 = 2 * (precision * recall) / (precision + recall)

            F1_l.append(F1)



        _recall.append(recall_l)
        _precision.append(precision_l)
        _F1.append(F1_l)

    # Valores de Recall e Precision para plotar gráficos.
    print('Recall list = ', _recall)
    print('Precision list = ', _precision)
    print('F1-score list = ', _F1)


