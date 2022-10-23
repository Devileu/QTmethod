## Foram implementadas e testadas melhorias na arvore.
## Árvore suporta dados de alta dimensionalidade.
## Esse codigo está sendo usado para rodar as bases da literatura SEA_Normal e elecNormNew2
## O classificador é reinicializado no momento do Drift.
## Corrigido questão do dado de verificação ser restirado da arvore.
## Implementado Quantificação e outros classificadores.
## Regra de atualização com dados da janela


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
# from my_eddm import EDDM   # Tirar esse comentário para habilitar o EDDM modificador
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W
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

    # h = math.ceil(h)  #  teto do valor de h
    # h = round(h)  #  Arredonda para o inteiro mais proximo
    # h = math.floor(h)  #  piso do valor de h

    if mode == 'f_teto':
        h = math.ceil(h)  # teto do valor de h
    else:
        h = math.floor(h)  # piso do valor de h

    return h




# Escolher o valor do parâmetro rho

# rho = ['naive', 'arvore2', 'arvore3']  # rho value
auto_h = 'arvore2'


# Ecolher qual detector vai ser usado

# switch = ['qt', adwin', 'ddm', 'eddm', 'ph', 'hddm_a', 'hddm_w']  # Atenção com a versão do EDDM
switch = ['ddm', 'qt']

detec_acc = []
detec_acc_drift = []
total_drift = []

for ind, detec in enumerate(switch):

    print('>> Detector ', detec, ' <<')

    # Setup the File Stream   # my4data20k2  elecNormNew3  SEA_Normal  SINE1_drift_5k   SINE2_drift_5k   RBFG_5k_5k.csv USENET1
    # Gradual: SINE_G1R_5k_1000   SINE_G2R_5k_1000  myGradual myGradual14k   RotatingHyperplane
    stream = FileStream("C:/Users/devil/Desktop/DOC/BKup_14_07_22/Jounal/2022_junho/Bases/SINE2_drift_5k.csv")


    # Tamanho do intervalo de detecção
    # detection delay
    detectiondelay = [500]

    # Posição em que o Drift acontece
    # Drift position
    driftposition = [5000]

    # Tamanho total da base
    fullsize = 10000



    # stream = SEAGenerator(classification_function = 2,
    #                       random_state = 112,
    #                       balance_classes = False,
    #                       noise_percentage = 0.28)

    # stream = LEDGeneratorDrift(random_state = 112,
    #                            noise_percentage = 0.28,
    #                            has_noise = False,
    #                            n_drift_features= 6)

    # stream = STAGGERGenerator(classification_function = 2,
    #                           random_state = 112,
    #                           balance_classes = False)

    # For high dimension data tests
    # stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=2, n_features=400, n_centroids=100)  # n_features=500 tem drift
    # stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=2, n_features=200, n_centroids=10)

    # # # For Gradual drift
    # stream = ConceptDriftStream(stream=STAGGERGenerator(classification_function = 0, random_state = 112, balance_classes = False),
    #                             drift_stream=STAGGERGenerator(classification_function = 2, random_state = 112, balance_classes = False),
    #                             position=1200,
    #                             width=500,
    #                             random_state=None,
    #                             alpha=0.0)

    # stream = ConceptDriftStream(stream=STAGGERGenerator(classification_function = 2, random_state = 112, balance_classes = False),
    #                             drift_stream=RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=2, n_features=3, n_centroids=5),
    #                             position=1200,
    #                             width=400,
    #                             random_state=None,
    #                             alpha=0.0)


    # Prepare stream for use
    stream.prepare_for_use()

    # Adaptive Windowing method for concept drift detection
    adwin = ADWIN()  # Original 0.002
    # Drift Detection Method
    ddm = DDM()  # original 3.0
    # Early Drift Detection Method
    eddm = EDDM()
    # Page-Hinkley method for concept drift detection
    ph = PageHinkley()  # Original 50
    # Drift Detection Method based on Hoeffding’s bounds with moving average-test.
    hddm_a = HDDM_A()  # drift_confidence=0.001
    # Drift Detection Method based on Hoeffding’s bounds with moving weighted average-test.
    hddm_w = HDDM_W()  # drift_confidence=0.001 original


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

    # Setup the desired estimator
    n_neighbors = 5
    # estimator = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    # estimator = GaussianNB()  # Não tem apresentado bons resultados
    # estimator = SVC(C=1.0, kernel='linear', degree=3, gamma='scale')  # kernel='rbf' ou 'linear'
    # estimator = SVC(gamma=2, C=51, kernel='rbf')  # Configuração da SVM para a base Luas_Circulo.
    # estimator = GaussianProcessClassifier(1.0 * RBF(1.0))  # Funciona bem
    # estimator = KNN(n_neighbors=5, max_window_size=20000, leaf_size=1)
    # estimator = SVC(C=1.0, kernel='linear')  # Parametros paper: Concept Drift Detection and Adaptation with
    # estimator = NaiveBayes(nominal_attributes=None)
    estimator = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=0)


    # Pre training the classifier with 200 samples
    X_train, y_train = stream.next_sample(pre_training_n_samples)
    estimator.fit(X_train, y_train)  # GaussianNB() estimator AND neighbors.KNeighborsClassifier()
    # estimator.partial_fit(X_train, y_train)


    # # Plot model + partial data
    # model = 'Estimador'
    # MyModelPlot(X_train, y_train, estimator, model)

    if detec == 'qt':
        # ### Altura dinâmica naive ###
        if auto_h == 'naive':
            print('>> Método naive <<')
            height_list = []
            mid = [0.5] * X_train[0].size  # Self adapt to first data dimension
            ndtaN = NDT(mid, 0.5, 20)  # NDTheight(meio, raio, altura maxima da arvore)
            # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
            for i, j in enumerate(X_train[-100:]):
                ndtaN.insert(j)

            midXadeep = catchallMIDdeep(ndtaN)
            alturamax = max([row[1] for row in midXadeep])
            print(' >>> Altura Naive: ', alturamax, '<<<')
            height_list.append(alturamax)
            # Apagar a QT criada
            del(ndtaN)
        # ### Fim Altura naive ###



        # ### Altura FIXO ###
        if auto_h == 'fixo':
            print('>> Altura FIXA <<')
            alturamax = 4
            height_list = []
            height_list.append(alturamax)
        # ### Fim FIXO ###



        # ### Altura dinâmica via arvore ###
        # Aqui implemente detecção de "alturamax"
        if auto_h == 'arvore2' or auto_h == 'arvore3':
            print('>> Método da ', auto_h, " <<")
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
            print(' >>> Altura determinada pelo método da Árvore: ', alturamax, '<<<')
            height_list.append(alturamax)
            # Apagar a QT criada
            del(ndtaH)
        # ### Fim Altura dinâmica via arvore ###



        ### Altura dinamica Formula ###
        # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
        if auto_h == 'f_piso':
            print('>> Método da Formula Piso<<')
            height_list = []
            alturamax = myKDE_height(X_train[-100:], auto_h)  # usar X_train[:100] chess
            height_list.append(alturamax)
            print('Altura dinâmica determinada pela função: ', alturamax, '<<<')
        ### FIM Altura dinamica funcao ###



        ### Altura dinamica Formula ###
        # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
        if auto_h == 'f_teto':
            print('>> Método da Formula Teto<<')
            height_list = []
            alturamax = myKDE_height(X_train[-100:], auto_h)  # usar  X_train[:100] chess
            height_list.append(alturamax)
            print('Altura dinâmica determinada pela função: ', alturamax, '<<<')
        ### FIM Altura dinamica funcao ###





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

        # # Convert
        # X_win = np.array(X_win)
        # y_win = np.array(y_win)


    # Convert
    X_win = np.array(X_win)
    y_win = np.array(y_win)


    # ## teste ##
    # len(X_train)
    # len(X_win)
    # resultadoXa = catchallNEW(ndta)
    # resultadoya = [0] * len(resultadoXa)
    # resultadoXb = catchallNEW(ndtb)
    # resultadoyb = [1] * len(resultadoXb)
    # X = resultadoXa + resultadoXb
    # y = resultadoya + resultadoyb
    # len(X)
    # # Plot model + partial data
    # X = np.array(X)
    # y = np.array(y)
    # estimator.fit(X, y)
    # model = 'Arvore'
    # MyModelPlot(X, y, estimator, model)
    # ## FIM ##

    # # KNN outlier detector
    # n_neighbors = 5
    # outlier_detector = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    # outlier_detector.fit(X_win, y_win)

    # # Plot model + partial data
    # model = 'Outlier'
    # MyModelPlot(X_win, y_win, outlier_detector, model)

    # Acurácia
    acc = []
    acc_drift = []
    cont_drift = 0
    correct_cnt_drift = 0
    y_roc = []
    y_pred_roc = []

    # Quantitativos
    # drift = [10000]  # Posição em que os drifts estão localizados na base
    drift_found = []  # Posição em que os drifts foram localizados

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
           # Treino incremental
           X_train = np.concatenate((X_train, X), axis=0)
           y_train = np.concatenate((y_train, y), axis=0)
           estimator.fit(X_train, y_train)

       # Andamento do processo
       print('.', end="")
       if not n_samples % 100:
           print(end="\r")
           print('Streaming :', n_samples, ' ', end="")


       # ## TESTE
       # # plotar o modelo no momento antes do drift
       # if n_samples == 3000:
       #    model = 'Estimador SVC linear Arvore h=4  DRIFT 1'
       #    MyModelPlot(X_win, y_win, estimator, model)
       #
       # if n_samples == 8000:
       #    model = 'Estimador SVC linear Arvore h=4  DRIFT 2'
       #    MyModelPlot(X_win, y_win, estimator, model)
       #
       # if n_samples == 13000:
       #    model = 'Estimador SVC linear Arvore h=4  DRIFT 3'
       #    MyModelPlot(X_win, y_win, estimator, model)
       #
       # if n_samples == 17000:
       #    model = 'Estimador SVC linear Arvore h=4  DRIFT 4'
       #    MyModelPlot(X_win, y_win, estimator, model)
       # ## FIM TESTE


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
           # if ddm.detected_warning_zone():
               # print('(ddm)Warning zone has been detected in data: ' + str(n_samples))
               # X_train = np.concatenate((X_train, X), axis=0)
               # y_train = np.concatenate((y_train, y), axis=0)
               # estimator.fit(X_train, y_train)
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
           # if eddm.detected_warning_zone():
               # print('(eddm)Warning zone has been detected in data: ' + str(n_samples))
               # X_train = np.concatenate((X_train, X), axis=0)
               # y_train = np.concatenate((y_train, y), axis=0)
               # estimator.fit(X_train, y_train)
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




       # hddm_a results
       if detec == 'hddm_a':
           hddm_a.add_element(pred_result)
           if hddm_a.detected_change():
               # print('(hddm_a)Change has been detected in data: ' + str(n_samples), '**********')
               # anota a posição em que aconteceu o drift
               drift_found.append(n_samples + pre_training_n_samples)
               # tecnica usa os dados da janela
               cont_drift = 0
               correct_cnt_drift = 0
               hddm_a.reset()
               X_train = np.concatenate((X_win, X), axis=0)
               y_train = np.concatenate((y_win, y), axis=0)
               estimator.fit(X_train, y_train)



       # hddm_w results
       if detec == 'hddm_w':
           hddm_w.add_element(pred_result)
           if hddm_w.detected_change():
               # print('(hddm_a)Change has been detected in data: ' + str(n_samples), '**********')
               # anota a posição em que aconteceu o drift
               drift_found.append(n_samples + pre_training_n_samples)
               # tecnica usa os dados da janela
               cont_drift = 0
               correct_cnt_drift = 0
               hddm_w.reset()
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
                           del (ndta, ndtb)

                           # ### Altura dinâmica naive ###
                           if auto_h == 'naive':
                               ndtaN = NDT(mid, 0.5, 20)  # NDTheight(meio, raio, altura maxima da arvore)
                               # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                               for i, j in enumerate(X_train[-100:]):
                                   ndtaN.insert(j)

                               midXadeep = catchallMIDdeep(ndtaN)
                               alturamax = max([row[1] for row in
                                                midXadeep]) - 0  ## Alterado o '-1'  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                               alturamax = max([row[1] for row in
                                                midXadeep]) + 0  ## Foi alterado com o '+ 1' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                           del (ndta, ndtb)
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
                           del (ndta, ndtb)

                           # ### Altura dinâmica naive ###
                           if auto_h == 'naive':
                               ndtaN = NDT(mid, 0.5, 20)  # NDTheight(meio, raio, altura maxima da arvore)
                               # utilizo os ultimos 100 elementos do vetor "X_train[-100:]"
                               for i, j in enumerate(X_train[-100:]):
                                   ndtaN.insert(j)

                               midXadeep = catchallMIDdeep(ndtaN)
                               alturamax = max([row[1] for row in
                                                midXadeep]) - 0  ## Alterado o '-1' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                               alturamax = max([row[1] for row in
                                                midXadeep]) + 0  # ## Alterado o '+1' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                           del (ndta, ndtb)
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


    # Acirácia
    print('{} samples analyzed.'.format(n_samples))
    print('Estimator accuracy: {}'.format(correct_cnt / n_samples))
    print('Valor da AUC: ', metrics.roc_auc_score(y_roc, y_pred_roc))

    # Qualitativos
    TP = []  # Detectado dentro do intervalo
    FN = []  # Não detectado dentro
    FP = []  # Detectado fora do internvalo
    win_verification = 100
    # print('Posições com drift:', drift)
    print('Posições que ocorreram drift:', drift_found)
    print('Quantidade de drift:', len(drift_found))

    total_drift.append(drift_found)  # Guarto dodos os drifts encontrados





    # time and memory spend
    memoryUse = ps.memory_info()
    print('Memória usada: ', memoryUse.rss/1000000, "MB usados no PID")
    end = time.time()
    print('Tempo gasto: ', end - start)

    # # Plotar histórico de acuracia do modelo
    # plt.plot(acc)
    # plt.plot(acc_drift)
    # # for i, j in enumerate(drift):
    # #     plt.axvline((j - (i * 100) - 200), color = 'b')  # Plota linha vertical
    # # Plota linha vertical que aponta drift
    # # for i, j in enumerate(drift_found):
    # #     plt.axvline((j - pre_training_n_samples), color = 'r')  # Plota linha vertical
    # plt.ylabel('Acuracia Geral')
    # plt.show()

    detec_acc.append(acc)
    detec_acc_drift.append(acc_drift)



# plot results

# # Plotar histórico de acuracia do modelo
# plt.plot(detec_acc[0])
# plt.plot(detec_acc_drift[1])
# # for i, j in enumerate(drift):
# #     plt.axvline((j - (i * 100) - 200), color = 'b')  # Plota linha vertical
# # Plota linha vertical que aponta drift
# # for i, j in enumerate(drift_found):
# #     plt.axvline((j - pre_training_n_samples), color = 'r')  # Plota linha vertical
# plt.ylabel('Acuracia Geral')
# plt.show()



# # Save variables to plot in future
# save_data = [detec_acc, detec_acc_drift]
# pickle.dump(save_data, open("C:/Users/Avell/Desktop/Jounal/2022_julho/NB_D6_weather_Norm.dat", "wb"))


# # Load pickle data to plot results
# # Manter comentado. Descomantar e redar o trecho final do codigo para plotar os resultados
# load_data = pickle.load(open("C:/Users/Avell/Desktop/Jounal/2022_junho/D1_air_norm.dat", "rb"))
# detec_acc = load_data[0]
# detec_acc_drift = load_data[1]
#
#
# # ################################
# # Atualizar o arquivo de resultados da base
# # Carrego antigo, apago o ultimo, carrego o novo final, junto os resultados e salvo
#
# load_data = pickle.load(open("C:/Users/Avell/Desktop/Jounal/2022_junho/Daw_air_norm.dat", "rb"))
# detec_accX = load_data[0]
# detec_acc_driftX = load_data[1]
#
# # Unindo os resultados
# detec_acc.append(detec_accX[0])
# detec_acc_drift.append(detec_acc_driftX[0])
# detec_acc.append(detec_accX[1])
# detec_acc_drift.append(detec_acc_driftX[1])
# detec_acc.append(detec_accX[2])
# detec_acc_drift.append(detec_acc_driftX[2])
# detec_acc.append(detec_accX[3])
# detec_acc_drift.append(detec_acc_driftX[3])
# detec_acc.append(detec_accX[4])
# detec_acc_drift.append(detec_acc_driftX[4])
# detec_acc.append(detec_accX[5])
# detec_acc_drift.append(detec_acc_driftX[5])
#
#
#
#
# # Cria valores para o eixo x do plot
# # O valore de 200 representa os dados usados para pretreino.
# xline_l = list(range(200, len(detec_acc[0]) + 200))
#
#
# # Plot results
# plt.plot(xline_l, detec_acc[0], 'k',
#          xline_l, detec_acc[1], 'b',
#          xline_l, detec_acc[2], 'g',
#          xline_l, detec_acc[3], 'r-',
#          xline_l, detec_acc[4], 'c-',
#          xline_l, detec_acc[5], 'm:',
#          xline_l, detec_acc[6], 'y:')
# plt.ylim(ymin=0.5, ymax=1.0)  # this line
# plt.legend(['QT', 'ADWIN', 'DDM', 'EDDM', 'PH', 'HDDMa', 'HDDMw'], loc='upper right')  # loc='upper lower right'
# plt.grid(True)
# plt.show()
#
#
# plt.plot(xline_l, detec_acc_drift[0], 'k',
#          xline_l, detec_acc_drift[1], 'b',
#          xline_l, detec_acc_drift[2], 'g',
#          xline_l, detec_acc_drift[3], 'r-',
#          xline_l, detec_acc_drift[4], 'c-',
#          xline_l, detec_acc_drift[5], 'm:',
#          xline_l, detec_acc_drift[6], 'y:')
# plt.ylim(ymax=1.008)  # this line
# plt.legend(['QT', 'ADWIN', 'DDM', 'EDDM', 'PH', 'HDDMa', 'HDDMw'], loc='lower right')
# plt.grid(True)
# plt.show()














##### Recall Precisione and F1-score #####


## Importante - a seguencia dos resultados segue a sequencia dos detectores que estão no início.
_drift_found = total_drift


# Loop para cada detector:
for a, ddetec in enumerate(_drift_found):
    drift_found = ddetec
    # print(drift_found)

# # Tamanho do intervalo de detecção
# # detection delay
# detectiondelay = [500]
# # detectiondelay = [100]
#
# # Posição em que o Drift acontece
# # Drift position
# driftposition = [5000]
#
# # Tamanho total da base
# fullsize = 10000


# TP = []  # Detectado dentro do intervalo
# FN = []  # Não detectado dentro
# FP = []  # Detectado fora do internvalo

# Recall = TP/(TP + FN)
# Precision = TP/(TP + FP)
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


        # print('TP = ', TP)
        # print('FN = ', FN)
        # print('FP = ', FP)

        ## Calculo do Recall e Precision
        recall = TP/(TP + FN)

        if TP == 0:
            precision = 0
        else:
            # Equação original
            # precision = TP / (TP + FP) # Formula original

            # Equação modificada
            # precision = TP/(TP + (FP * ((fullsize/(len(driftposition) + 1) - d_delay))/d_delay))  # Original

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

        # print('Recall = ', recall)
        # print('Precision = ', precision)

        # Calculo do F1-score
        # F1 = 2 * (precision * recall) / (precision + recall)
        if (precision and recall) == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        F1_l.append(F1)

        # print('F1-score', F1)

    # # Valores de Recall e Precision para plotar gráficos.
    # print('Recall list = ', recall_l)
    # print('Precision list = ', precision_l)
    # print('F1-score list = ', F1_l)

    _recall.append(recall_l)
    _precision.append(precision_l)
    _F1.append(F1_l)

# Valores de Recall e Precision para plotar gráficos.
print('')
print('Method list', switch)
print('Recall list = ', _recall)
print('Precision list = ', _precision)
print('F1-score list = ', _F1)




