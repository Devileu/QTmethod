# Code for paper "Concept Drift Detection with Quadtree-based Spatial Mapping of Streaming Data"
 This code allows researchers to replicate the experiments.
 
 
### Abstract
  Online learning is a complex task, especially when the data stream changes its distribution over time. It's challenging to monitor and detect these changes to maintain the performance of the learning algorithm. In this work, we present a novel detection method built from a different perspective of other preexisting detectors from literature. It analyzes the space occupied by the data, assuming that it would be immutable unless changes in this space occur among data of different classes. The data is mapped into a quadtree-based memory structure that provides knowledge about which class (label) is dominant in a given region of the feature space. Drifts are detected by checking whether data assigned to a given class occupy spaces considered relevant to the other class. The proposed method was evaluated on benchmark binary classification problems. The results show that our method can compete with well-known drift detectors from the literature on synthetic and real-world datasets.
  
  
###  QT - Quadtree-based drift detectior

**Prerequisites**

1. Install the latest Python3 installer from [Python Downloads Page](https://www.python.org/downloads/)

2. Install dependencies

	2.1. Install [numpy](https://numpy.org/install/)

	2.2. Install [psutil](https://pypi.org/project/psutil/)

	2.3. Install [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/installation.html)
    
  
**Running QT**

Before running, open the `main_J_QT.py` file in your editor:

Choose which detectors you want to use, edit the `switch` list.

Set up the dataset file path.

Set up the dataset information (`detectiondelay`, `driftposition`, and `fullsize`). This information is presented in the `Config` file inside the `Dataset` folder.

Run `main_J_QT.py` to execute the experiments.