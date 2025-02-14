
# Probabilistic Deep Learning

## Main Idea of the course
In this short course on deep learning (DL) focus on practical aspects of DL. We understand DL models as probabilistic models which model a (conditional) distribution for the outcome and not only a point estimation. Often this is achieved by modeling the parameters of a probability distribution and the distribution parameters are controlled by a neural network. The DL approach has the advantage, that the input to these neural networks can be all kind of data: tabular (structured) data, but also unstructured raw data like images or text. From this prespective DL models are just a complex generalisation of statistical models like linear regression. The parameters of the involved neural networks itself (often called weights) can be determined by the maximum likelihood principle. You can think of this approach as an extension of generalized linear model to complex data (e.g. images) and non-linear models providing a distribution for the outcome (like a Poisson distribution for count data or Gaussian Distribution for continious data).

The basic idea can be sketched as: 

<img src="https://raw.githubusercontent.com/tensorchiefs/dlwbl_eth25/master/images/prob_dl.png" width="80%">

## Technicalities
This course is done in python since the R support of DL is quite limited. We use a high-level API (Keras) which allows to define neural networks in a very intuitive way running on top of pytorch. The course is designed so that you can run the code in the cloud on [Colab](https://colab.research.google.com/). 

If you want you can also run the code on your computer, you need to install the required libraries (see [local_installation.md](local_installation.md)). However, we recommend to use Colab, since it is easier to set up and you can run the code on a GPU for free and can only provide limited support for local installations.



## Other resources 
We took inspiration (and sometimes slides / figures) from the following resources.

* **[Probabilistic Deep Learning](https://www.manning.com/books/probabilistic-deep-learning?a_aid=probabilistic_deep_learning&a_bid=78e55885)**  
  This book, authored by us (the tensorchiefs), explores the probabilistic approach to deep learning using Python, Keras, and TensorFlow. While we reference some content from the book, we will not cover all aspects during the course.


* **[Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition)**  
  Written by François Chollet, the creator of Keras, this book provides an in-depth introduction to deep learning, with a focus on practical implementation using Keras and TensorFlow.

* **[Keras Documentation](https://keras.io/)**  
  François Chollet initially developed Keras as an easy-to-use high-level API for building and training neural networks. Its simple interface and powerful capabilities make it a cornerstone of deep learning research and application.  
## Dates 
The course is split in 4 lectures, which excercises. You will also work on a project with data of your own choosing. For rooms see [ETHZ Course Catalogue](https://www.lehrbetrieb.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=189821&semkez=2025S&lang=en).

| Date     |      Lectures
|:--------:|:--------------|
|  03.02.2025  afternoon |   Intro to probabilistic DL and Keras with exercises 
|  10.02.2025 all day| DL with different NN architectures based on images and tabular data with exercises and project time 
|  17.02.2025 all day | Model evaluation, LLM and transformers, and project time
|  24.02.2025 morning | project presentation and uncertainty

## Syllabus 
- **Lecture 1**  
  - **Topic and Slides:** [Introduction to probabilistic deep learning](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/slides/01_Introduction.pdf)  
  - **Notebooks:**  
    - [Getting Started with Colab](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/00_colab_intro_forR.ipynb)
    - [Banknote Example 01_fcnn_with_banknotes.ipynb](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/01_fcnn_with_banknotes.ipynb)
 
  - **Optional Notebooks:**  
    - [Optional: Training with low-level liberaries](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/01_optional_training_details.ipynb)
  - **Additional Material:** [Network Playground](https://playground.tensorflow.org/)  

- **Lecture 2**
  - **Topic and Slides:** [NN training, different NN architectures (fcNN, CNN), working with pretrained NNs](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/slides/02_DL_2025.pdf)
  - **Notebooks:**  
    - [Custom loss function with NLL (02_custom_loss.ipynb)](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/02_custom_loss.ipynb)
    - [The simplest possible CNN (02_cnn_edge_lover.ipynb)](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/02_cnn_edge_lover.ipynb)
    - [Transfer Learning with CNNs (02_transfer_learning.ipynb)](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/02_transfer_learning.ipynb) 

- **Lecture 3**
  - **Topic and Slides:** [Model evaluation and transformer NNs ](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/slides/03_DL_2025.pdf)
  - **Notebooks:**  
    - [🚧 Custom loss function with NLL (03_uk_faces)](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/03_uk_faces.ipynb)
    - [Sampling from pretrained LMM 03_sampling_llm](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/notebooks/03_sampling_llm.ipynb)
    


<!-- Notebooks and slides below are  unfinished🚧
- [MNISTwithsimpleFCNN (02_nb_ch02_02a.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/02_nb_ch02_02a.ipynb) 
  - **Topic and Slides:** [Convolutional neural networks (CNNs) slides](https://github.com/tensorchiefs/dl_course_2022/blob/master/slides/02_CNN_DUMM.pdf)  
  - **Additional Material:** [Understanding convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)  
  - **Exercises and Homework:**  
    - [ArtLover_CNN (03_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/03_nb_ch02_03.ipynb)  
    - [CIFAR10_CNN (04_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb)  

- **Lecture 3**  
  - **Topic and Slides:** [CNN cntd. slides](https://github.com/tensorchiefs/dl_course_2022/blob/master/slides/03_CNN_cntd_DUMM.pdf)  
  - **Additional Material:** [Understanding convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)  
  - **Exercises and Homework:**  
    - [CIFAR10_CNN (04_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb)  
    - [DL_few_data (05_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/05-nb-ch02-03.ipynb)  

- **Lecture 4**  
  - **Topic and Slides:** [Prob. Deep Learning slides](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/slides/04_prob_dl_DUMM.pdf)  
  - **Additional Material:** [Probabilistic Deep Learning](https://www.manning.com/books/probabilistic-deep-learning?a_aid=probabilistic_deep_learning&a_bid=78e55885)  
  - **Exercises and Homework:**  
    - Notebooks 6-8 (see below)  

- **Lecture 5**  
  - **Topic and Slides:** [Semi-structured interpretable DL models slides](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/slides/05_semi_structured_DUMM.pdf)  
  - **Additional Material:** [Deep and interpretable ordinal regression models](https://arxiv.org/abs/2010.08376)  
  - **Exercises and Homework:** No notebooks   -->


## Projects
Please register your project by 11 February 2025 in the following spreadsheet: [Project Registration](https://docs.google.com/spreadsheets/d/1Ird8xaOq7sqO0Hpyh_wBFYyFvMn8jjzHQ0e7uTF15hI/edit?usp=sharing)
<!-- Friday 30 October 2022 in https://docs.google.com/spreadsheets/d/1r4pWgXwxeJ6jWNaHeOUpjRnuUSktZYKcpo3lBk7it6I/edit?usp=sharing -->


## Example Projects

Below are two example projects you could use, but it is also possible to come up with your own ideas. 

1. **Probabilistic Prediction of Temperature (Tabular Data)**
  There are serveral possibilities to make a probilistic prediction of weather.  
  The [this notebook](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/project_starters/weather/weather_pred.ipynb) contains starter code for predicting temperatures based on historic data. 
   
2. **Cuteness of Animal Images**  
   In the kaggle competition [Petfinder Pawpularity Score](https://www.kaggle.com/competitions/petfinder-pawpularity-score) you are asked to predict the popularity of a pet based on its image. This is a regression task, where the target is the popularity score. Starter code is provided in the [notebook](https://github.com/tensorchiefs/dlwbl_eth25/blob/master/project_starters/pawpularity/pawpularity_starter.ipynb).

<!-- 2. **Predit the age of bones from X-ray images**
   Dataset: [RSNA Bone Age](https://www.kaggle.com/datasets/kmader/rsna-bone-age)  

3. **Probabilistic Prediction of House Prices from House Images and Tabular Data**  
   Dataset: [House Prices and Images SoCal](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal)  
   Possible Solution: [House Prices from Images and Tabular Data](https://www.kaggle.com/code/valentinmmueller/house-prices-from-images-and-tabular-data) -->
