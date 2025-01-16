
# Deep Learning with R

## Main Idea of the course
In this short course on deep learning (DL) focus on practical aspects of DL. We understand DL models as probabilistic models which model a (conditional) distribution for the outcome and not only a point estimation. Often this is achieved by modeling the parameters of a probability distribution and the distribution parameters are controlled by a neural network. The DL approach has the advantage, that the input to these neural networks can be all kind of data: tabular (structured) data, but also unstructured raw data like images or text. From this prespective DL models are just a complex generalisation of statistical models like linear regression. The parameters of the involved neural networks itself (often called weights) can be determined by the maximum likelihood principle. You can think of this approach as an extension of generalized linear model to complex data (e.g. images) and non-linear models providing a distribution for the outcome (like a Poisson distribution for count data or Gaussian Distribution for continious data).

The basic idea can be sketched as: 

<img src="https://raw.githubusercontent.com/tensorchiefs/dlwbl_eth25/master/images/prob_dl.png" width="80%">

## Technicalities
This course is done in python since the R support of DL is very limited and unstable. We use a high-level API (Keras) which allows to define neural networks in a very intuitive way running on top of pytorch. The course is designed that you can run the code in the cloud on Colab. If you want you can also run the code on your computer, you need to install the required libraries (see local_installation.md). However, we recommend to use Colab, since it is easier to set up and you can run the code on a GPU for free and can only provide limited support for local installations.

### Colab Notebooks
To use Colab (you need a google account) and an internet connections.

## Other resources 
We took inspiration (and sometimes slides / figures) from the following resources.

* Probabilistic Deep Learning (DL-Book) [Probabilistic Deep Learning](https://www.manning.com/books/probabilistic-deep-learning?a_aid=probabilistic_deep_learning&a_bid=78e55885). This book is by us, the tensorchiefs, and covers the probabilistic approach to deep learning unsing Python, Keras, and TensorFlow. We will not cover all aspects of this book during the course.  

* Deep Learning Book (DL-Book) [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/). This is a quite comprehensive book which goes far beyond the scope of this course. 

* Convolutional Neural Networks for Visual Recognition [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/), has additional material and [youtube videos of the lectures](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC). While the focus is on computer vision, it also treats other topics such as optimization, backpropagation and RNNs. Lecture notes can be found at [http://cs231n.github.io/](http://cs231n.github.io/).

* Another applied course in DL: [TensorFlow and Deep Learning without a PhD](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)

* A further more applied course in python can be found at [https://tensorchiefs.github.io/dl_course_2022/](https://tensorchiefs.github.io/dl_course_2022/). 

## Dates 
The course is split in 5 lectures, which excercises. You will also work on a project with data of your own choosing. 

| Date     |      Lectures
|:--------:|:--------------|
|  03.02.2025 | Lecture 1   and Exercises 
|  10.02.2025 | Lecture 2,3 Exercises, Projects
|  17.02.2025 | Lecture 4   Exercises, Projects
|  24.02.2025 | Lecture 5 and presentatation of the projects

## Syllabus (might change during course) 

| Lecture  |      Topic and Slides    |      Additional Material    |		Exercises and homework  |
|:----------------:|:-----------------------|:----------------------------|:--------------------------------------|
| 1        | Introduction, Fully Connected Networks, Keras [slides](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/slides/01_Introduction.pdf) |[Network Playground](https://playground.tensorflow.org/) |[Banknoteexample (01_dl-r-banknote.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/01_nb_ch02_01.ipynb) <br>[MNISTwithsimpleFCNN (02_nb_ch02_02a.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/02_nb_ch02_02a.ipynb)
| 2        |Convolutional neural networks (CNNs) [slides](https://github.com/tensorchiefs/dl_course_2022/blob/master/slides/02_CNN.pdf) |[Understanding convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)|[ArtLover_CNN(03_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/03_nb_ch02_03.ipynb)  <br> [CIFAR10_CNN(04_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb)
| 3        |CNN cntd. [slides](https://github.com/tensorchiefs/dl_course_2022/blob/master/slides/03_CNN_cntd.pdf) |[Understanding convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)|   [CIFAR10_CNN(04_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb) <br> [DL_few_data(05_nb_ch02_03.ipynb)](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/05-nb-ch02-03.ipynb)
| 4        |Prob. Deep Learning. [slides](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/slides/04_prob_dl.pdf) |[Probabilistic Deep Learning](https://www.manning.com/books/probabilistic-deep-learning?a_aid=probabilistic_deep_learning&a_bid=78e55885)| Notebooks 6-8 (see below)
| 5        |Semi-structured interpretable DL models [slides](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/slides/05_semi_structured.pdf) |[Deep and interpretable ordinal regression models](https://arxiv.org/abs/2010.08376)| no notebooks <br>


## Links to the notebooks
<!-- The links are created with the script internal/Create_Notebooks_Table.R -->
The link in the Kaggle Column opens the Notebook in Kaggle. You might need to change the language from to python to R (File --> Language)

| Name                 | Topic                                    | Git-Hub                                                                                                    |
|:---------------------:|:----------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
| 00_R_Keras_TF_TFP.ipynb | Check Keras, TF, TFP                    | [00_R_Keras_TF_TFP.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb) |
| 01_dl-r-banknote.ipynb | Banknote Classification with fcNN       | [01_dl-r-banknote.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/01_dl-r-banknote.ipynb)   |
| 02_nb_ch02_02a.ipynb   | MNIST classification with fcNN          | [02_nb_ch02_02a.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/02_nb_ch02_02a.ipynb)       |
| 03_nb_ch02_03.ipynb    | CNN Art lover                           | [03_nb_ch02_03.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/03_nb_ch02_03.ipynb)         |
| 04_nb_ch02_03.ipynb    | CNN Cifar10                             | [04_nb_ch02_03.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb)         |
| 05-nb-ch02-03.ipynb    | Finetuning, pretrained CNNs on CIFAR10  | [05-nb-ch02-03.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/05-nb-ch02-03.ipynb)         |
| 06-into-tfp.ipynb      | Intro to TFP                            | [06-into-tfp.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/06-into-tfp.ipynb)             |
| 07-utkface.ipynb       | Age CNN: TFP for continuos regression   | [07-utkface.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/07-utkface.ipynb)               |
| 08-countdata.ipynb     | Fish fcNN: TFP for count regression     | [08-countdata.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/08-countdata.ipynb)           |

## Projects
Please register your project by TODO
<!-- Friday 30 October 2022 in https://docs.google.com/spreadsheets/d/1r4pWgXwxeJ6jWNaHeOUpjRnuUSktZYKcpo3lBk7it6I/edit?usp=sharing -->


