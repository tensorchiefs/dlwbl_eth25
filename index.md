
# Deep Learning with R

## Main Idea of the course
In this short course on deep learning (DL) with R we focus on practical aspects of DL. We understand DL models as probabilistic models which model a (conditional) distribution for the outcome and not only a point estimation. Often this is achieved by modeling the parameters of a probability distribution and the distribution parameters are controlled by a neural network. The DL approach has the advantage, that the input to these neural networks can be all kind of data: tabular (structured) data, but also unstructured raw data like images or text. From this prespective DL models are just a complex generalisation of statistical models like linear regression. The parameters of the involved neural networks itself (often called weights) can be determined by the maximum likelihood principle. The basic idea can be sketched as: 

<img src="https://github.com/tensorchiefs/dl_rcourse_2022/raw/main/ch05_00_opener.jpg" width="40%">

## Technicalities
This course is done in R. The used DL libraries are Keras, Tensorflow, and Tensorflow Probability. You can run the code in the cloud (Kaggle / Colab) or on you computer. For 2022, we support the Kaggle approach (Colab does currently not support GPUs from R).

### Kaggle Notebooks
For doing the hands-on part you can use kaggle notebooks, which allow you to use a GPU. Go to [kaggle.com](kaggle.com) and register. After your first login, you need to do a phone verfication in order to be able to use GPU acceleration and internet (click to your symbol on thr right, account, phone verification). You can directly open the kaggle notebooks from the links provided at the end of this webpage. On the right you have a 3-dots dropdown, where you can select acceleration with GPU, if needed for that notebook.
If you want to open NB from an github account, you can select "code" on the menu (left), new notebook, import notebook via github (cat symbol) (e.g. from `tensorchiefs/dl_rcourse_2022`).  

### Colab Notebooks
You also can use use Colab (you need a google account) and an internet connections. However, at the moment (12 Sep 2022) GPU is not accessible for R-notebooks. An empty notebook for R can be started by clicking the following link 
[https://colab.research.google.com/notebook#create=true&language=r](https://colab.research.google.com/notebook#create=true&language=r) (starts a colab notebook with R) 
An example notebook, which installs the required DL-Liberies is  [00_R_Keras_TF_TFP] (https://colab.research.google.com/github/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb). 

### Local Installation
If you want to do it without Internet connection, on your own computer you can try to install Tensorflow, Keras, and (for the latter lessons) TensorFlow Probability. Check if the notebook [00_R_Keras_TF_TFP](https://colab.research.google.com/github/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb) is running.   


## Other resources 
We took inspiration (and sometimes slides / figures) from the following resources.

* Probabilistic Deep Learning (DL-Book) [Probabilistic Deep Learning](https://www.manning.com/books/probabilistic-deep-learning?a_aid=probabilistic_deep_learning&a_bid=78e55885). This book is by us, the tensorchiefs, and covers the probabilistic approach to deep learning unsing Python, Keras, and TensorFlow. We will not cover all aspects of this book during the course.  

* Deep Learning with R [https://tensorflow.rstudio.com/](https://tensorflow.rstudio.com/). Nice resource with DL tutorials in R.

* R Markdown Scripts for DL in R [https://github.com/fmmattioni/deep-learning-with-r-notebooks](https://github.com/fmmattioni/deep-learning-with-r-notebooks) from the author of the Book [Deep Learning with R](https://www.manning.com/books/deep-learning-with-r)

* Deep Learning Book (DL-Book) [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/). This is a quite comprehensive book which goes far beyond the scope of this course. 

* Convolutional Neural Networks for Visual Recognition [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/), has additional material and [youtube videos of the lectures](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC). While the focus is on computer vision, it also treats other topics such as optimization, backpropagation and RNNs. Lecture notes can be found at [http://cs231n.github.io/](http://cs231n.github.io/).

* Another applied course in DL: [TensorFlow and Deep Learning without a PhD](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)

* A further more applied course in python can be found at [https://tensorchiefs.github.io/dl_course_2022/](https://tensorchiefs.github.io/dl_course_2022/). 

## Dates 
The course is split in 5 lectures, which excercises. You will also work on a project with data of your own choosing. 

| Date     |      Lectures
|:--------:|:--------------|
|  12.09.2022| Lecture 1 (10:15-12:00), Lecture 2 (14:15-16:00) and Exercises (16:15-18:00)
|  19.09.2022| Lecture 3 (14:15-16:00) and Exercises (16:15-18:00)
|  26.09.2022| Lecture 4 (14:15-16:00) and Exercises (16:15-18:00)
|  03.10.2022| Hands-on DL: This day is reserved to work on your projects on site
|  10.10.2022| Lecture 5 and presentatation of the projects



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


| Name     | topic |Kaggle | Git-Hub |
|:----------------:|:------------------------:|:-------|:------|
|00_R_Keras_TF_TFP.ipynb|Check Keras, TF, TFP|[00_R_Keras_TF_TFP.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb)|[00_R_Keras_TF_TFP.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb)|
|01_dl-r-banknote.ipynb|Banknote Classification with fcNN |[01_dl-r-banknote.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/01_dl-r-banknote.ipynb)|[01_dl-r-banknote.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/01_dl-r-banknote.ipynb)|
|02_nb_ch02_02a.ipynb| MNIST classification with fcNN|[02_nb_ch02_02a.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/02_nb_ch02_02a.ipynb)|[02_nb_ch02_02a.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/02_nb_ch02_02a.ipynb)|
|03_nb_ch02_03.ipynb|CNN Art lover|[03_nb_ch02_03.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/03_nb_ch02_03.ipynb)|[03_nb_ch02_03.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/03_nb_ch02_03.ipynb)|
|04_nb_ch02_03.ipynb|CNN Cifar10|[04_nb_ch02_03.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb)|[04_nb_ch02_03.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/04_nb_ch02_03.ipynb)|
|05-nb-ch02-03.ipynb|Finetuning, pretrained CNNs on CIFAR10|[05-nb-ch02-03.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/05-nb-ch02-03.ipynb)|[05-nb-ch02-03.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/05-nb-ch02-03.ipynb)|
|06-into-tfp.ipynb|Intro to TFP|[06-into-tfp.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/06-into-tfp.ipynb)|[06-into-tfp.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/06-into-tfp.ipynb)|
|07-utkface.ipynb|Age CNN: TFP for continuos regression|[07-utkface.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/07-utkface.ipynb)|[07-utkface.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/07-utkface.ipynb)|
|08-countdata.ipynb|Fish fcNN: TFP for count regression|[08-countdata.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/08-countdata.ipynb)|[08-countdata.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/08-countdata.ipynb)|

## Projects
Please register your project by Friday 30 October 2022 in https://docs.google.com/spreadsheets/d/1r4pWgXwxeJ6jWNaHeOUpjRnuUSktZYKcpo3lBk7it6I/edit?usp=sharing


