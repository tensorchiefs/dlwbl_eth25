# Some internal notes for the course
In the 2022 course in the R-Version of colab the GPU did not work. Therefore, we did switch to kaggle notebooks.

## Kaggle
The downside with kaggle NB is that they do not have TFP installed. How to install TFP is described in [tfp-with-r](https://www.kaggle.com/code/oduerr/tfp-with-r). The notebooks are in github and can be opened as descriped [here](https://www.kaggle.com/product-feedback/152480) this also shows how "direct" links to the notebooks can be generated. 

Generation of links. The URL should have the following from `https://kaggle.com/kernels/welcome?src=` + `URL_OF_NOTEBOOK`
The `URL_OF_NOTEBOOK` can be found by going to the notebook in github and from the context menu clicking "Copy Link Address" e.g. `https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/06-1d-regression.ipynb`, so a valid URL is 

[https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/06-1d-regression.ipynb](https://kaggle.com/kernels/welcome?src=https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/06-1d-regression.ipynb)

Note however, that the language might needed to be changed to R in Kaggle. This script can create the  

## Colab

### Handling notebooks
See also https://github.com/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb 

### Opening colab notebooks for students
* The notebook in github: [https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb](https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb)
Note that `blob` in the url, with can be created by browsing in github to the relevant location and pressing the `raw` button or in the context menu use `copy link address` 

* The link that opens in notebook in colab [https://colab.research.google.com/github/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb](https://colab.research.google.com/github/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb)
The link is created by appending `https://colab.research.google.com/github/` to the repository location `tensorchiefs/dl_rcourse_2022/blob/main/notebooks/00_R_Keras_TF_TFP.ipynb`.

### Editing notebooks for teacher
Currently the simplest way is to download the notebooks from colab and upload them to github. Directly checking into the reposity does not work.  
