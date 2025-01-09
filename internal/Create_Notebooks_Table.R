#You need to be in the notebooks directory
#setwd('~/Documents/workspace/dl_rcourse_2022/notebooks')
URL_KAGGLE_prefix = 'https://kaggle.com/kernels/welcome?src='
URL_GITHUB_prefix = 'https://github.com/tensorchiefs/dl_rcourse_2022/blob/main/notebooks/'
library(tidyverse)
nbs = list.files(path = ".", pattern = NULL)
str = "| Name     | Kaggle | Git-Hub |"
str = paste(str, "|:----------------:|:-------|:------|", sep='\n')
cat(str)
for (nb in nbs){
  NB_IN_GITHUB = paste0(URL_GITHUB_prefix,nb)
  NB_IN_KAGGLE = paste0(URL_KAGGLE_prefix, NB_IN_GITHUB)
  str = paste(str, noquote(paste0('|', nb , '|', '[', nb, ']', '(', NB_IN_KAGGLE, ')', '|', '[', nb, ']', '(', NB_IN_GITHUB, ')|')), sep='\n')
}
cat(str)
