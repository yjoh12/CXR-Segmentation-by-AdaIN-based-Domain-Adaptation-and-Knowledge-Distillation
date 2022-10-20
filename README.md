# CXR-Segmentation-by-AdaIN-based-Domain-Adaptation-and-Knowledge-Distillation
-------

<img width="100%" src="https://user-images.githubusercontent.com/47205899/179411032-909de7f8-ff95-4ed7-bf61-89c1107fd932.png"/>

# Summary
This is the Github repository for "CXR-Segmentation-by-AdaIN-based-Domain-Adaptation-and-Knowledge-Distillation", which is accepted for ECCV 2022.


-------
# Instructions
1. v641ccc2 : Initial update for training the network 
2. v8b7f5e9 : Update code for testing the network
3. (To be updated) : Update pre-trained network parameter and evaluation code


-------
# Dataset
We provide links for public CXR datasets, which are used in our paper.
These CXR datasets are available by downloading in below links.

JSRT (Normal)                 http://db.jsrt.or.jp/eng.php

SCR (Mask)                    https://www.isi.uu.nl/Research/Databases/SCR/download.php

NLM (Normal)                  http://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33

RSNA (Pneumonia)              https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226443

Cohen (Pneumonia)             https://github.com/ieee8023/covid-chestxray-dataset

BIMCV (COVID-19 Pneumonia)    http://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33

BRIXIA (COVID-19 Pneumonia)   https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/


-------
# Pre-trained model
Download the checkpoints from the below Google Drive repository to directory expr/
https://drive.google.com/drive/folders/10CbZinWVsmzOFa0KmbyzuQ3HQo7R9Y98?usp=sharing


-------
# Publication
To cite this, 

@misc{https://doi.org/10.48550/arxiv.2104.05892,
  doi = {10.48550/ARXIV.2104.05892},
  url = {https://arxiv.org/abs/2104.05892},
  author = {Oh, Yujin and Ye, Jong Chul},
  title = {Unifying domain adaptation and self-supervised learning for CXR segmentation via AdaIN-based knowledge distillation},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

https://arxiv.org/abs/2104.05892
