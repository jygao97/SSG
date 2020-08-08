# SSG
pytorch implementation for SSG "Set-Sequence-Graph: A Multi-View Approach Towards Exploiting Reviews for Recommendation" accepted as long paper in the research track of CIKM 2020
## Packages
- python 3.6.9
- torch 1.5.0
- numpy 1.16.1
## Run
Take running SSG on the Amazon Instrument dataset as an example
- Step1: Create a folder named "dataset" outside this folder and download the [Google pretrained word embeddings](https://code.google.com/archive/p/word2vec/). Then create a subfolder named "Instrument" and download the json file from the [Amamzon dataset](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz) into the subfolder
- Step2: python load.py ; python preprocess.py ; python get_graph.py, to preprocesses the raw dataset
- Step3: python train.py, to train our model and report its performance
