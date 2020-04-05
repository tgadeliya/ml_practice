## Pytorch CBOW model implementation 
This repo contains two models:
- **vanilla CBOW model**. Implemented from "word2vec Parameter Learning Explained" by Xin Rong paper

- **CBOW + Negative Sampling model**. This optimization was proposed in original Word2Vec paper "__". Implementation based on previous vanilla model.  
    Main difference :
    - dataset additionaly returns negative samples based on (Unigram distribution)^3/4 (trick from original paper) 
    - Model outputs normalized over batch loss.
### word2Vec notebook structure
  - download, prepare data
  - create pytorch dataset and dataloader
  - train model
  - intrinsic evaluation
  - t-SNE visualization

### Data
 Used text8 file from http://mattmahoney.net/dc/textdata.html. This is Wikipedia cleaned text. 
 
### How to use
Code for models, training and batcher located in src/. 
Just run jupyter notebook

#### colab
   1) Go to colab : Upload Notebook / Github and paste link to notebook
       https://github.com/tgadeliya/ml_practice/blob/master/word2vec_CBOW/word2vec.ipynb
   2) Execute code below somewhere in the colab notebook. This will copy folder with source code to /content directory
       ```bash
       !git clone https://github.com/tgadeliya/ml_practice.git
       !mv "/content/ml_practice/word2vec_CBOW/src" "/content"
       !rm -r "/content/ml_practice"
       ```


