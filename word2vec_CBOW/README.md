## Pytorch CBOW model implementation 
This repo contains two models:
- vanilla CBOW model. Implemented from "word2vec Parameter Learning Explained" by Xin Rong paper
- CBOW + Negative Sampling model. This optimization was proposed in original Word2Vec paper "__". This implementation based on previous vanilla model.  
    Main difference :
    - dataset additionaly returns negative samples based on Unigram distribution to 3/4 power(trick from original paper) 
    - Model outputs loss
### word2Vec notebook structure
  - download, prepare data
  - train model
  - intrinsic evaluation
  - t-SNE visualization
  
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


