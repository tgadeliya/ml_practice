# Word2Vec model 
This repo contains two pytorch models:
- vanilla CBOW model. Implemented from "___" paper
- CBOW + Negative Sampling model. This optimization was proposed in original Word2Vec paper "__". This implementation based on previous vanilla model. Main difference :
  - dataset object return context, target and generated sample based on Unigram distribution to 3/4 power(trick from original paper) 
  - Model outputs loss
  
# How to use
Code for models, training and batcher located in src/
Run jupyter notebook

## from colab
  - 1) 
  
  "!git clone https://github.com/tgadeliya/ml_practice.git
  !mv "/content/ml_practice/word2vec_CBOW/src" "/content"
  !rm -r "/content/ml_practice"
  "
# Evaluation 
Only intrinsic evaluation on 

# Structure
src -source code for model


# t-SNE visualization of the model
