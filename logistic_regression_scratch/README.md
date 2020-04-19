


Implementation https://web.stanford.edu/~jurafsky/slp3/5.pdf
Stochastic Gradient Descent Tricks
L´eon Bottou

## Project structure:
    - data/  
    - src/ - models nad auxiliary function source code 
        preprocessing
        binary_model
        multi-class model
        evaluation
    - test_function - tests for auxiliary functions 
    - test_models - tests for models
    - download.sh - BASH script to download, move and unpack training and test data
    - logreg_rs_colab.ipynb - google colab notebook
    - classify_number_MNIST.ipynb - main notebook 
    
## TODO
  - Write tests for models
  - Fix overflow in loss function(not affecting learning process, but problem with intermediate evaluation)
  - Describe models