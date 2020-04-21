


- [Gradient implementation from Speech and Language Processing. Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/5.pdf)
- [Leon Bottou, Stochastic Gradient Descent Tricks](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf) 

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
