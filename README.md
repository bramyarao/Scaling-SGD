This work is copyrighted. 

*************************************************************
Implementation of our CPSC 521 - Final Project (Fall 2019)
Scaling up Stochastic Gradient Descent using Data Parallelism
*************************************************************

Authors:
Ramya Rao Basava,
Department of Computer Science,
University of British Columbia,
ramyarao@cs.ubc.ca
and
Ganesh Jawahar,
Department of Computer Science,
University of British Columbia,
ganeshjw@cs.ubc.ca

Guide: Dr. Alan Wagner

Dependencies:
[1] C
[2] MPICH
[3] Python 
[4] scikit-learn (https://scikit-learn.org/stable/)
[5] xlrd (https://pypi.org/project/xlrd/)
Last three dependencies are required for preprocessing the data.

Resources:
[1] Hong Kong Horse Racing Dataset: https://www.kaggle.com/alberthkcheng/hong-kong-horse-racing-explained-with-data

How to run?
------------------------------------------------------------------------
Step-1: Preprocess the data and create train, validation and test splits
------------------------------------------------------------------------
python preprocess_data.py all 1000
format: python preprocess_data.py <features> <vocabsize>
where,
<features>: 
  correspond to the feature we need to include for the ML task,
  specifying "all" will include all the 13 features (horse + race features) along with features from summary,
  specifying "horse_number,jockey,trainer" will include the specified three features along with features from summary.
<vocabsize>:
  correspond to the top <int> words to include as features from summary.

----------------------------------------------------
Step-2a: Start training for Synchronous SGD (SYNSGD)
----------------------------------------------------
mpicc -o main *.c
mpiexec -n 4 ./main 1 10  
<= starts the training with 4 processors and 
training halts once validation accuracy of current iteration reaches 10%

----------------------------------------------------------------------------
Step-2b: Start training for Elastic Averaging based Asynchronous SGD (EASGD)
----------------------------------------------------------------------------
mpicc -o main *.c
mpiexec -n 4 ./main 2 10  
<= starts the training with 4 processors and 
training halts once validation accuracy of current iteration reaches 10%

------------------------------------------------------------------
Step-2c: Start training for HOGWILD based Asynchronous SGD (HWSGD)
------------------------------------------------------------------
mpicc -o main *.c
mpiexec -n 4 ./main 3 10  
<= starts the training with 4 processors and 
training halts once validation accuracy of current iteration reaches 10%

Misc:
[1] If you want to change hyperparameters of the parallel algorithm or ML algorithm, look at input_variables.c


