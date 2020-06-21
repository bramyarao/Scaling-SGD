# Scaling up Stochastic Gradient Descent using Data Parallelism
## This work is copyrighted. 

Authors: <br/>
Ramya Rao Basava and Ganesh Jawahar <br/>
Department of Computer Science, <br/>
University of British Columbia.
<br/>
<br/>
Guide: <br/>
Prof. Alan Wagner
<br/>

<div style="text-align: justify"> 

Dependencies:<br/>
[1] C <br/>
[2] MPICH <br/>
[3] Python <br/> 
[4] scikit-learn (https://scikit-learn.org/stable/) <br/>
[5] xlrd (https://pypi.org/project/xlrd/) <br/>
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

</div>


