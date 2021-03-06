# Scaling up Stochastic Gradient Descent using Data Parallelism: Application to Horse racing prediction using Machine Learning
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

## Introduction
<div style="text-align: justify"> 
In this project, the winner of the Hong Kong Horse Racing is predicted using Machine Learning (ML). In addition to the standard horse and race features, the most frequent words from the match summary for each race are also used as input features. Least squares regression with L2 regularization is used. Given that this is a parallel computing project, apart from trying to get a good accuracy for the ML problem, one of the main aims of the project was to reduce the computation time by parallelizing the Stochastic Gradient Descent (SGD) step. For this, three
different approaches were compared as listed below:
<ul style="list-style-type:disc;"> 
<li> Synchronous SGD </li>
<li> Elastic Averaging based Asynchronous SGD </li>
<li> HOGWILD based Asynchronous SGD (based on distributed memory, not shared memory) </li>
</ul>

<p align="center">
<img width="400" src="1500f_10v_def.jpg">
</p>

The overview of the communication structure for the different methods, data processing, the hyperparameters used for different algorithms, MPI parallel environment calls and the results obtained are given in <a href="Report/main.pdf" target="blank">this pdf</a>.
</div>

<br/>
Dependencies:<br/>
[1] C <br/>
[2] MPICH <br/>
[3] Python <br/>
[4] <a href="https://scikit-learn.org/stable/" target="blank">scikit-learn</a> <br/>
[5] <a href="https://pypi.org/project/xlrd/" target="blank">xlrd</a> <br/>
Last three dependencies are required for preprocessing the data.
<br/>
<br/>
Resources: <br/>
[1] <a href="https://www.kaggle.com/alberthkcheng/hong-kong-horse-racing-explained-with-data" target="blank">Hong Kong Horse Racing Dataset</a> <br/>


## How to run?

### Step-1: Preprocess the data and create train, validation and test splits
<div style="text-align: justify">
Sample run: python preprocess_data.py all 1000 <br/>
Format: python preprocess_data.py &lt;features&gt; &lt;vocabsize&gt; <br/>
where, <br/> <br/>
</div>
 
<div style="text-align: justify">
&lt;features&gt;:
<ul style="list-style-type:disc;"> 
  <li>correspond to the feature we need to include for the ML task </li>
  <li>specifying "all" will include all the 13 features (horse + race features) along with features from summary </li>
  <li>specifying "horse_number, jockey, trainer" will include the specified three features along with features from summary. </li>
</ul>
</div> 

<div style="text-align: justify">
&lt;vocabsize&gt;:
<ul style="list-style-type:disc;">
  <li>correspond to the top &lt;int&gt; words to include as features from summary. </li>
</ul> 
</div> <br/>

### Step-2a: Start training for Synchronous SGD (SYNSGD)
<div style="text-align: justify">
mpicc -o main *.c <br/>
mpiexec -n 4 ./main 1 10   <br/>
 <br/>
The above command starts the training with 4 processors, using SYNSGD algorithm which has input code of 1 and training halts once validation accuracy of current iteration reaches 10%.
</div> <br/>

### Step-2b: Start training for Elastic Averaging based Asynchronous SGD (EASGD)
<div style="text-align: justify">
mpicc -o main *.c <br/>
mpiexec -n 4 ./main 2 10   <br/>
 <br/>
The above command starts the training with 4 processors, using EASGD algorithm which has input code of 2 and training halts once validation accuracy of current iteration reaches 10%.
</div> <br/>

### Step-2c: Start training for HOGWILD based Asynchronous SGD (HWSGD)
<div style="text-align: justify">
mpicc -o main *.c <br/>
mpiexec -n 4 ./main 3 10   <br/>
 <br/>
The above command starts the training with 4 processors, using HWSGD algorithm which has input code of 3 and training halts once validation accuracy of current iteration reaches 10%.
</div> <br/>

### Misc:
<div style="text-align: justify">
[1] If you want to change hyperparameters of the parallel algorithm or ML algorithm, look at input_variables.c
</div>




