/*
 Contains the SyncSGD with Elastic Averaging functions definitions

 References:
 (1) https://papers.nips.cc/paper/5761-deep-learning-with-elastic-averaging-sgd.pdf
 (2) https://arxiv.org/pdf/1611.04581.pdf
 (3) https://joerihermans.com/ramblings/distributed-deep-learning-part-1-an-introduction/
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <mpi.h>

#include"math_funcs.h"
#include"other_funcs.h"
#include"define.h"

void func_ASyncSGD_EA(unsigned int nn, unsigned int dd, unsigned int vv, unsigned int tt,
    double*** XTrain, double** yTrain, double*** XValid, double** yValid, double*** XTest, double** yTest,
    int noRacesValid, int noRacesTest, int** noHorseInRaceValid, int** noHorseInRaceTest, 
    struct INPUT_PARAMETERS IP, int rank, int comm_size, int root, double valid_to_stop)
{
  // variables
  int numWorkerCores;
  int errorCode;

  // temp variables
  int i, j;
  int pass_i;
  int tempbsize;
  double normAvgGrad;  

  double** b_XTrainT = NULL;
  double** b_XTX = NULL;
  double* b_XTXw = NULL;
  double* b_XTy = NULL;
  double* Lamw = NULL;

  double* R1 = NULL;
  double* R2 = NULL;

  // early stopping bookkeeping
  double* bestWeightsGlobalSoFar = NULL; 
  int patience = 0;
  int bestIteration = -1;
  double bestValidPerfSoFar = -1;
  double curValidPerf = -1;
  time_t start;
  time_t end;

  //Parameters
  double** b_XTrain = NULL; //random batch extracted from XTrain
  double* b_yTrain = NULL;  //corresponding random batch extracted from yTrain
  double* b_XTrainVec = NULL;
  double* weightsGlobal = NULL; //Weight vector
  double* weightsLocal = NULL; //Weight vector
  double* interWeightsGlobal = NULL; //Weight vector
  double* gradientsLocal = NULL; //Gradient Vector
  double* gradientsGlobal = NULL; //Gradient Vector
  double* XTrain_flat = NULL;
  double* yTrain_flat = NULL;
  double learning_rate = IP.init_learning_rate; // learning rate can be changed if needed

  //******************************************
  // GETTING numWorkerCores
  //******************************************

  //All cores except rank 0 are workers, that is they do gradient descent
  //rank 0 will create the big random mini-batch and split it accross processes
  numWorkerCores = comm_size - 1; //All except rank 0
  
  //******************************************
  // MEMORY ALLOCATION
  //******************************************
  
  if (rank != root) // Worker Processors
  {
    allocateMatrixMemory(XTrain, nn, dd);
    allocateVectorMemoryDouble(yTrain, nn);
    allocateMatrixMemory(&b_XTrain, IP.bsize, dd);
    allocateVectorMemoryDouble(&b_yTrain, IP.bsize);
    allocateVectorMemoryDouble(&b_XTrainVec, IP.bsize * dd);

    // Intermediate result arrays/matrices
    allocateMatrixMemory(&b_XTrainT, dd, IP.bsize);
    allocateMatrixMemory(&b_XTX, dd, dd); // XTrainT*XTrain  
    allocateVectorMemoryDouble(&b_XTXw, dd);
    allocateVectorMemoryDouble(&b_XTy, dd);
    allocateVectorMemoryDouble(&Lamw, dd);
    allocateVectorMemoryDouble(&R1, dd);
  } 
  else
  {
    allocateVectorMemoryDouble(&bestWeightsGlobalSoFar, dd);
  } 

  allocateVectorMemoryDouble(&R1, dd);
  allocateVectorMemoryDouble(&R2, dd);
  allocateVectorMemoryDouble(&XTrain_flat, nn*dd);
  allocateVectorMemoryDouble(&yTrain_flat, nn);

  // WEIGHTS AND GRADIENTS: MEMORY ALLOCATION, INITILIZE THE WEIGHTS
  // Initialized on all processors
  allocateVectorMemoryDouble(&gradientsGlobal, dd);
  allocateVectorMemoryDouble(&gradientsLocal, dd);
  allocateVectorMemoryDouble(&weightsGlobal, dd); 
  allocateVectorMemoryDouble(&weightsLocal, dd); 
  allocateVectorMemoryDouble(&interWeightsGlobal, dd);

  //Initialize weights to zero
  for (i = 0; i < dd; i++) 
  {
    weightsGlobal[i] = 0.0; 
    weightsLocal[i] = 0.0;
  }

  if (rank == root)
  {   
    // Initialize weightsGlobal random normal distribution
    init_weights(&weightsGlobal, dd);
  }

  //******************************************
  // TRAINING PHASE
  //******************************************

  if (rank == root) 
  {
    start = time(NULL);
    // send the full dataset to all the worker processes
    for (i=0; i<nn; i++)
    {
        for (j=0; j<dd; j++)
        {
        XTrain_flat[dd*i+j] = (*XTrain)[i][j];
        }
        yTrain_flat[i] = (*yTrain)[i];
    }
    MPI_Bcast(XTrain_flat, nn*dd, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(yTrain_flat, nn, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Bcast(weightsGlobal, dd, MPI_DOUBLE, root, MPI_COMM_WORLD);

    for (pass_i = 0; pass_i < IP.MAX_ITERATIONS/IP.communication_period; pass_i++)
    {      
      // Broadcast the weightsGlobal to all the processors
      MPI_Bcast(weightsGlobal, dd, MPI_DOUBLE, root, MPI_COMM_WORLD);

      // Takes all weightsLocal from the worker processors and reduces them to the intermediate weightsGlobal on the root process.
      MPI_Reduce(weightsLocal, interWeightsGlobal, dd, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

      //Perform update for weightsGlobal
      //Re-initializing the temp vectors
      for (i = 0; i < dd; i++)
      {
        R1[i] = 0.0;
        R2[i] = 0.0;
      }
      for (i = 0; i < dd; i++)
      {
        weightsGlobal[i] = interWeightsGlobal[i]/numWorkerCores ; // R1[i] + R2[i];
      } 

      //EARLY STOPPING LOGIC
      if (pass_i !=0 /*&& pass_i % 100 == 0*/) //Checking Validation error after everytime master updates values
      {
        end = time(NULL);
        curValidPerf = func_Prediction(dd,  XValid, yValid, noRacesValid, noHorseInRaceValid,
                    &weightsGlobal, rank, comm_size, root);
        if (curValidPerf > bestValidPerfSoFar) 
        {
            bestValidPerfSoFar = curValidPerf;
            patience = 0;
            bestIteration = pass_i;
            //store best weights found so far
            for (i = 0; i < dd; i++)  bestWeightsGlobalSoFar[i] = weightsGlobal[i]; 
        }
        else
        {
            patience += 1;
        }

        MPI_Bcast(&patience, 1, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(&curValidPerf, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        if ((patience > IP.patience) || (100.0 * curValidPerf > valid_to_stop))
        {
            // copy the best weights to weightsGlobal
            for (i = 0; i < dd; i++)  weightsGlobal[i] = bestWeightsGlobalSoFar[i];

            // print the summary
            curValidPerf = func_Prediction(dd, XTest, yTest, noRacesTest, noHorseInRaceTest,
                &weightsGlobal, rank, comm_size, root);
            printf("Validation: Percentage of races predicted correctly: %.2f%%\n", bestValidPerfSoFar * 100.0);
            printf("Testing: Percentage of races predicted correctly: %.2f%%\n", curValidPerf * 100.0);
            break;
        }
        start = time(NULL);
      }

    }

  }
  else
  {
    MPI_Bcast(XTrain_flat, nn*dd, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(yTrain_flat, nn, MPI_DOUBLE, root, MPI_COMM_WORLD);
    for (i=0; i<nn; i++)
    {
      for (j=0; j<dd; j++)
      {
         (*XTrain)[i][j] = XTrain_flat[dd*i+j];
      }
      (*yTrain)[i] = yTrain_flat[i];
    }

    MPI_Bcast(weightsLocal, dd, MPI_DOUBLE, root, MPI_COMM_WORLD);

    //int count = 0;
    for (pass_i = 0; pass_i < IP.MAX_ITERATIONS; pass_i++)
    {
      if (pass_i % IP.communication_period == 0) 
      {
        // Receives the weightsGlobal from the master processor
        MPI_Bcast(weightsGlobal, dd, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        // update the local weights
        for (i=0; i<dd; i++)
        {
          double tmp = weightsLocal[i];
          // w = w - learning_rate * rho * (theta_t^i - theta_t^c)
          weightsLocal[i] = weightsLocal[i] - ( learning_rate * IP.rho * (weightsLocal[i] - weightsGlobal[i]));
          // prepare learning_rate * rho * (theta_t^i - theta_t^c) to be sent to master
          weightsGlobal[i] = learning_rate * IP.rho * (weightsLocal[i] - weightsGlobal[i]);
          //weightsGlobal[i] = tmp; 
        }

        // send  learning_rate * rho * (theta_t^i - theta_t^c) to master
        MPI_Reduce(weightsGlobal, interWeightsGlobal, dd, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

        // EARLY STOPPING LOGIC
        if (pass_i != 0 /*&& pass_i % 100 == 0*/) //Checking Validation error after everytime master updates values
        {
            MPI_Bcast(&patience, 1, MPI_INT, root, MPI_COMM_WORLD);
            MPI_Bcast(&curValidPerf, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

            if ((patience > IP.patience) || (100.0 * curValidPerf > valid_to_stop))
            {
                break;
            }
        }

      }

      // create a batch randomly
      create_big_minibatch(&b_XTrain, &b_yTrain, IP.bsize, XTrain, yTrain, nn, dd);

      // update weightsLocal based on the batch
      // Compute the gradients
      // X^TXw - X^Ty + lambda*w
      transpose(&b_XTrain, IP.bsize, dd, &b_XTrainT, dd, IP.bsize);           //X^T
      matMatMul(&b_XTrainT, dd, IP.bsize, &b_XTrain, IP.bsize, dd, &b_XTX, dd, dd);   //X^TX
      matVecMul(&b_XTX, dd, dd, &weightsLocal, dd, &b_XTXw, dd);         //X^TXw
      matVecMul(&b_XTrainT, dd, IP.bsize, &b_yTrain, IP.bsize, &b_XTy, dd);       //X^Ty
      vecScalarMul(&weightsLocal, dd, IP.lambda, &Lamw, dd);             //lambda*w
      subTwoVectors(&b_XTXw, dd, &b_XTy, dd, &R1, dd);
      addTwoVectors(&R1, dd, &Lamw, dd, &gradientsLocal, dd);
      // w = w - lr/bsize * grads
      vecScalarMul(&gradientsLocal, dd, (learning_rate / IP.bsize), &R1, dd);
      subTwoVectors(&weightsLocal, dd, &R1, dd, &R2, dd);
      for (i = 0; i < dd; i++)  weightsLocal[i] = R2[i]; //Update new weights
    }

  }

  // Free all the allocated memory
  if (b_yTrain != NULL)   free(b_yTrain);
  if (b_XTrainVec != NULL)  free(b_XTrainVec);
  if (gradientsGlobal != NULL)  free(gradientsGlobal);
  if (gradientsLocal != NULL) free(gradientsLocal);
  if (weightsGlobal != NULL)    free(weightsGlobal);
  if (weightsLocal != NULL)    free(weightsLocal);
  if (interWeightsGlobal != NULL) free(interWeightsGlobal);
  if (R1 != NULL) free(R1);
  if (R2 != NULL) free(R2);
  if (b_XTrain != NULL) deallocateMatrixMemory(&b_XTrain, IP.bsize);
  if (b_XTrainT != NULL) deallocateMatrixMemory(&b_XTrainT, dd);
  if (b_XTX != NULL) deallocateMatrixMemory(&b_XTX, dd);
  if (b_XTXw != NULL) free(b_XTXw);
  if (b_XTy != NULL)  free(b_XTy);
  if (Lamw != NULL) free(Lamw);
  if (bestWeightsGlobalSoFar != NULL) free(bestWeightsGlobalSoFar);

}



