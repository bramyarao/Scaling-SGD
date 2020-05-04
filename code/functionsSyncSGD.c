// Contains the SyncSGD functions definitions
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <mpi.h>

#include"math_funcs.h"
#include"other_funcs.h"
#include"define.h"


void func_SyncSGD(unsigned int nn, unsigned int dd, unsigned int vv, unsigned int tt,
				  double*** XTrain, double** yTrain, double*** XValid, double** yValid, double*** XTest, double** yTest,
				  int noRacesValid, int noRacesTest, int** noHorseInRaceValid, int** noHorseInRaceTest, 
				  struct INPUT_PARAMETERS IP,
				  int rank, int comm_size, int root, double valid_to_stop)
{
	//******************************************
	// VARIABLES
	//******************************************
	int numWorkerCores;
	int errorCode;

	//Temp variables
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
	unsigned int big_bsize;
	double** b_XTrain; //random batch extracted from XTrain
	double* b_yTrain;  //corresponding random batch extracted from yTrain
	int* sendcounts;
	int* displs;
	double* b_XTrainVec;
	double* weightsGlobal = NULL; //Weight vector
	double* gradientsLocal = NULL; //Gradient Vector
	double* gradientsGlobal = NULL; //Gradient Vector
	double learning_rate = IP.init_learning_rate; // learning rate can be changed if needed
	double validError, testError;

	//******************************************
	// GETTING numWorkerCores
	//******************************************

	//All cores except rank 0 are workers, that is they do gradient descent
	//rank 0 will create the big random mini-batch and split it accross processes
	numWorkerCores = comm_size - 1; //All except rank 0
	
	//******************************************
	// MEMORY ALLOCATION
	//******************************************

	if (rank == root)
	{
		//Allocate memory to Batch Arrays
		big_bsize = IP.bsize * numWorkerCores;
		allocateMatrixMemory(&b_XTrain, big_bsize, dd);
		allocateVectorMemoryDouble(&b_yTrain, big_bsize);
		allocateVectorMemoryDouble(&b_XTrainVec, big_bsize * dd);
		allocateVectorMemoryDouble(&R1, dd);
		allocateVectorMemoryDouble(&R2, dd);
    allocateVectorMemoryDouble(&bestWeightsGlobalSoFar, dd);
	}
	else //Worker Processors
	{
		allocateMatrixMemory(&b_XTrain, IP.bsize, dd);
		allocateVectorMemoryDouble(&b_yTrain, IP.bsize);
		allocateVectorMemoryDouble(&b_XTrainVec, IP.bsize * dd);

		//Intermediate result arrays/matrices
		allocateMatrixMemory(&b_XTrainT, dd, IP.bsize);
		allocateMatrixMemory(&b_XTX, dd, dd); //XTrainT*XTrain  
		allocateVectorMemoryDouble(&b_XTXw, dd);
		allocateVectorMemoryDouble(&b_XTy, dd);
		allocateVectorMemoryDouble(&Lamw, dd);

		allocateVectorMemoryDouble(&R1, dd);
	}

	//scatterv:
	//Receive buffer for each worker process will be b_XTrain or b_yTrain, not required for sending process rank 0
	int buff[1] = { 0 }; //Dummy send buffer for receiving worker process in 
	sendcounts = malloc(sizeof(int) * comm_size); //No. of elements to send to each processor
	displs = malloc(sizeof(int) * comm_size);     //Starting point of elements to send to each processor
	sendcounts[0] = 0; //Rank = 0, no elements are sent
	displs[0] = 0;

	//WEIGHTS AND GRADIENTS: MEMORY ALLOCATION, INITILIZE THE WEIGHTS
	//Initialized on all processors
	allocateVectorMemoryDouble(&gradientsGlobal, dd);
	allocateVectorMemoryDouble(&gradientsLocal, dd);
	allocateVectorMemoryDouble(&weightsGlobal, dd); 

	//Initialize weights to zero
	for (i = 0; i < dd; i++)	weightsGlobal[i] = 0.0;		

	if (rank == root)
	{		
		//Initialize weightsGlobal random normal distribution
		init_weights(&weightsGlobal, dd);
	}

	//Broadcast the weightsGlobal to all the processors
	MPI_Bcast(weightsGlobal, dd, MPI_DOUBLE, root, MPI_COMM_WORLD);

	//******************************************
	// TRAINING PHASE
	//******************************************

	//---------------------------------------
	//LOOPING OVER NUMBER OF PASSES, LOOP WILL BREAK IF CONVERGENCE OF SGD IS ACHIEVED
	//---------------------------------------
	start = time(NULL);
	int is_done = 0;
	for (pass_i = 0; pass_i < IP.MAX_ITERATIONS; pass_i++)
	{
		//---------------------------------------
		// CREATING THE BIG MINI-BATCH ON MASTER (rank = 0)
		//---------------------------------------
		if (rank == root)
		{
			//The big random mini-batch will be of the size bsize*numWorkerCores, saved in b_XTrain & b_yTrain
			create_big_minibatch(&b_XTrain, &b_yTrain, big_bsize, XTrain, yTrain, nn, dd);
		}

		//---------------------------------------
		//SPLITTING yTrain
		//---------------------------------------
		for (i = 1; i < comm_size; i++) sendcounts[i] = IP.bsize;
		for (i = 1; i < comm_size; i++) displs[i] = (i - 1) * IP.bsize;

		if (rank == root)
		{
			//Splitting the b_yTrain data accross worker processors (i.e rank zero is excluded)
			//rank 0 will be sending the data 
			MPI_Scatterv(b_yTrain, sendcounts, displs, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, root, MPI_COMM_WORLD);
		}
		else
		{
			//All worker processes receive the split data in their respective b_yTrain
			MPI_Scatterv(&buff, sendcounts, displs, MPI_DOUBLE, b_yTrain, IP.bsize, MPI_DOUBLE, root, MPI_COMM_WORLD);
		}

		//---------------------------------------
		//SPLITTING XTrain
		//---------------------------------------
		//NOTE: double pointers cannot be passed as MPI requires continuous memory to copy data accross processors
		//So need to convert them to array pointers, and re-convert back to matrices after MPI scatter
		for (i = 1; i < comm_size; i++) sendcounts[i] = IP.bsize * dd;
		for (i = 1; i < comm_size; i++) displs[i] = (i - 1) * IP.bsize * dd;

		if (rank == root)
		{
			convertMatToVec(&b_XTrain, &b_XTrainVec, big_bsize, dd);
			//Splitting the b_XTrainVec data accross worker processors (i.e rank zero is excluded)
			//rank 0 will be sending the data 
			MPI_Scatterv(b_XTrainVec, sendcounts, displs, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, root, MPI_COMM_WORLD);
		}
		else
		{
			//All worker processes receive the split data in their respective b_XTrainVec
			MPI_Scatterv(&buff, sendcounts, displs, MPI_DOUBLE, b_XTrainVec, IP.bsize * dd, MPI_DOUBLE, root, MPI_COMM_WORLD);
			convertVecToMat(&b_XTrain, &b_XTrainVec, IP.bsize, dd);
		}

		//---------------------------------------
		//TRAIN MODEL: SGD BY WORKER PROCESSORS
		//---------------------------------------

		//initialize all gradients to zero
		for (i = 0; i < dd; i++)
		{
			gradientsGlobal[i] = 0.0;
			gradientsLocal[i] = 0.0;
		}
			
		//---------------------------------------
		//FIND GRADIENT ON WORKER PROCESSORS
		//---------------------------------------
		if (rank != root) 
		{
			//Compute the gradients
			// X^TXw - X^Ty + lambda*w

			transpose(&b_XTrain, IP.bsize, dd, &b_XTrainT, dd, IP.bsize);						//X^T
			matMatMul(&b_XTrainT, dd, IP.bsize, &b_XTrain, IP.bsize, dd, &b_XTX, dd, dd);		//X^TX
			matVecMul(&b_XTX, dd, dd, &weightsGlobal, dd, &b_XTXw, dd);					//X^TXw
			matVecMul(&b_XTrainT, dd, IP.bsize, &b_yTrain, IP.bsize, &b_XTy, dd);				//X^Ty
			vecScalarMul(&weightsGlobal, dd, IP.lambda, &Lamw, dd);							//lambda*w

			subTwoVectors(&b_XTXw, dd, &b_XTy, dd, &R1, dd);
			addTwoVectors(&R1, dd, &Lamw, dd, &gradientsLocal, dd);
		}

		//---------------------------------------
		//REDUCE GRADIENTS ON RANK ZERO
		//---------------------------------------
		//Takes all gradientsLocal from the worker processors and reduces them to the gradientsGlobal on the root process.
		MPI_Reduce(gradientsLocal, gradientsGlobal, dd, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

		//---------------------------------------
		//ROOT PROCESS UPDATES THE WEIGHTS AND BRAOCASTS/TERMINATES ITERATIONS
		//---------------------------------------
		// w(t) = w(t-1) - ((learning_rate/big_bsize) * gradients)
		if (rank == root)
		{
			//Re-initializing the temp vectors
			for (i = 0; i < dd; i++)
			{
			R1[i] = 0.0;
			R2[i] = 0.0;
			}

			vecScalarMul(&gradientsGlobal, dd, (learning_rate / big_bsize), &R1, dd);
			subTwoVectors(&weightsGlobal, dd, &R1, dd, &R2, dd);

			for (i = 0; i < dd; i++)	weightsGlobal[i] = R2[i]; //Update new weights

			//---------------------------------------
			//CHECK FOR CONVERGENCE:
			//---------------------------------------
			//Evaluate the average gradient for this iteration

			vecScalarMul(&gradientsGlobal, dd, (1.0 / big_bsize), &R1, dd);
			normAvgGrad = normArray(&R1, dd);
      
			// EARLY STOPPING LOGIC
			if (pass_i!=0 && pass_i % 100 == 0) //Checking validation error for every 100 iterations
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

				if (patience > IP.patience)	is_done = 1; 
				if (100.0*curValidPerf > valid_to_stop)	is_done = 1;
				start = time(NULL);
			}
		}

		MPI_Bcast(weightsGlobal, dd, MPI_DOUBLE, root, MPI_COMM_WORLD);
		MPI_Bcast(&normAvgGrad, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

		//Check for Convergence
		MPI_Bcast(&is_done, 1, MPI_INT, root, MPI_COMM_WORLD);
		if (is_done == 1) break;

	}//Loop over MAX_ITERATIONS

	if (rank == root)
	{
		// copy the best weights to weightsGlobal
		for (i = 0; i < dd; i++)  weightsGlobal[i] = bestWeightsGlobalSoFar[i]; 
		// print the summary
		curValidPerf = func_Prediction(dd, XTest, yTest, noRacesTest, noHorseInRaceTest,
										&weightsGlobal, rank, comm_size, root);
		printf("Validation: Percentage of races predicted correctly: %.2f%%\n", bestValidPerfSoFar * 100.0);
		printf("Testing: Percentage of races predicted correctly: %.2f%%\n", curValidPerf * 100.0);
	}

	//Free memory
	if (b_yTrain != NULL)		free(b_yTrain);
	if (sendcounts != NULL)		free(sendcounts);
	if (displs != NULL)			free(displs);
	if (b_XTrainVec != NULL)	free(b_XTrainVec);
	if (gradientsGlobal != NULL)	free(gradientsGlobal);
	if (gradientsLocal != NULL)	free(gradientsLocal);
	if (weightsGlobal != NULL)		free(weightsGlobal);
	if (R1 != NULL)	free(R1);
	if (bestWeightsGlobalSoFar != NULL) free(bestWeightsGlobalSoFar);

	if (rank == root)
	{
		deallocateMatrixMemory(&b_XTrain, big_bsize);
		if (R2 != NULL)	free(R2);
	}
	else //worker processors
	{
		deallocateMatrixMemory(&b_XTrain, IP.bsize);
		deallocateMatrixMemory(&b_XTrainT, dd);
		deallocateMatrixMemory(&b_XTX, dd);
		if (b_XTXw != NULL)	free(b_XTXw);
		if (b_XTy != NULL)	free(b_XTy);
		if (Lamw != NULL)	free(Lamw);
	}

}
