/*************************************************************
Implementation of our CPSC 521 - Final Project (Fall 2019)
Scaling up Stochastic Gradient Descent using Data Parallelism
Main Function
**************************************************************/

#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <mpi.h>

#include"input_variables.h"
#include"functionsSyncSGD.h"
#include"functionsASyncSGD_EA.h"
#include"functionsASyncSGD_HW.h"
#include"other_funcs.h"
#include"define.h"

// main function
main(int argc, char** argv)
{
	// process the command line arguments
	if ( argc < 3 ) {
		printf("error: incorrect inputs to the program\n");
		printf("format: mpiexec -n <num_proc> ./main <algo_num> <valid_accuracy>");
    exit(-1);
   }
  int algo_to_run;
  double valid_to_stop;
  sscanf(*(argv+1), "%d", &algo_to_run);
  sscanf(*(argv+2), "%lf", &valid_to_stop);
  
  // MPI - Initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
		
	//**********************************************
	// Read input features, labels from all datasets
	//**********************************************
	if (rank == root)
	{
		start = time(NULL);

		// Rank 0 is the main processor that reads data, allocates, collects, prints etc.
		// Getting the Training data and sizes from files
		// Read from train set: XTrain, yTrain, 'n' and 'd'
		read_train(&XTrain, &yTrain, &nn, &dd, train_file);
		// Read from validation set
		read_test(&XValid, &yValid, &vv, &noRacesValid, &noHorseInRaceValid, valid_file, dd);
		// Read from test set
		read_test(&XTest, &yTest, &tt, &noRacesTest, &noHorseInRaceTest, test_file, dd);

		end = time(NULL);
	}

	// Broadcast the value of 'n' and 'd' to all the worker processes from the root process
	MPI_Bcast(&nn, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&dd, 1, MPI_INT, root, MPI_COMM_WORLD);

	//****************************************************
	// TRAINING AND TESTING: SPLIT CODE BASED ON ALGO TYPE
	// 1 - SYNSGD
	// 2 - EASGD
	// 3 - HWSGD
	//****************************************************

	if (algo_to_run == 1)
	{
		// SYNSGD logic
		if (rank == root)
		{
			start = time(NULL);
		}

		func_SyncSGD(nn, dd, vv, tt,
					 &XTrain, &yTrain, &XValid, &yValid, &XTest, &yTest,
					 noRacesValid, noRacesTest, &noHorseInRaceValid, &noHorseInRaceTest,
					 IP,
					 rank, comm_size, root, valid_to_stop);

		if (rank == root)
		{
			end = time(NULL);
			printf("SyncSGD wall time %.2f seconds (excluding time for reading input)\n", difftime(end, start));
		}

	}
	else if(algo_to_run == 2)
	{	
		// EASGD logic
		if (rank == root)
		{
			start = time(NULL);
		}

		func_ASyncSGD_EA(nn, dd, vv, tt,
					 &XTrain, &yTrain, &XValid, &yValid, &XTest, &yTest,
					 noRacesValid, noRacesTest, &noHorseInRaceValid, &noHorseInRaceTest,
					 IP,
					 rank, comm_size, root, valid_to_stop);

		if (rank == root)
		{
			end = time(NULL);
			printf("ASyncSGD_EA wall time %.2f seconds (excluding time for reading input)\n", difftime(end, start));
		}
	}
	else if(algo_to_run == 3)
	{
		// HWSGD logic
		if (rank == root)
		{
			start = time(NULL);
		}

		func_ASyncSGD_HW(nn, dd, vv, tt,
					 &XTrain, &yTrain, &XValid, &yValid, &XTest, &yTest,
					 noRacesValid, noRacesTest, &noHorseInRaceValid, &noHorseInRaceTest,
					 IP,
					 rank, comm_size, root, valid_to_stop);

		if (rank == root)
		{
			end = time(NULL);
			printf("ASyncSGD_HW wall time %.2f seconds (excluding time for reading input)\n", difftime(end, start));
		}

	}

	// MPI - Termination
	MPI_Finalize();

  // Deallocate all the memory explicitly
	if (rank == root)
	{
		deallocateMatrixMemory(&XTrain, nn);
		if(yTrain != NULL)	free(yTrain);
		deallocateMatrixMemory(&XValid, vv);
		if (yValid != NULL)	free(yValid);
		deallocateMatrixMemory(&XTest, tt);
		if (yTest != NULL)	free(yTest);
	}
	if (noHorseInRaceValid != NULL)	free(noHorseInRaceValid);
	if (noHorseInRaceTest != NULL)	free(noHorseInRaceTest);

}


