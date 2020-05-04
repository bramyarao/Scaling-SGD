// Contains the any other functions definitions
#pragma warning(disable:4996)
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include"define.h"
#include"math_funcs.h"

//*************************************************************************************
//MEMORY ALLOCATION FUNCTIONS
//*************************************************************************************
//FUNCTION TO ALLOCATE MEMORY TO POINTER (VECTOR)
void allocateVectorMemoryInt(int** inputVec, int vecLen)
{
	*inputVec = NULL;
	*inputVec = (int*)malloc(vecLen * sizeof(int));
	if (*inputVec == NULL)  printf("Memory not allocated to inputVec.\n");
}

void allocateVectorMemoryDouble(double** inputVec, int vecLen)
{
	*inputVec = NULL;
	*inputVec = (double*)malloc(vecLen * sizeof(double));
	if (*inputVec == NULL)  printf("Memory not allocated to inputVec.\n");
}

//FUNCTION TO ALLOCATE MEMORY TO A DOUBLE POINTER (MATRIX)
void allocateMatrixMemory(double*** inputMat, int rows, int cols)
{
	int i;
	*inputMat = NULL;

	*inputMat = (double**)malloc(rows * sizeof(double*)); //malloc rows
	for (i = 0; i < rows; i++)
		(*inputMat)[i] = (double*)malloc(cols * sizeof(double)); //malloc cols

	if (*inputMat == NULL)  printf("Memory not allocated to inputMat.\n");
}

//FUNCTION TO DE-ALLOCATE MEMORY TO A DOUBLE POINTER (MATRIX)
void deallocateMatrixMemory(double*** inputMat, int rows) 
{
	int i;

	for (i = 0; i < rows; i++)
		free((*inputMat)[i]);
	free(*inputMat);
}

//*************************************************************************************
//READING DATA FUNCTIONS
//*************************************************************************************
//FUNCTION FOR READING IN TRAINING DATA
void read_train(double*** XTrain, double** yTrain, unsigned int* nn, unsigned int* dd, char* train_file)
{
	// open the file
	FILE* fp = fopen(train_file, "r");
	int i, j;
	int error; 

	if (fp == NULL)
	{
		printf("ERROR OPENING THE TRAIN DATA FILE.\n");
		exit(1);
	}

	// store n and d
	error = fscanf(fp, "%d,%d", nn, dd); //error = number of values read
	if (error != 2)	printf("ERROR IN READING n, d\n");

	// read all the training X's and y's
	*dd = *dd + 1; // add 1 for bias

	allocateMatrixMemory(XTrain, *nn, *dd);
	allocateVectorMemoryDouble(yTrain, *nn);

	i = 0;
	while (i < *nn) 
	{
		error = fscanf(fp, "%lf", &(*yTrain)[i]);
		if (error != 1)	printf("ERROR IN READING yTrain\n");

		// set bias (first feature)
		(*XTrain)[i][0] = 1;

		for (j = 1; j < *dd; j++)
		{
			error = fscanf(fp, "%lf", &(*XTrain)[i][j]);
			if (error != 1)	printf("ERROR IN READING XTrain\n");
		}

		i++;
	}

	// close the file
	if (fp != NULL) fclose(fp);
}


void read_test(double*** X, double** y, unsigned int* tt, int* noRaces, int** noHorseInRace, char* input_file, unsigned int dd)
{
	int iCount1, iCount2;
	int i, j;
	int error;
	int num_horses;

	// open the file
	FILE* fp = fopen(input_file, "r");

	if (fp == NULL)
	{
		printf("ERROR OPENING THE TEST/VALID DATA FILE.\n");
		exit(1);
	}

	//DO INITIAL PASS TO GET THE noRaces and tt
	*tt = 0;
	*noRaces = 0;
	allocateVectorMemoryDouble(y, dd); //temporary

	while (fscanf(fp, "%d", &num_horses) != EOF)
	{
		*tt += num_horses;
		*noRaces += 1;

		for (i = 0; i < num_horses; i++)
		{
			for (j = 0; j < dd; j++)
			{
				error = fscanf(fp, "%lf", &(*y)[j]);
				if (error != 1)	printf("ERROR IN READING TEST/VALID DATA\n");					
			}
		}
	}

	//Filling the matrices	
	free(*y);
	allocateVectorMemoryDouble(y, *tt);
	allocateMatrixMemory(X, *tt, dd);
	allocateVectorMemoryInt(noHorseInRace, *noRaces);
	iCount1 = 0;
	iCount2 = 0;

	rewind(fp);
	while (fscanf(fp, "%d", &num_horses) != EOF)
	{
		(*noHorseInRace)[iCount1] = num_horses;
		iCount1 += 1;

		for (i = 0; i < num_horses; i++)
		{		
			error = fscanf(fp, "%lf", &(*y)[iCount2]);
			if (error != 1)	printf("ERROR IN READING TEST/VALID DATA\n");

			// set bias (first feature)
			(*X)[iCount2][0] = 1;

			for (j = 1; j < dd; j++)
			{
				error = fscanf(fp, "%lf", &(*X)[iCount2][j]);
				if (error != 1)	printf("ERROR IN READING TEST/VALID DATA\n");
			}

			iCount2 += 1;
		}
	}
}

//*************************************************************************************
//TRAINING FUNCTIONS
//*************************************************************************************
//GEBERATE A RANDOM PERMUTATION FROM 0 to n-1
int* genRandomPermutation(int n)
{
	int* indices = (int*)malloc(n * sizeof(int));
	int i;
	// initialize from 0 to n-1
	for (i = 0; i < n; i++)
		*(indices + i) = i;
	// randomize the array
	int j, tmp;
	for (i = n - 1; i > 0; i--) {
		j = rand() % (i + 1);
		tmp = indices[i];
		indices[i] = indices[j];
		indices[j] = tmp;
	}
	return indices;
}

//FUNCTION THAT CREATES THE RANDOM MINI-BATCH
void create_big_minibatch(double*** b_XTrain, double** b_yTrain, int big_bsize, double*** XTrain, double** yTrain, unsigned int nn, unsigned int dd)
{
	int bi, di, cur_i;
	int* rand_idx;

	rand_idx = genRandomPermutation(nn); //Vector

	//Prepare the big batch
	for (bi = 0; bi < big_bsize; bi++)
	{
		// current example
		cur_i = rand_idx[bi];		

		// fill features
		for (di = 0; di < dd; di++) 
		{
			(*b_XTrain)[bi][di] = (*XTrain)[cur_i][di];
		}
		// fill y
		(*b_yTrain)[bi] = (*yTrain)[cur_i];
	}

}

// RANDOM NUMBER GENERATOR FOR GENERATING 1 NUMBER FROM A GAUSSIAN DISTRIBUTION
// USED TO INTIALIZE WEIGHTS
double randn(double mu, double sigma)
{
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	if (call == 1)
	{
		call = !call;
		return (mu + sigma * (double)X2);
	}

	do
	{
		U1 = -1 + ((double)rand() / RAND_MAX) * 2;
		U2 = -1 + ((double)rand() / RAND_MAX) * 2;
		W = pow(U1, 2) + pow(U2, 2);
	} while (W >= 1 || W == 0);

	mult = sqrt((-2 * log(W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double)X1);
}

// FUNCTION THAT INITALIZES A VECTOR RANDOMLY
void init_weights(double** weights, int n)
{
	int i;
	for (i = 0; i < n; i++)
		(*weights)[i] = randn(0, 0.01);
}


//*************************************************************************************
//CHECK BY PRINTING FUNCTIONS
//*************************************************************************************
//THIS FUNCTION PRINTS ANY ROW OF THE X MATRIX AND CORRESPONDING ROW OF THE y MATRIX
//JUST TO CHECK IF DATA IS SCANNED CORRECTLY FROM FILES
void checkPrinting_Input(double*** X, double** y, unsigned int rows, unsigned int cols, int rowToPrint)
{
	int i;
	int iCount;

	i = rowToPrint;
	printf("Printing row: %d\n", rowToPrint+1);
	printf("X:\n");
	for (iCount = 0; iCount < cols; iCount++)
	{
		printf("X = %.5f\n", (*X)[i][iCount]);
	}
	printf("\ny:\n");
	printf("y = %.5f\n", (*y)[i]);
}


void checkPrinting_DoubleMATRIX(double*** b_XTrain, int tempbsize, int dd, int rank, int comm_size)
{
	int i, j;
	int iCount;

	for (iCount = 0; iCount < comm_size; iCount++)
	{
		if (iCount == rank)
		{
			printf("\nmy rank: %d, no. of rows = %d, no. of cols = %d\n", rank, tempbsize, dd);
			for (i = 0; i < tempbsize; i++)
			{
				printf("\n");
				for (j = 0; j < dd; j++)
				{
					printf("%.5f ",(*b_XTrain)[i][j]);
				}
			}
		}
	}
}

void checkPrinting_DoubleVECTOR(double** arr, int n, char* text, int rank, int comm_size)
{
	int iCount;
	printf("\n\n");

	for (iCount = 0; iCount < comm_size; iCount++)
	{
		if (iCount == rank)
		{
			if (n < 1) return;
			printf("My Rank: %d, Vector length %d\n", rank,n);
			printf("%s\n%.5f", text, (*arr)[0]);
			if (n > 1)
			{
				int i;
				for (i = 1; i < n; i++)
					printf(",%.5f", (*arr)[i]);
			}
			printf("\n");
		}
	}
}


//*************************************************************************************
//MATRIX-VEC CONVERSIONS FUNCTIONS
//*************************************************************************************
void convertMatToVec(double*** b_XTrain, double** b_XTrainVec, int bsize, int dd)
{
	int i, j;
	int iCount = 0;

	for (i = 0; i < bsize; i++)
	{
		for (j = 0; j < dd; j++)
		{
			(*b_XTrainVec)[iCount] = (*b_XTrain)[i][j];  //Arranging row wise
			iCount += 1;
		}
	}

}

void convertVecToMat(double*** b_XTrain, double** b_XTrainVec, int bsize, int dd)
{
	int i, j;
	int iCount = 0;

	for (i = 0; i < bsize; i++)
	{
		for (j = 0; j < dd; j++)
		{
			(*b_XTrain)[i][j] = (*b_XTrainVec)[iCount];  //Arranging back row wise
			iCount += 1;
		}
	}
}



//PRECTION PHASE COMMON FOR ALL METHODS
double func_Prediction(unsigned int dd, 
					double*** X, double** y,
					int noRaces, int** noHorseInRace,
					double** weightsGlobal,
					int rank, int comm_size, int root)
{
	//Only root process does the evaluation
	int i, j, k;
	int counterX;
	int numHorses;

	int goldHorseID, predHorseID;
	double Gold_y, Pred_y, curGold_y, curPred_y;
	int numCorrect = 0;
	double accuracy;

	//Loop over the number of races
	counterX = 0; //Gives the row position in X or y
	for (i = 0; i < noRaces; i++)
	{
		//Taking the first horse parameters for initial step:
		goldHorseID = 0;
		predHorseID = 0;

		curGold_y = (*y)[counterX];
		Gold_y = curGold_y;
		Pred_y = 0.0;
		for (k = 0; k < dd; k++) Pred_y += (*X)[counterX][k] * (*weightsGlobal)[k];

		//Loop over the remaining horses in the race
		counterX += 1;
		numHorses = (*noHorseInRace)[i];
		for (j = 1; j < numHorses; j++)
		{
			//Update gold horse
			curGold_y = (*y)[counterX];
			if (curGold_y < Gold_y)
			{
				Gold_y = curGold_y;
				goldHorseID = j;
			}
			//Update pred horse
			curPred_y = 0.0;
			for (k = 0; k < dd; k++)	curPred_y += (*X)[counterX][k] * (*weightsGlobal)[k];
		  //printf("%lf ", curPred_y);
			if (curPred_y < Pred_y)
			{
				Pred_y = curPred_y;
				predHorseID = j;
			}
			counterX += 1;
		}
		if (goldHorseID == predHorseID) numCorrect += 1;
	}//Loop over no. of races

	accuracy = (double)numCorrect / noRaces;
	return accuracy;
}

