// All global variables/constants are declared in this file
#include <stdio.h>
#include <time.h>
#include"define.h"

//*** NOTE: VALUES THAT NEED TO BE ENTERED ***
char* train_file = "data/proc/train.txt";
char* valid_file = "data/proc/valid.txt";
char* test_file = "data/proc/test.txt";
int root = 0;				//Choose which process is the root, usually chosen to be zero

//INPUT PARAMETERS - Global structure
struct INPUT_PARAMETERS IP =
{
.bsize = 5,
.MAX_ITERATIONS = 10000,
.lambda = 0.5,
.init_learning_rate = 0.01,
.toleranceConv = 100.0,
.rho = 0.8,  // elasticity parameter for EA
.communication_period = 5, // tau for EA
.patience = 100, // early stopping hyperparameter: Number of iterations valid error does not improve
.wait_for = 10 // HW parameter
};

//MPI variables
int rank, comm_size;

//MAIN Variables					

unsigned int nn; //Number of examples in the Training data
unsigned int tt; //Number of examples in the Test data
unsigned int vv; //Number of examples in the Validation data
unsigned int dd; //Number of features in the model

double** XTrain; //Training data X matrix, this variable is a double pointer
double** XValid; //Validation data X matrix, this variable is a double pointer
double** XTest;  //Test data X matrix, this variable is a double pointer

double* yTrain; //Training data y vector
double* yValid; //Validation data y vector
double* yTest;  //Test data y vector

int noRacesValid;
int noRacesTest;
int* noHorseInRaceValid;  //Number of horses in each race
int* noHorseInRaceTest;

//Time
time_t start;
time_t end;




