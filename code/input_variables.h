#pragma once
extern char* train_file;
extern char* valid_file;
extern char* test_file;

extern int ALGO_TYPE;
extern int root;					//Choose which process is the root, usually chosen to be zero

//INPUT PARAMETERS
extern struct INPUT_PARAMETERS IP;


//MPI variables
extern int rank;
extern int comm_size;


//MAIN Variables					

extern unsigned int nn; //Number of examples in the Training data
extern unsigned int tt; //Number of examples in the Test data
extern unsigned int vv; //Number of examples in the Validation data
extern unsigned int dd; //Number of features in the model

extern double** XTrain; //Training data X matrix, this variable is a double pointer
extern double** XValid; //Validation data X matrix, this variable is a double pointer
extern double** XTest;  //Test data X matrix, this variable is a double pointer

extern double* yTrain; //Training data y vector
extern double* yValid; //Validation data y vector
extern double* yTest;  //Test data y vector

extern int noRacesValid;
extern int noRacesTest;
extern int* noHorseInRaceValid;  //Number of horses in each race
extern int* noHorseInRaceTest;

//Time
extern time_t start;
extern time_t end;


