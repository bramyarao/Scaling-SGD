#pragma once
// Contains the any other functions prototypes (declarations)

//MEMORY ALLOCATION
void allocateVectorMemoryInt(int** inputVec, int vecLen);
void allocateVectorMemoryDouble(double** inputVec, int vecLen); //Valid only for double values
void allocateMatrixMemory(double*** inputMat, int rows, int cols); //Valid only for double values
void deallocateMatrixMemory(double*** inputMat, int rows);

//READING DATA
void read_train(double*** XTrain, double** yTrain, unsigned int* nn, unsigned int* dd, char* train_file);
void read_test(double*** X, double** y, unsigned int* tt, int* noRaces, int** noHorseInRace, char* input_file, unsigned int dd);

//TRAINING
int* genRandomPermutation(int n);
void create_big_minibatch(double*** b_XTrain, double** b_yTrain, int big_bsize, double*** XTrain, double** yTrain, unsigned int nn, unsigned int dd);
void init_weights(double** weights, int n);
double randn(double mu, double sigma);

//CHECK DATA ENTRY BY PRINTING
void checkPrinting_Input(double*** X, double** y, unsigned int rows, unsigned int cols, int rowToPrint);
void checkPrinting_DoubleMATRIX(double*** b_XTrain, int tempbsize, int dd, int rank, int comm_size);
void checkPrinting_DoubleVECTOR(double** arr, int n, char* text, int rank, int comm_size);

//MATRIX-VEC CONVERSIONS
void convertMatToVec(double*** b_XTrain, double** b_XTrainVec, int bsize, int dd);
void convertVecToMat(double*** b_XTrain, double** b_XTrainVec, int bsize, int dd);

//PRECTION PHASE COMMON FOR ALL METHODS
double func_Prediction(unsigned int dd,
						double*** X, double** y,
						int noRaces, int** noHorseInRace,
						double** weightsGlobal,
						int rank, int comm_size, int root);




