#pragma once
#define SyncSGD     0	//0 Synchronous SGD
#define ASyncSGD_EA 1	//1 Asynchronous SGD: Elastic Averaging
#define ASyncSGD_HW 2	//2 Asynchronous SGD: Hogwild

//STRUCT DECLARATIONS
struct INPUT_PARAMETERS
{
	unsigned int bsize;			// mini-batch size for each processor
	unsigned int MAX_ITERATIONS;// number of passes over the data //TODO: need to remove this later
	double lambda;				// regularization parameter
	double init_learning_rate;	// initial value for learning rate
	double toleranceConv;
  double rho; // elasticity parameter for EA
  unsigned int communication_period; // tau for EA
  unsigned int patience; // early stopping hyperparameter
  unsigned int wait_for; // parameter for HW
};
