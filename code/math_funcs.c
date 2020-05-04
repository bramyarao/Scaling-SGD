// Contains the math functions definitions
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//TRANSPOSE OF A MATRIX
void transpose(double*** X, int r1, int c1, double*** R, int r2, int c2)
{
	int i, j;

	//Check for compatibility
	if ((r1 != c2) || (c1 != r2))
	{
		printf("ERROR: MATRIX TRANSPOSE SIZES NOT MATCHING !!!\n");
		return;
	}		

	for (i = 0; i < r1; i++) 
	{
		for (j = 0; j < c1; j++) 
		{
			(*R)[j][i] = (*X)[i][j];
		}
	}
}


//MATRIX MATRIX MULTIPLICATION
void matMatMul(double*** X, int r1, int c1, double*** Y, int r2, int c2, double*** R, int r3, int c3)
{
	//Compatibility checks
	if ((c1 != r2) || (r3 != r1) || (c3 != c2))
	{
		printf("MATRIX-MATRIX MULTI NOT COMPATIBLE !!!, CHECK INPUT TO FUNCTION.\n");
		return;
	}

	int i, j, k;
	for (i = 0; i < r1; i++) 
	{
		for (j = 0; j < c2; j++) 
		{
			(*R)[i][j] = 0;
			for (k = 0; k < c1; k++) 
			{
				(*R)[i][j] += (*X)[i][k] * (*Y)[k][j];
			}
		}
	}

}

//MATRIX VECTOR MULTIPLICATION
// multiply X (bxd) with y (dx1) return Xy (bx1)
void matVecMul(double*** X, int r1, int c1, double** y, int r2, double** R, int r3)
{
	// assumes the size of y to be c1
	//Check compatibility
	if ((c1 != r2) || (r3 != r1))
	{
		printf("MATRIX-VECTOR MULTI NOT COMPATIBLE !!!, CHECK INPUT TO FUNCTION.\n");
		return;
	}

	int i, j;
	for (i = 0; i < r1; i++) 
	{
		(*R)[i] = 0.0;
		for (j = 0; j < c1; j++) 
		{
			(*R)[i] += (*X)[i][j] * (*y)[j];
		}
	}

}

//VECTOR SCALAR MULTIPLICATION
void vecScalarMul(double** x, int n1, double scalar, double** R, int n2)
{
	int i;

	//check compatibility
	if (n1 != n2)
	{
		printf("VECTOR_SCALAR MULTI NOT COMPATIBLE !!!, CHECK INPUT TO FUNCTION.\n");
		return;
	}

	for (i = 0; i < n1; i++)
		(*R)[i] = (*x)[i] * scalar;
}

//VECTOR-VECTOR MULTIPLICATION
double vecVecMul(double** x, int n1, double** y, int n2)
{
	double res = 0.0;
	int i;
	for (i = 0; i < n1; i++)
		res += (*x)[i] * (*y)[i];
	return res;
}


//SUBTRACT 2 VECTORS x-y=R
void subTwoVectors(double** x, int n1, double** y, int n2, double** R, int n3)
{
	int i;

	//Check compatibility
	if ((n1 != n2) || (n3 != n1))	printf("VECTOR_VECTOR SUBTRACTION NOT COMPATIBLE !!!, CHECK INPUT TO FUNCTION.\n");

	for (i = 0; i < n1; i++)
		(*R)[i] = (*x)[i] - (*y)[i];
}

//ADD 2 VECTORS x+y=R
void addTwoVectors(double** x, int n1, double** y, int n2, double** R, int n3)
{
	int i;

	//Check compatibility
	if ((n1 != n2) || (n3 != n1))	printf("VECTOR_VECTOR ADDITION NOT COMPATIBLE !!!, CHECK INPUT TO FUNCTION.\n");

	for (i = 0; i < n1; i++)
		(*R)[i] = (*x)[i] + (*y)[i];
}


//NORM OF AN ARRAY
double normArray(double** x, int n)
{
	double res = 0.0;
	int i;

	for (i = 0; i < n; i++)
		res += (*x)[i] * (*x)[i];

	return sqrt(res);
}

