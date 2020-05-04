#pragma once
// Contains the math functions prototypes (declarations)

//LINEAR ALGEBRA: MATRIX, VECTOR OPERATIONS
void transpose(double*** X, int r, int c, double*** R, int r2, int c2);
void matMatMul(double*** X, int r1, int c1, double*** Y, int r2, int c2, double*** R, int r3, int c3);
void matVecMul(double*** X, int r1, int c1, double** y, int r2, double** R, int r3);
void vecScalarMul(double** x, int n1, double scalar, double** R, int n2);
double vecVecMul(double** x, int n1, double** y, int n2);
void subTwoVectors(double** x, int n1, double** y, int n2, double** R, int n3);
void addTwoVectors(double** x, int n1, double** y, int n2, double** R, int n3);

//NORMS
double normArray(double** x, int n);