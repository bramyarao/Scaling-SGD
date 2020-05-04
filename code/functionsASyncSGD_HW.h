#pragma once
// Contains the ASyncSGD_HW functions prototypes (declarations)

void func_ASyncSGD_HW(unsigned int nn, unsigned int dd, unsigned int vv, unsigned int tt,
          double*** XTrain, double** yTrain, double*** XValid, double** yValid, double*** XTest, double** yTest,
          int noRacesValid, int noRacesTest, int** noHorseInRaceValid, int ** noHorseInRaceTest,
          struct INPUT_PARAMETERS IP,
          int rank, int comm_size, int root, double valid_to_stop);