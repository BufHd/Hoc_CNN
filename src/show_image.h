#pragma once
#ifndef SHOW_IMAGE_H
#define SHOW_IMAGE_H

#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"
#include <vector>
#include <cstring>
#include <iostream>


void showC(Matrix show, std::string name, int height, int width, int channel);
void showC(Matrix show, int height, int width, int channel);

void showI(Matrix img, int height, int width);
void showW(Matrix w, int height, int width, int channel);

#endif // !SHOW_IMAGE_H
