#ifndef LOGGING_HPP
#define LOGGING_HPP

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <errorHandling.hpp>
#include <unistd.h>
#include <limits.h>

FILE* createLogCsv(std::string filename, std::string header);
void writeLineToFile(FILE* logfile, char* dataLine);
char* getMachineName();

#endif