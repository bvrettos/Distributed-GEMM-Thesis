#include <logging.hpp>

FILE* createLogCsv(std::string filename, std::string header)
{
    FILE* logfile;

    logfile = fopen(filename.c_str(), "a+");
    if (logfile == NULL) {
        exit(EXIT_FAILURE);
    }

    char* headerLine = NULL;
    size_t len = 0;
    if(getline(&headerLine, &len, logfile) != -1) {
        if (headerLine == (header + "\n")) {
            std::cout << "Header already placed, skip it" << std::endl;
        }
        else {
            fputs(header.c_str(), logfile);
            fputs("\n", logfile);
        }
    }
    else {
        std::cout << "Header not placed" << std::endl;
        fputs(header.c_str(), logfile);
        fputs("\n", logfile);
    }

    return logfile;
}

void writeLineToFile(FILE* logfile, char* dataLine)
{
    fputs(dataLine, logfile);
}

char* getMachineName()
{
    char machineName[HOST_NAME_MAX];
    size_t namelength;
    int ret = gethostname(machineName, HOST_NAME_MAX);

    if (ret) {
        return "unknown";
    }

    return machineName;
}