# Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall
SRC = main.cpp linearRegression.cpp 
OBJ = $(SRC:.cpp=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
