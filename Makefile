# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -O2 -I/usr/include/eigen3

# Target executable
TARGET = main

# Source files
SRC = prediction.cpp

# Default rule
all: $(TARGET)

# Compile
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Clean up
clean:
	rm -f $(TARGET)
