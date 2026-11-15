# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -march=native -flto -std=c++17 -Wall -pthread
LDFLAGS = -pthread -flto

# Target name (final executable)
TARGET = KANKAN-15

# Source files
SRCS = KANKAN-15.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)