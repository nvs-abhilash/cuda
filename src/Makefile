# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -w

# Include directories
INCLUDES = -I/usr/include/opencv4

# Library directories and libraries
LIBS = -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Source files
SRCS = 01_rgb_to_gray.cu

# Output executable
TARGET = 01.out

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(LIBS) $(SRCS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
