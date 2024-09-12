# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -w

# Include directories
INCLUDES = -I/usr/include/opencv4

# Library directories and libraries
LIBS = -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Source files
SRCS_01 = 01_rgb_to_gray.cu
SRCS_02 = 02_blur.cu

# Output executables
TARGET_01 = 01
TARGET_02 = 02

# Default target
all: $(TARGET_01) $(TARGET_02)

# Build target 01
$(TARGET_01): $(SRCS_01)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(LIBS) $(SRCS_01) -o bin/$(TARGET_01).out

# Build target 02
$(TARGET_02): $(SRCS_02)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(LIBS) $(SRCS_02) -o bin/$(TARGET_02).out

# Clean target
clean:
	rm -f $(TARGET_01) $(TARGET_02)
