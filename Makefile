# Compiler
CC = clang++
CXX = $(CC)

# Homebrew Apple Silicon
CFLAGS = -std=c++11 -I/opt/homebrew/include/opencv4 -I./include
CXXFLAGS = $(CFLAGS)

# Library Paths
LDFLAGS = -L/opt/homebrew/lib
LDLIBS = -framework AVFoundation -framework CoreMedia -framework CoreVideo \
         -framework CoreServices -framework CoreGraphics -framework AppKit \
         -framework OpenCL -lopencv_core -lopencv_highgui -lopencv_video \
         -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc

# Directories
BINDIR = ./bin
SRCDIR = ./src
INCDIR = ./include

# Build targets
all: readfiles baseline query

readfiles: $(SRCDIR)/readfiles.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

baseline: $(SRCDIR)/baseline.o $(SRCDIR)/features.o $(SRCDIR)/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

query: $(SRCDIR)/query.o $(SRCDIR)/features.o $(SRCDIR)/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(SRCDIR)/*.o *~ 
	rm -f $(BINDIR)/*

 .PHONY: all clean