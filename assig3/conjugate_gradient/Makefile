# TODO
# Adjust these values, if necessary
GRID_SIZE=43

CXX=mpicc
CXXFLAGS=-O0 -xMIC-AVX512 # -qopenmp
# LDFLAGS=-qopenmp
DEFINES=-DGRID_SIZE=$(GRID_SIZE)
SOURCES=poisson_par.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=cg

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@ $(DEFINES)
	
$(SOURCES): Makefile
	touch $@

clean:
	rm *.o $(EXECUTABLE)
