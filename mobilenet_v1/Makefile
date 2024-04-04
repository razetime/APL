CXX = g++
CXXFLAGS = -Wwrite-strings -Wunused-result -std=c++11 -ldl -Ofast -lpthread -ldl
INC = -I ../include
LIB = 
OBJS = nnlib.o


mobilenetTest: mobilenet.cpp ${OBJS}
	LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ ${CXX} ${CXXFLAGS} ${INC} -o $@ $^ -lOpenCL
${OBJS}: ../nnlib.c
	LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ ${CXX} -c $< ${CXXFLAGS} 
clean:
	rm -f mobilenetTest *.o *.out
