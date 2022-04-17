CXX = g++
CXXFLAGS = 
INC = -I ../include
LIB = 

all = mobilenetTest
OBJS = nnlib.o


all: mobilenet.cpp ${OBJS}
	${CXX} mobilenet.cpp ${OBJS} ${CXXFLAGS} ${INC} -std=c++11 -ldl -Ofast -o mobilenetTest -lpthread -ldl
${OBJS}: ../nnlib.c
	${CXX} ../nnlib.c ${CXXFLAGS} -Wwrite-strings -ldl -Ofast -c -lpthread -ldl 
clean:
	rm -f mobilenetTest *.o *.out
