CXX = g++
CXXFLAGS = -Wwrite-strings -Wunused-result -std=c++11 -ldl -Ofast -lpthread -ldl
INC = -I ../include
LIB = 
OBJS = nnlib.o


mnistTest: main.c ${OBJS}
	${CXX} ${CXXFLAGS} ${INC} -o $@ $^
${OBJS}: ../nnlib.c
	${CXX} -c $< ${CXXFLAGS} 
clean:
	rm -f mnistTest *.o *.out