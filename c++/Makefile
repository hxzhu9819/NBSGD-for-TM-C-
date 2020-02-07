CC=g++ -g -Wall

SOURCES=forward.cpp

OBJS=${SOURCES:.cpp=.o}

all: app

app: ${OBJS}
	${CC} -o app ${OBJS}

%.o: %.cpp
	${CC} -c $<

clean:
	rm -rf *.o
