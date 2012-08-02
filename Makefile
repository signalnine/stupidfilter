# Thirdparty directory
TP = thirdparty
LIBSVM = $(TP)/libsvm/
# Location of boost library headers
BOOST = -I$(TP)
# Path to binary boost serialization library must be defined here
BOOSTLIB = "/usr/lib64/"

all : stupidfilter

stupidfilter : stupidfilter.o svm.o SVMUtil.o parametersearch.o
	cd bin && \
	g++ -L$(BOOSTLIB) -o"stupidfilter" stupidfilter.o svm.o SVMUtil.o parametersearch.o /usr/lib64/libboost_serialization.a -lfl 

stupidfilter.o : stupidfilter.cpp SVMUtil.cpp SVMUtil.h $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ -c $(BOOST) -o"bin/stupidfilter.o" stupidfilter.cpp

SVMUtil.o : SVMUtil.cpp SVMUtil.h parametersearch.cpp parametersearch.h parameterresult.h $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ -c $(BOOST) -o"bin/SVMUtil.o" SVMUtil.cpp
	
parametersearch.o : parametersearch.cpp parametersearch.h parameterresult.h $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ -c $(BOOST) -o"bin/parametersearch.o" parametersearch.cpp
	
svm.o : $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ -c -o"bin/svm.o" $(LIBSVM)svm.cpp
	

clean:
	rm -f bin/*.o

install: 
	cp bin/stupidfilter /usr/bin/.	    	
