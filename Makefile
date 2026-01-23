# Thirdparty directory
TP = thirdparty
LIBSVM = $(TP)/libsvm/
# Location of boost library headers (use system Boost, fall back to bundled for libsvm)
BOOST = -I/usr/include -I$(TP)
# Path to libraries (adjust for your system - /usr/lib64 on RHEL, /usr/lib/x86_64-linux-gnu on Debian)
BOOSTLIB = /usr/lib/x86_64-linux-gnu
# C++11 required for modern Boost; add -Wno-register for flex-generated code
CXXFLAGS = -std=c++11 -Wno-register

all : stupidfilter

stupidfilter : stupidfilter.o svm.o SVMUtil.o parametersearch.o
	cd bin && \
	g++ -L$(BOOSTLIB) -o"stupidfilter" stupidfilter.o svm.o SVMUtil.o parametersearch.o -lboost_serialization

stupidfilter.o : stupidfilter.cpp SVMUtil.cpp SVMUtil.h $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ $(CXXFLAGS) -c $(BOOST) -o"bin/stupidfilter.o" stupidfilter.cpp

SVMUtil.o : SVMUtil.cpp SVMUtil.h parametersearch.cpp parametersearch.h parameterresult.h $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ $(CXXFLAGS) -c $(BOOST) -o"bin/SVMUtil.o" SVMUtil.cpp

parametersearch.o : parametersearch.cpp parametersearch.h parameterresult.h $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ $(CXXFLAGS) -c $(BOOST) -o"bin/parametersearch.o" parametersearch.cpp

svm.o : $(LIBSVM)svm.cpp $(LIBSVM)svm.h
	g++ $(CXXFLAGS) -c -o"bin/svm.o" $(LIBSVM)svm.cpp
	

clean:
	rm -f bin/*.o

install: 
	cp bin/stupidfilter /usr/bin/.	    	
