# Thirdparty directory
TP = thirdparty
LIBSVM = $(TP)/libsvm/
# Location of boost library headers (use system Boost, fall back to bundled for libsvm)
BOOST = -I/usr/include -I$(TP)
# Path to libraries (adjust for your system - /usr/lib64 on RHEL, /usr/lib/x86_64-linux-gnu on Debian)
BOOSTLIB = /usr/lib/x86_64-linux-gnu
# C++11 required for modern Boost; add -Wno-register for flex-generated code
CXXFLAGS = -std=c++11 -Wno-register

PREFIX ?= /usr/local
DESTDIR ?=

.PHONY: all clean install

all : bin/stupidfilter

bin:
	mkdir -p bin

bin/stupidfilter : bin/stupidfilter.o bin/svm.o bin/SVMUtil.o bin/parametersearch.o | bin
	g++ -L$(BOOSTLIB) -o $@ $^ -lboost_serialization

bin/stupidfilter.o : stupidfilter.cpp SVMUtil.h $(LIBSVM)svm.h | bin
	g++ $(CXXFLAGS) -c $(BOOST) -o $@ stupidfilter.cpp

bin/SVMUtil.o : SVMUtil.cpp SVMUtil.h parametersearch.h parameterresult.h $(LIBSVM)svm.h | bin
	g++ $(CXXFLAGS) -c $(BOOST) -o $@ SVMUtil.cpp

bin/parametersearch.o : parametersearch.cpp parametersearch.h parameterresult.h $(LIBSVM)svm.h | bin
	g++ $(CXXFLAGS) -c $(BOOST) -o $@ parametersearch.cpp

bin/svm.o : $(LIBSVM)svm.cpp $(LIBSVM)svm.h | bin
	g++ $(CXXFLAGS) -c -o $@ $(LIBSVM)svm.cpp


clean:
	rm -f bin/*.o bin/stupidfilter

install: bin/stupidfilter
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 bin/stupidfilter $(DESTDIR)$(PREFIX)/bin/stupidfilter
