stupidfilter
============
To run the StupidFilter directly just type bin/stupidfilter data/c_rbf

It will take data from standard in followed by a EOF and return a 0.000000 classification for stupid text and a 1.000000 for nonstupid text. Once we have regression working more accurately, this number will be actually be a floating point that describes how sure we are of the classification, so it's worth keeping it as a float in your implementations.

We have provided an example bash implementation in classify.sh. Note that we're normalizing whitespace with a call to sed. It's a good idea to strip HTML and normalize whitespace to avoid false positives.
