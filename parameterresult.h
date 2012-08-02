// Copyright 2008 Rarefied Technologies, Inc.
// Distributed under the GPL v2 please see
// LICENSE file for more information.

#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


class ParameterResult
{
public:
	ParameterResult()	{ bRefined = false; fError=0; nLevel=0; };
		
	float fError;
	float fStdDev;
	float fWrong; // percent of predictions that would yield the wrong class
	float fParam1;
	float fParam2;
	bool bRefined; // indicates if a refinement search has been spawned from this result
	int nLevel; // which level of refinement this result is from

private:

	friend class boost::serialization::access;
	
	template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
		ar & fWrong;
		ar & fError;
		ar & fStdDev;
		ar & fParam1;
		ar & fParam2;
		ar & bRefined;
		ar & nLevel;
    }
	friend std::ostream& operator<<(std::ostream &os, const ParameterResult &pr)
	{
		os << pr.fWrong << '\t';
		os << pr.fError << '\t';
		os << pr.fStdDev << '\t';
		os << pr.fParam1 << '\t';
		os << pr.fParam2 << '\t';
		os << pr.bRefined << '\t';
		os << pr.nLevel;
				
	    return os;
	}
};


class lessthan
{
public:
	bool operator()(const ParameterResult* left, const ParameterResult* right) const
	{
		bool bRet = (left->fWrong < right->fWrong);
		if(!bRet && (left->fWrong == right->fWrong ))
		{
			bRet = left->fError < right->fError;
		
		}
		
		return bRet;
	}

};

