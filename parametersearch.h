#include <set>
#include "../thirdparty/boost/archive/text_oarchive.hpp"
#include "../thirdparty/boost/archive/text_iarchive.hpp"
#include "../thirdparty/boost/serialization/set.hpp"
#include <ostream>
#include "parameterresult.h"
#include <string>

struct svm_problem;
struct svm_parameter;

using namespace std;


typedef multiset<ParameterResult*, lessthan> ResultsSet;

struct RangeParameters
{
	float fParam1Min;
	float fParam1Max;
	float fParam1Step;
	bool bParam1UseLog;

	float fParam2Min;
	float fParam2Max;
	float fParam2Step;
	bool bParam2UseLog;
	
	// members below could be moved to a derived class
	float fParam1RefinementFactor;
	float fParam2RefinementFactor;
	int	nLevels; // alternatively have a minimum refined step for each param
	
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & fParam1Min;
		ar & fParam1Max;
		ar & fParam1Step;
		ar & bParam1UseLog;
		ar & fParam2Min;
		ar & fParam2Max;
		ar & fParam2Step;
		ar & bParam2UseLog;
		ar & fParam1RefinementFactor;
		ar & fParam2RefinementFactor;
	}
	
//	friend ostream& operator<<(ostream &os, const RangeParameters &rp);
	friend ostream& operator<<(ostream &os, const RangeParameters &rp)
	{
		os << rp.fParam1Min << endl;
		os << rp.fParam1Max << endl;
		os << rp.fParam1Step << endl;
		os << rp.bParam1UseLog << endl;
		os << rp.fParam2Min << endl;
		os << rp.fParam2Max << endl;
		os << rp.fParam2Step << endl;
		os << rp.bParam2UseLog << endl;
		os << rp.fParam1RefinementFactor << endl;
		os << rp.fParam2RefinementFactor << endl;
		return os;
	/*	return os << rp.fParam1Min 
		<< rp.fParam1Max
		<< rp.fParam1Step
		<< rp.bParam1UseLog
		<< rp.fParam2Min
		<< rp.fParam2Max
		<< rp.fParam2Step
		<< rp.bParam2UseLog
		<< rp.fParam1RefinementFactor
		<< rp.fParam2RefinementFactor;*/
	}
};


/*class SearchParameters : class RangeParameters
{
public:
	float fRefinementFactor; 
};*/

class CParameterSearch
{
public:

	CParameterSearch(svm_problem* pProb, svm_parameter* pSvmParam, std::string);
	~CParameterSearch();
	
	bool SearchRange(ParameterResult* pResult, RangeParameters&);
	bool GetRefinedParameters(int nLevel, float fParam1, float fParam2, RangeParameters& paramsOut);
	ParameterResult* GetNextResult();
	
	svm_problem* m_pProblem;
	svm_parameter* m_pSvmParam;
	ResultsSet m_searchResults;
	RangeParameters m_rangeParameters;
	
	
private:
	friend class boost::serialization::access;
	friend std::ostream & operator<<(std::ostream &os, const CParameterSearch &ps);

	template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
		ar.register_type(static_cast<ParameterResult *>(NULL));
        ar & m_rangeParameters;
        ar & m_searchResults;
    }
	
	void ResetSerialization();
	void SerializeData();
	void SaveTextResults();
	
	std::ofstream* m_pofs;
	boost::archive::text_oarchive* m_pOA;
	std::string m_strFilename;
};
