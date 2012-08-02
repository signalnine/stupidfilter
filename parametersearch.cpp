// Copyright 2008 Rarefied Technologies, Inc.
// Distributed under the GPL v2 please see
// LICENSE file for more information.

#include "parametersearch.h"
#include "parameterresult.h"
#include "../thirdparty/libsvm/svm.h"
#include <cmath>
#include <iostream>
#include <fstream>

#define REFINED_RANGE 1.33

CParameterSearch::CParameterSearch(svm_problem* pProb, svm_parameter* pSvmParam, string strFilename)
{
	if(!pProb || !pSvmParam)
		return;
	
	m_pOA = NULL;
	m_pofs = NULL;
	m_strFilename = strFilename;
	
	ResetSerialization();
	
	m_pProblem = pProb;
	m_pSvmParam = pSvmParam;

	m_rangeParameters.fParam1Min = -15;
	//m_rangeParameters.fParam1Max = -1;
	m_rangeParameters.fParam1Max = 3;
	m_rangeParameters.fParam1Step = 4;
	m_rangeParameters.bParam1UseLog = true;
	m_rangeParameters.fParam1RefinementFactor = 2;
	m_rangeParameters.fParam2Min = -13;
	//m_rangeParameters.fParam2Max = -1;
	m_rangeParameters.fParam2Max = -5;
	m_rangeParameters.fParam2Step = 4;
	m_rangeParameters.bParam2UseLog = true;
	m_rangeParameters.fParam2RefinementFactor = 2;
	
	ParameterResult* pResult = new ParameterResult;

	const RangeParameters tempRP = m_rangeParameters;
	*m_pOA << tempRP;
	m_pofs->flush();
	
	SearchRange( pResult, m_rangeParameters);
	
	// semi-infinite, breakable loop
	while( pResult = GetNextResult())
	{			
		RangeParameters Params;
		GetRefinedParameters(pResult->nLevel, pResult->fParam1, pResult->fParam2, Params);
		if(!SearchRange( pResult, Params))
			break;
	}
}

CParameterSearch::~CParameterSearch()
{
	
}

ParameterResult* CParameterSearch::GetNextResult()
{
	ParameterResult* pResult = NULL;
	ResultsSet::iterator it = m_searchResults.begin();
	while(it != m_searchResults.end())
	{
		pResult = *it;
		if(!pResult->bRefined)
			return pResult;
		++it;
	}
	return NULL;	
}

bool CParameterSearch::SearchRange(ParameterResult* pResult, RangeParameters& Params)
{
	if(!m_pProblem || !m_pSvmParam || !pResult)
		return false;
	
	float fParam1 = 0;
	float fParam2 = 0;
	
	double* target = new double[m_pProblem->l];
/*	for(int i=0; i<m_pProblem->l; i++)
	{
		target[i] = 0;
	}*/
	
	for(fParam1=Params.fParam1Min; fParam1<=Params.fParam1Max; fParam1+=Params.fParam1Step)
	{
		if(Params.bParam1UseLog)
			m_pSvmParam->p = ::pow(2,fParam1);
		else
			m_pSvmParam->p = fParam1;
		
		for(fParam2=Params.fParam2Min; fParam2<=Params.fParam2Max; fParam2+=Params.fParam2Step)
		{	
			if(Params.bParam2UseLog)
				m_pSvmParam->C = ::pow(2,fParam2);
			else
				m_pSvmParam->C = fParam2;
			
			int nFolds = 2;
			svm_cross_validation(m_pProblem, m_pSvmParam, nFolds, target);
			float fError = 0;
			float fWrong = 0;
			for(int i=0; i<m_pProblem->l; i++)
			{
				fError += abs(m_pProblem->y[i] - target[i]);
				if( m_pProblem->y[i] >= 0.5 && target[i] < 0.5)
						fWrong++;
				else if( m_pProblem->y[i] < 0.5 && target[i] >= 0.5)
						fWrong++;
			}
			fError = (float)fError/m_pProblem->l;
			fWrong = (float) fWrong/m_pProblem->l;
			
			float fStdDev = 0;
			for(int i=0; i<m_pProblem->l; i++)
			{
				fStdDev += pow(fError - abs(m_pProblem->y[i] - target[i]), 2) ;
			}
			fStdDev = pow(fStdDev, (float)0.5) /  m_pProblem->l;
			
			std::cout << "\n****************" << std::endl;
			std::cout << "C = 2^" << fParam2 << ", epsilon = 2^" << fParam1 << std::endl;
			std::cout << "Avg Error: " << fError << "  Std Dev: " << fStdDev << std::endl;
			std::cout << "Percent wrong: " << fWrong << std::endl;
			
			ParameterResult* pNewResult = new ParameterResult;
			pNewResult->fError = fError;
			pNewResult->fStdDev = fStdDev;
			pNewResult->fWrong = fWrong;
			pNewResult->fParam1 = fParam1;
			pNewResult->fParam2 = fParam2;
			pNewResult->nLevel = pResult->nLevel + 1;
			m_searchResults.insert(pNewResult);
				
			const ParameterResult* pConstResult = pNewResult;
							
			*m_pOA << *pConstResult;
			m_pofs->flush();
			
			//SaveTextResults();
		}
	}
	pResult->bRefined = true;
	
	SerializeData();
	
	return true;
}


// nLevel is desired level, not current level
bool CParameterSearch::GetRefinedParameters(int nLevel, float fParam1, float fParam2, RangeParameters& paramsOut)
{
	paramsOut.bParam1UseLog = m_rangeParameters.bParam1UseLog;
	float fParam1StepPrev = m_rangeParameters.fParam1Step / pow(m_rangeParameters.fParam1RefinementFactor, nLevel-1);
	paramsOut.fParam1Min = max( m_rangeParameters.fParam1Min, fParam1 - (float)REFINED_RANGE * fParam1StepPrev );
	paramsOut.fParam1Max = min( m_rangeParameters.fParam1Max, fParam1 + (float)REFINED_RANGE * fParam1StepPrev );
	paramsOut.fParam1Step = m_rangeParameters.fParam1Step / pow(m_rangeParameters.fParam1RefinementFactor, nLevel);

	paramsOut.bParam2UseLog = m_rangeParameters.bParam2UseLog;
	float fParam2StepPrev = m_rangeParameters.fParam2Step / pow(m_rangeParameters.fParam2RefinementFactor, nLevel-1);
	paramsOut.fParam2Min = max( m_rangeParameters.fParam2Min, fParam2 - (float)REFINED_RANGE * fParam2StepPrev );
	paramsOut.fParam2Max = min( m_rangeParameters.fParam2Max, fParam2 + (float)REFINED_RANGE * fParam2StepPrev );
	paramsOut.fParam2Step = m_rangeParameters.fParam2Step / pow(m_rangeParameters.fParam2RefinementFactor, nLevel);
	
	return true;
}

void CParameterSearch::ResetSerialization()
{
	delete m_pOA;
	delete m_pofs;
	
	m_pofs = new std::ofstream(m_strFilename.c_str());
	m_pOA = new boost::archive::text_oarchive(*m_pofs);
}

void CParameterSearch::SerializeData()
{
	ResetSerialization();

	const CParameterSearch* pSearch = this;
	
	*m_pOA << *pSearch;
	
	m_pofs->flush();
}

void CParameterSearch::SaveTextResults()
{
	std::ofstream ofs("SearchResults.txt");
	ofs << *this;
}

std::ostream & operator<<(std::ostream &os, const CParameterSearch &ps)
{
	os << ps.m_rangeParameters << std::endl << std::endl;
	ResultsSet::iterator it = ps.m_searchResults.begin();
	while(it != ps.m_searchResults.end())
	{
		const ParameterResult* pResult = *it;
		
		os << *pResult << '\n';

		++it;
	}

    return os;
}

