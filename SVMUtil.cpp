// Copyright 2008 Rarefied Technologies, Inc.
// Distributed under the GPL v2 please see
// LICENSE file for more information.

#include "SVMUtil.h"
#include <string> 
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include "parametersearch.h"
#include "parameterresult.h" 
#include "../thirdparty/libsvm/svm.h"
#include "../thirdparty/boost/serialization/set.hpp"

void myfunction(int);

//#define SCALED_MAX 1
//#define SCALED_MIN 0 //assume features > 0
//#define SCALED_MIN -1 // if features can be < 0

//svm_problem  ParseTrainingFile(string strFilename);

#define MAX_LINE_LENGTH 1024

SVMUtil::SVMUtil()
{
	m_pProblem = NULL;
	m_pScaleFactors = NULL;
	m_nParams = 0;
	m_pModel = NULL;
}

SVMUtil::~SVMUtil()
{
	// problem destroyed when the model is
	//delete m_pProblem;
	//m_pProblem = NULL;
	if(m_pModel)
		svm_destroy_model(m_pModel);
}

// borrowed from read_problem in svm-train.c
svm_problem* SVMUtil::ParseTrainingFile(std::string strFilename)
{
	m_pProblem = new svm_problem;
	svm_node *x_space;
	svm_parameter param;
	
	const char* filename = strFilename.c_str();
	
	int elements, i, j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	m_pProblem->l = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				++m_pProblem->l;
				// fall through,
				// count the '-1' element
			case ':':
				++elements;
				break;
			case EOF:
				goto out;
			default:
				;
		}
	}
out:
	rewind(fp);

	m_pProblem->y = Malloc(double,m_pProblem->l);
	m_pProblem->x = Malloc(struct svm_node *,m_pProblem->l);
	
	int nParamCountGuess = elements / m_pProblem->l;

	m_nParams = 0;

	for(i=0;i<m_pProblem->l;i++)
	{
		double label;
		x_space = Malloc(struct svm_node, nParamCountGuess+1);
		m_pProblem->x[i] = x_space;
		fscanf(fp,"%lf",&label);
		m_pProblem->y[i] = label;
		
		j=0;
		
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			
			int nIndex;
			double dValue;
			
			if (fscanf(fp,"%d:%lf", &nIndex, &dValue) < 2)
			{
				fprintf(stderr,"Wrong input format at line %d\n", i+1);
				exit(1);
			}
			if(dValue!=0)
			{
				x_space[j].index = nIndex;
				x_space[j].value = dValue;
			
				++j;
			}
		}	
out2:
		if(j>=1 && x_space[j-1].index > m_nParams)
			m_nParams = x_space[j-1].index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0)
		param.gamma = 1.0/m_nParams;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<m_pProblem->l;i++)
		{
			if (m_pProblem->x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)m_pProblem->x[i][0].value <= 0 || (int)m_pProblem->x[i][0].value > m_nParams)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
	
	ScaleTrainingData();
	SaveScaleFactors(strFilename + ".sf");
	
	return m_pProblem; 
}

bool SVMUtil::ScaleTrainingData()
{
	if(!m_pProblem)
	{
		assert(0);
		return false;
	}
	
	if(!DetermineScaleFactors())
		return false;
	
	svm_node* pNode = NULL;
	
	for(int i=0; i < m_pProblem->l; i++)
	{
		pNode = m_pProblem->x[i];
		ScaleNode(pNode);
	}
	
	return true;
}

bool SVMUtil::DetermineScaleFactors()
{
	if(!m_pProblem)
		return false;
	
	svm_node* pNode = NULL;
	double* pMaxValue = Malloc(double, m_nParams);
	m_pScaleFactors = Malloc(double, m_nParams);
	
	for(int j=0; j < m_nParams; j++)
	{
		pMaxValue[j] = 0; // assumes values should be scaled between 0 and 1
	}
	
	for(int i=0; i < m_pProblem->l; i++)
	{
		pNode = m_pProblem->x[i];
		int nindex = 0;
		int j=0;
		while(pNode)
		{
			nindex = pNode[j].index;
			if(nindex==-1)
				break;
			
			pMaxValue[nindex-1] = max(pMaxValue[nindex-1], pNode[j].value); // assume values are positive 
			j++;
		}
	}
	
	for(int j=0; j < m_nParams; j++)
	{
		if(pMaxValue[j] > 0)
			m_pScaleFactors[j] = (double)1./pMaxValue[j];
		else
			m_pScaleFactors[j] = 1; 
	}	

	return true;
}

bool SVMUtil::ScaleNode(svm_node* pNode)
{
	if(!pNode)
	{
		cerr << "error scaling" << endl;
		assert(0);
		return false;
	}
	if(!m_pScaleFactors)
	{
		if(m_pProblem)
			DetermineScaleFactors();
		else
		{
			assert(0);
			return false;
		}
	}
	
	int i = 0;
	
	while(pNode[i].index != -1)
	{		
		pNode[i].value *= m_pScaleFactors[pNode[i].index-1]; 
		i++;
	}
	return true;
}

bool SVMUtil::ParameterSearch(svm_parameter* pSvmParam, string strFilename)
{
	if(!m_pProblem || !pSvmParam)
		return false;
	
	//struct sigaction sa;
	//sa.sa_handler = &myfunction;
	//sigaction(SIGINT, &sa, NULL);
	
	CParameterSearch* paramSearch = new CParameterSearch(m_pProblem, pSvmParam, strFilename);
	
	//SaveSearch(paramSearch);

	delete paramSearch;
	
	return true;
}

bool SVMUtil::SaveSearch(const CParameterSearch* p_Search)
{
	std::ofstream ofs("searchResults.txt");
	boost::archive::text_oarchive oa(ofs);
	oa << *p_Search;
	return true;
}
	
void myfunction(int number)
{
	int ten;
	int five = number;

	ten = five + five;
	
}

bool SVMUtil::Load(string filename)
{
	if(m_pModel)
		svm_destroy_model(m_pModel);
	
	string modelname = filename + ".mod";
	string scalename = filename + ".sf";
	m_pModel = svm_load_model(modelname.c_str());
	LoadScaleFactors(scalename);
	
	return true;
}

bool SVMUtil::Save(string filename)
{
	string modelname = filename + ".mod";
	string scalename = filename + ".sf";
	svm_save_model(modelname.c_str(), m_pModel);
	SaveScaleFactors(scalename);
	
	return true;
}

void SVMUtil::SaveScaleFactors(string filename)
{
	ofstream fout;
	fout.open(filename.c_str());
	fout << m_nParams << endl;
	for(int i=0; i<m_nParams; i++)
	{
		fout <<  m_pScaleFactors[i] << endl;
	}
	fout.close();
}


bool SVMUtil::LoadScaleFactors(string filename)
{
	ifstream fin;
	fin.open(filename.c_str());
	fin >> m_nParams;
	
	if(m_pScaleFactors)
		delete m_pScaleFactors;
	m_pScaleFactors = new double[m_nParams];
	
	for(int i=0; i<m_nParams; i++)
	{
		fin >>  m_pScaleFactors[i];	
	}
	return true;
}

bool SVMUtil::CrossValidate(int nFolds, svm_parameter* pParam)
{
	double* target = new double[m_pProblem->l];
	if(nFolds>0)
		svm_cross_validation(m_pProblem, pParam, nFolds, target);
	else
	{
		if(!m_pModel)
			m_pModel = svm_train(m_pProblem, pParam);
		for(int i=0; i<m_pProblem->l; i++)
		{
			target[i] = svm_predict(m_pModel, m_pProblem->x[i]);
		}
	}
	
	float fError = 0;
	float fWrong = 0;
	for(int i=0; i<m_pProblem->l; i++)
	{
		fError += abs(m_pProblem->y[i] - target[i]) ;
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
	
	std::cout << "Percent wrong: " << fWrong << "   Avg Error: " << fError << "  Std Dev: " << fStdDev << std::endl;
	return true;
}
