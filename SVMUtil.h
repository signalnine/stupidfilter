#pragma once

struct svm_problem;
struct svm_node;
struct svm_parameter;
struct svm_model;

class CParameterSearch;

#include <string> 

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class SVMUtil
{
public:
	SVMUtil();

	~SVMUtil();
	
	svm_problem* ParseTrainingFile(std::string strFilename);
	bool ParameterSearch( svm_parameter* pSvmParam, std::string strFilename);
	bool ScaleNode(svm_node*);
	bool CrossValidate(int nFolds, svm_parameter*);
	bool Load(std::string);
	bool Save(std::string);
	
	svm_model* m_pModel;

private:
	
	bool ScaleTrainingData();
	bool DetermineScaleFactors();
	bool SaveSearch(const CParameterSearch* p_Search);
	void SaveScaleFactors(std::string);
	bool LoadScaleFactors(std::string);
	
	svm_problem* m_pProblem;
	
	double* m_pScaleFactors;
	int m_nParams;
};




