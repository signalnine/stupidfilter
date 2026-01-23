/* fclassify.flex - Reconstructed from stupidfilter.cpp DFA tables
 * Original: Copyright 2008 Rarefied Technologies, Inc.
 * Reconstruction based on reverse engineering the flex-generated code
 */

%option noyywrap

%{
float misspell = 0, word_length = 0, num_total = 0, num_lowers = 0, num_caps = 0, num_punct = 0, word_count = 0, initial_cap = 0, intercap = 0, repeat_emphasis = 0;
%}

%%
    /* Rule 1: Count all characters */
.                           ++num_total; REJECT;

    /* Rule 2: Count lowercase letters */
[a-z]                       ++num_lowers; REJECT;

    /* Rule 3: Count uppercase letters */
[A-Z]                       ++num_caps; REJECT;

    /* Rule 4: Count punctuation */
[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]  ++num_punct; REJECT;

    /* Rule 5: Count words */
[a-zA-Z]+                   ++word_count; REJECT;

    /* Rule 6: Words starting with capital (after whitespace/start) */
(^|[ \t\n])[A-Z][a-z]*      ++initial_cap; REJECT;

    /* Rule 7: InterCap words (camelCase) */
[a-z]+[A-Z][a-z]*           ++intercap; REJECT;

    /* Rule 8: Repeated emphasis (!! or ??) */
[!]{2,}|[?]{2,}             ++repeat_emphasis; REJECT;

    /* Rule 9: Common misspellings / l33t speak patterns
 * Based on yy_ec showing special classes for '1', '4', '8'
 * Common patterns: u/ur (you/your), 4 (for), gr8 (great), etc.
 */
[Uu][Rr]?/[^a-zA-Z]         ++misspell; REJECT;
[0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+[a-zA-Z]*  ++misspell; REJECT;

    /* Rule 10: Whitespace - ignore */
[ \t\n]+                    ;

    /* Rule 11: Other characters - ignore */
.                           ;

%%

#include "SVMUtil.h"
#include "thirdparty/libsvm/svm.h"
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("usage: %s [model filename]\n", argv[0]);
        return 1;
    }

    yylex();

    num_lowers = num_lowers / num_total;
    num_caps = num_caps / num_total;
    num_punct = num_punct / num_total;
    initial_cap = initial_cap / word_count;
    intercap = intercap / word_count;
    word_length = word_count / num_total;

    // Model/scale factor filename (no extension)
    std::string strFilename = argv[1];

    SVMUtil svmutil;

    if (svmutil.Load(strFilename.c_str()))
    {
        const int num_attributes = 8;

        svm_node* node = new svm_node[num_attributes + 1];

        for (int i = 0; i < num_attributes; i++)
        {
            node[i].index = i + 1;
        }
        node[num_attributes].index = -1;

        node[0].value = num_lowers;
        node[1].value = num_caps;
        node[2].value = num_punct;
        node[3].value = repeat_emphasis;
        node[4].value = initial_cap;
        node[5].value = intercap;
        node[6].value = word_length;
        node[7].value = misspell;

        svmutil.ScaleNode(node);

        double dPredictedClass = svm_predict(svmutil.m_pModel, node);
        printf("%f\n", dPredictedClass);

        delete[] node;
    }

    return 0;
}
