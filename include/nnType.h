#ifndef NNTYPE
#define NNTYPE

#include <vector>
#include <string>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <list>

#define NNEF extern "C"
using namespace std;

//- global shape -

//----------------
/*
template <class T>
class tensor
{

    public:
        //data
        vector<T> D;

        //shap
        vector<int> shape; //Array

        //data format
        string data_format = "None";

        //dtype
        string dtype = "None";

        //print
        void print()
        {
            vector<int> DimCount;
            vector<int> DimNum;
            DimCount.resize(shape.size());
            DimNum.resize(shape.size());
            int totalCount = 0;
            //int loc = 0; //unuse
            //e.g. 2*3*4
            //     0 1 2
            for (unsigned int i = 0; i < shape.size(); i++)
            {
                int dim = 1;
                // 3, 4
                for (unsigned int j = i + 1; j <= (shape.size() - 1); j++)
                { 
                    dim = dim * shape[j];
                } 
                dim = dim * shape[i];
                DimNum[i] = dim;
                //cout << " " <<DimCount[i];
                //cout << " " <<DimNum[i];
            }

            for (unsigned int i = 0; i < D.size(); i++)
            {
                // set [[[xxx.....
                bool initPos = true; 
                for (unsigned int start = 0; start < shape.size(); start++)
                {
                    if (DimCount[start] == 0)
                    {
                        //init start pos
                        if (initPos)
                        {
                            for (int i = 0; i < totalCount; i++)
                                cout << " ";
                            initPos = false;
                        }

                        cout << "[";
                        totalCount++;
                    }
                    DimCount[start]++;                    
                }

                cout << setw(2) << D[i]; 

                // set .....xxx]]]
                int outmark = 0;
                for (unsigned int end = 0; end < shape.size(); end++)
                {

                    if (DimCount[end] == (DimNum[end]))
                    {
                        outmark++;
                        DimCount[end] = 0;
                    }
                }

                if (outmark > 0)
                {  
                    for (int i = 0; i < outmark; i++)
                    {
                        cout << "]";
                        totalCount--;
                    }
                    for (int i = 0; i < outmark; i++)  
                        cout << "\n";
                }
                else
                    cout << " "; 
            }  
        }

        //N-Dim
        T load (vector<int> &pos )
        {
            int loc = 0;

            for (int i = 0; i < pos.size(); i++)
            {
                int dim = 1;
                for (int j = i; j < pos.size() - 1; j++)
                { 
                    dim = dim * shape[j + 1];
                } 
                dim = dim * pos[i];
                loc = loc + dim;                
            }

            return D[loc];
        }

        //N-Dim
        int pos (vector<int> &pos )
        {
            int loc = 0;

            for (int i = 0; i < pos.size(); i++)
            {
                int dim = 1;
                for (int j = i; j < pos.size() - 1; j++)
                { 
                    dim = dim * shape[j + 1];
                } 
                dim = dim * pos[i];
                loc = loc + dim;                
            }

            return loc;
        }

        //1-D
        T load(int x)
        {
            return D[x];
        }

        //2-D
        T load(int x, int y)
        {
            assert(shape.size() == 2);
            int pos = x * shape[1] + y;
            return D[pos];
        }

        //3-D
        T load(int x, int y, int z)
        {
            assert(shape.size() == 3);
            int pos = x * shape[1] * shape[2] + y * shape[2] + z;
            return D[pos];
        }

        //4-D
        T load(int x, int y, int z, int t)
        {
            assert(shape.size() == 4);
            int pos = x * shape[1] * shape[2] * shape[3]  + y * shape[2] * shape[3] + z * shape[3] + t;
            return D[pos];
        }

        //1-D
        int pos(int x)
        {
            return x;
        }

        //2-D
        int pos(int x, int y)
        {
            assert(shape.size() == 2);
            int pos = x * shape[1] + y;
            return pos;
        }

        //3-D
        int pos(int x, int y, int z)
        {
            assert(shape.size() == 3);
            int pos = x * shape[1] * shape[2] + y * shape[2] + z;
            return pos;
        }

        //4-D
        int pos(int x, int y, int z, int t)
        {
            assert(shape.size() == 4);
            int pos = x * shape[1] * shape[2] * shape[3]  + y * shape[2] * shape[3] + z * shape[3] + t;
            return pos;
        }

        T exp(int index)
        {
            assert(index < D.size());
            T value;
            value = std::exp(D[index]);
            return value;
        }

        void exp()
        {
            for (int i = 0; i < D.size(); i++)
                D[i] = std::exp(D[i]);
        }

        T sum()
        {
            T sum = 0;

            for (int i = 0; i < D.size(); i++)
               sum = sum + D[i];

            // Comparison range
            //assert(sum < numeric_limits<T>::max);
            //assert(sum > numeric_limits<T>::min);
            //assert(sum < FLT_MAX);
            //assert(sum > FLT_MIN);
            return sum;
        }

        tensor(){};
};
*/
#if 1

extern string KernelCode;
//static int NODEID = 0;

//enum Kind { None, Integer, Scalar, Logical, String, Tensor, Array, Tuple };
enum Kind { None, Integer, Scalar, Logical, String, Identifier, Array, Tuple, ShapeOf };

class Token
{
public:
    string sName;
    vector<int> iInteger;
	string sString;
    vector<int> viArray;
    vector<string> vsArray;
    Kind eKind;
    int iTensorNum = -1;
    // if (Kind == None) output sContent
    string sContent;
};

//nodeGraph
//#include "node.h"
class nnNode //: public node
{
    public: 
        string NAME;
        string funType;

        vector <int> shape;
        //perm
/*
        vector <int> perm; //Array
        vector <int> size;
        vector <int> stride;
        vector <int> dilation;
        vector <int> padding;
        string label;
        string border;
        bool trA,trB;

        float scalar; 
        int groups;
*/
        vector<Token> result;
        vector<Token> param;

        vector<int> resultNum;
        vector<int> paramNum;

        int ID;
        string DATA;
        //string NAME;
        int inDegree;
        list<nnNode*> IN;
        list<nnNode*> OUT;

        int resultCount = -1;
        string operation;
        int paramCount = -1;
        vector<Token> tokenSet;

        nnNode& operator >> (nnNode &x);
        nnNode& operator << (nnNode &x);

        void run();
        void run(int type);
        nnNode();
};
#endif

class nnKernel
{

    public:
        string KernelSourceFile;
        vector<nnNode> Graph;
};

#endif
