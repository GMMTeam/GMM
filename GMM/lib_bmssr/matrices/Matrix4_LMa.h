#if !defined(___MATRIX___)
#define ___MATRIX___

#define _WRITE_ERROR_

#ifdef __GNUC__
#include <inttypes.h>
#endif

// #include <stdexcpt.h>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

#define ___MAX_SIZE___ 15000

#if !defined(_NO_NAMESPACE)
using namespace std;
#endif

class matrix_error //: public logic_error
{
public:
  //matrix_error (const string& what_arg) : logic_error( what_arg) {}
   matrix_error (const string& what_arg) {}
};

typedef enum {GEN=1, SYM=2, DIA=3, VEC=4} MATRIX_TYPE;



class MATRIX  
{

public:
	int _MultAtBA_GenVectVect(MATRIX *A, MATRIX *B);
	void SetXY(int i,int j, double co);
	int _MultAtBA_GenDiaVect(MATRIX *A, MATRIX *B);
	bool IsDiag();
	double GetXY(int x,int y);
	int _GetRowR(int kerej, MATRIX *x);
	int _Add_MultAtA_Gen(MATRIX *A, double alpha);
	int _Add_MultAtA_IntGen(MATRIX *A, double alpha);
	int Add_MultAtA(MATRIX *A,double alpha);
	void _MultFloatArray(float **p, int N, int M); //vynasobi float 2D pole s matici formatu MATRIX
	void _MultFloatArrayT(float **p, int N, int M, float *vM, float *vTM); //to same, ale pred nasobenim odecte vektor vM a po vynasobeni pricte vektor vTM

  static int _create;  
  static int _delete; 
  static int _threadA;
  static double _pomA[___MAX_SIZE___];
  static double _pomB[___MAX_SIZE___];
  static int _ipivA[___MAX_SIZE___];
  static int _ipivB[___MAX_SIZE___];
 
   


	double Norm(int type);
	int Sub(MATRIX *A,MATRIX *B);
  
  
  MATRIX_TYPE TYPE;
  
  
  int   Mult (MATRIX *A);
  
  int MultAB     (MATRIX *A, MATRIX *B);
  int MultAtB    (MATRIX *A, MATRIX *B);
  int MultAtBA   (MATRIX *A, MATRIX *B);
  int Mult_tr    (MATRIX *A);
  
  int _MultAtB_Gen     (MATRIX *A, MATRIX *B);  
  int _MultAtB_IntGen  (MATRIX *A, MATRIX *B);
  
  int _MultAB_Gen      (MATRIX *A, MATRIX *B);
  int _MultAB_IntGen   (MATRIX *A, MATRIX *B);
  int _MultAB_IntSym   (MATRIX *A, MATRIX *B);
  
  
  
  int _MultAtBA_Gen			(MATRIX *A, MATRIX *B);
  int _MultAtBA_Raw			(MATRIX *A, MATRIX *B);
  int _MultAtBA_IntGen		(MATRIX *A, MATRIX *B);
  int _MultAtBA_IntSym		(MATRIX *A, MATRIX *B);
  int _MultAtBA_IntDiag		(MATRIX *A, MATRIX *B);
  int _MultAtBA_IntVect		(MATRIX *A, MATRIX *B);
  int _MultAtBA_IntSymVect	(MATRIX *A, MATRIX *B);

  double  Det();
  double _Det_Gen();
  double _Det_IntGen();
  double _Det_Sym();
 // double _Det_IntelSym();

  double LogDetSPD();
  double LogDet();




  int  Invert();
  int _Invert_IntGen();
  int _Invert_Gen();






  
  
  
  int size_a;
  
  //double & _data(int i, int j);
  
  
  
  double Cofact (int row, int col);
  int SetDiag(void);
  int SetSym(void);
  int GetRowR(int kerej, MATRIX *x);
  int Col();
  int Row();
  int Eig(MATRIX *V,MATRIX *D);
  static int CompareDouble(const void *FirstArgument, const void *SecondArgument);
  int Dim();
  int LinearSolve(double eigVal, double *y, double *x);
  int GetEigVect(double *d,MATRIX *eigVEct,long nr);
  int Hqr(double *wr,double *wi);
  int Elmhes(void);
  int Balance(void);
  int Jacobi(MATRIX *d, MATRIX *v,long *nrot);
  int Choleski(MATRIX &L);
  
 
  
  int pivot (int row);
  
  void SetZero  (void);
  void SetValue (double x);
  void SetDiag  (double x);
  
  
  void  Mult (double c);
  
  void  Div  (double c);	
  
  int   Add  (MATRIX *B);
  
  
  double Euclidian_Distance(double* S1,double * S2,long dim);
  void copy_FV(double * src,double * dest,long dim );
  void random_unit_vector(double * x,long dim );
  double vector_len(double *V,long dim );
  void norm_vector(double * V,long dim );
  void get_random(long max,long * ran );
  int ludcmp(double **a,long n,long *indx,double *d);
  void gaussj(double **a,int n,double **b,int m);
  
  
  
  
  double * GetRowDouble(int x);
  MATRIX * GetRow(int x);
  
  
  
  int Create(MATRIX *vzor);
  
  int Trans(void);
  int Rand();
  
  
  int Delete();
  int Copy(MATRIX *B);
  
  
  
  
  void Print(void);
  void PrintE(void);
  void PrintDiagE(void);
  void PrintDiag(void);
  void PrintM(void);
  
  
  
  
  int SaveTxt (char *filename);
  int LoadHTK (char *filename);
  int LoadTxt (char *filename);
  int LoadBin (char *filename,int size_x);
  int LoadBin (char *filename,int size_x,int size_y);
  int LoadBinF(char *filename);
  int LoadBinD(char *filename);
  int SaveBinD(char *filename);
  int SaveBinF(char *filename);
  
  
  
  double * da_ta;
  int Create(int _x,int _y);
  int size_y;
  int size_x;
  char ID[10];
  
  
  MATRIX();
  
  
  virtual ~MATRIX();
  
  
  double& operator() (int i, int j);
  double* operator[] (int i);
  
  
  MATRIX& operator += (MATRIX X);
  MATRIX& operator -= (MATRIX X);
  MATRIX& operator *= (double c);
  
  
  MATRIX  operator +  (double a);
  MATRIX  operator -  (double a);
  MATRIX  operator *  (double a);
  MATRIX  operator /  (double a);
  MATRIX& operator /= (double c);  
  
  MATRIX  operator + (int a);
  MATRIX  operator - (int a);
  MATRIX  operator * (int a);
  MATRIX  operator / (int a);
  
  MATRIX  operator + (MATRIX X);
  MATRIX  operator - (MATRIX X);
  MATRIX  operator * (MATRIX X);
  MATRIX  operator / (MATRIX X);
  MATRIX& operator = (MATRIX X);
  
  MATRIX  operator~();
  MATRIX  operator!();
  
  
  bool kill;
};


MATRIX operator*(double c, const MATRIX &A);


#endif // !defined(___MATRIX___)


