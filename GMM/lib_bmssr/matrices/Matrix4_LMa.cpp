//#define __VYHOD_CHYBY__
//#define ___MKL___

#include <new>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "general/GlobalDefine.h"
#include "matrices/Matrix4_LMa.h"

//#include <windows.h>

#ifdef ___MKL___
#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_types.h>
#endif


int MATRIX::Row(){return size_x;}
int MATRIX::Col(){return size_y;}

int  MATRIX::_create = 0;  
int  MATRIX::_delete = 0;  
int MATRIX::_threadA = 0;


double MATRIX::_pomA[___MAX_SIZE___];
double MATRIX::_pomB[___MAX_SIZE___];
int MATRIX::_ipivA[___MAX_SIZE___];
int MATRIX::_ipivB[___MAX_SIZE___];





//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MATRIX::MATRIX()
{
  size_x  = 0;
  size_y  = 0;
  size_a  = 0;
  da_ta   = NULL;
  kill    = false;
  TYPE    = GEN; 
  ID[0]   ='\0';
}

MATRIX::~MATRIX()
{
  size_x  = 0;
  size_y  = 0;
  size_a  = 0;
  da_ta   = NULL;
  TYPE    = GEN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      vytvoreni kopirovani zruseni
//
////////////////////////////////////////////////////////////////////////////////////////////////////

int MATRIX::Create(int _x, int _y)
{
  _create++;
  
  if((size_x!=0)||(size_y!=0))
    Delete();
  
  size_x  = _x;
  size_y  = _y;
  size_a  = size_x*size_y;  
  
  da_ta= new(std::nothrow) double [size_a];
  
  if(da_ta==NULL)
    return RETURN_NOT_ENOUGH_MEMORY;  
  
  SetZero();
  TYPE = GEN;
  
  return RETURN_OK;
}

int MATRIX::Create(MATRIX *vzor)
{
  int iRet;
  iRet = Create(vzor->size_x,vzor->size_y);
  if(iRet!=RETURN_OK)
    return iRet;
  
  iRet = Copy(vzor);
  if(iRet!=RETURN_OK)
    return iRet;
  
  strcpy(ID,vzor->ID);
  
  return RETURN_OK;
}


int MATRIX::Copy(MATRIX *B)
{
  
  if((size_x!=B->size_x)||(size_y!=B->size_y))
  { 
    Delete();
    
    size_x = B->size_x;
    size_y = B->size_y;
    size_a = size_x*size_y;    
	// LMa 29.5.2009
    int iRet = Create(size_x,size_y);
	if(iRet != RETURN_OK)
		return iRet;
    TYPE   = B->TYPE;
  }
  
  for(int i=0;i<9;i++) ID[i]=B->ID[i];
  //for(int a=0;a<size_a;a++)   da_ta[a] = B->da_ta[a];
  memcpy( da_ta , B->da_ta , sizeof(double) * size_a);  
  
  return RETURN_OK;
}

int MATRIX::Delete()
{
  if(da_ta!=NULL)
  {
    _delete++;
    delete []da_ta;
  }
  
  da_ta   = NULL;
  size_x  = 0;
  size_y  = 0;
  size_a  = 0;
  kill    = false;  
  return RETURN_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      nulovani scitani nasobeni atd ...
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void MATRIX::SetZero()
{
  for(int a=0;a<size_a;a++)  da_ta[a]=0.0f;
}

void MATRIX::SetValue(double X)
{
  for(int a=0;a<size_a;a++)  da_ta[a]=X;
}

void MATRIX::SetDiag(double X)
{
  int x,y;
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      da_ta[x * size_y + y]=0.0f;
      if(x==y)
        da_ta[x * size_y + x]=X;
      
    }
  } 
}

int MATRIX::SetDiag(void)
{
  int x,y;
  int iSize = min(size_x,size_y);
  
  for(x=0;x<iSize;x++)
  {
    for(y=0;y<iSize;y++)
      if(x!=y)
        da_ta[x * size_y + y]=0.0;    
  } 
  return RETURN_OK;
}

int MATRIX::SetSym(void)
{
  int x,y;
  int iSize = min(size_x,size_y);
  
  for(x=0;x<iSize;x++)
  {
    for(y=x;y<iSize;y++)
      if(x!=y)
        da_ta[x * size_y + y]=da_ta[y * size_y + x];    
  } 
  return RETURN_OK;
}


int MATRIX::Add(MATRIX *B)
{
  if((B->size_x!=size_x)||(B->size_y!=size_y))
    return RETURN_WRONG_MATRIX_SIZE;
  
  for(int a=0;a<size_a;a++)
    da_ta[a] += B->da_ta[a];
  
  return RETURN_OK;
}

int MATRIX::Sub(MATRIX *A, MATRIX *B)
{
  int iRet;
  
  if((A->size_x!=B->size_x)||(A->size_y!=B->size_y))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::Sub(MATRIX *A, MATRIX *B): Incompatible matrix sizes!");
#endif
    printf("MATRIX:: Sub(MATRIX *A, MATRIX *B) : Incompatible matrix sizes {%d,%d} {%d,%d}\n",A->size_x,A->size_y,A->size_x,A->size_y);
    return RETURN_WRONG_MATRIX_SIZE;
  }
  
  if((size_x!=A->size_x)||(size_y!=A->size_y))
  {
    iRet = Create(A->size_x,A->size_y);
    if(iRet!=RETURN_OK)
      return iRet;
  }    
  
  double *_a = A->da_ta;
  double *_b = B->da_ta;
  int a;
  
  for(a = 0; a < size_a; a++ )  da_ta[a] = *(_a++) - *(_b++);  
  // for(a = 0; a < size_a; a++ )  da_ta[a] = A->da_ta[a] - B->da_ta[a];  
  
  return RETURN_OK;
  
}



void MATRIX::Mult(double c)
{
  for(int a=0;a<size_a;a++)  da_ta[a]*=c;  
}

void MATRIX::Div(double c)
{ 
  
#ifdef __VYHOD_CHYBY__
  if(c==0.0f) 
    REPORT_ERROR( "MATRIX::Div(c): c=0.0 !");
#endif
  
  for(int a=0;a<size_a;a++)    da_ta[a]/=c;
}


int MATRIX::Mult(MATRIX *A)
{
  if(size_y!=A->size_x)
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::Mult() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::Mult() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",size_x,size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  int _size_x,_size_y,_size_a;
  
  _size_x = size_x;
  _size_y = A->size_y;
  _size_a = _size_x * _size_y;
  
  double *d=NULL;
  
  d= new double [_size_a];
  
  if(d==NULL)
    return RETURN_NOT_ENOUGH_MEMORY;
  
  int i,j,k;
  
  for( i = 0; i < _size_x; i++ )
  {
    for( j = 0; j < _size_y; j++)
    {
      d[i*_size_y + j] = 0.0;
      for( k = 0; k < size_y; k++ )
      {
        d[i*_size_y + j] += da_ta[i * size_y + k] * A->da_ta[ k * A->size_y + j];
      }
    }
  }  
  
  delete [] da_ta;
  
  da_ta    = d;
  size_x  = _size_x;
  size_y  = _size_y;
  size_a  = _size_a;
  
  return RETURN_OK;
}
int MATRIX::MultAB(MATRIX *A, MATRIX *B)
{
  int iRet;
  
  switch( A->TYPE ) 
  {
  case GEN:
    {
#ifdef ___MKL___  
      iRet = _MultAB_IntGen(A,B);
#else
      iRet = _MultAB_Gen(A,B);
#endif      
      break;
    }
  case SYM:
    {
#ifdef ___MKL___  
      iRet = _MultAB_IntSym(A,B);
#else
      iRet = _MultAB_Gen(A,B);
#endif      
      break;
    }
  case DIA:
    {
#ifdef ___MKL___  
      iRet = _MultAB_IntSym(A,B);
#else
      iRet = _MultAB_Gen(A,B);
#endif      
      break;
    }
  default :
    {      
#ifdef __VYHOD_CHYBY__  
      REPORT_ERROR( "MATRIX::MultAB() : Incompatible matrix type!");
#endif
      printf("MATRIX::MultAB() : Incompatible matrix type\n");
      return RETURN_FAIL;      
    }
  }
  
  return iRet;
  
}

int MATRIX::_MultAB_Gen(MATRIX *A, MATRIX *B)
{
  int iRet;
  
  if(A->size_y!=B->size_x)
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::MultAB() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::MultAB() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_x,A->size_y,B->size_x,B->size_y);
    return RETURN_FAIL;
  }
  
  int X,Y,Z;
  
  X = A->size_x;
  Y = B->size_y;
  Z = A->size_y;
  
  if((size_x!=X)||(size_y!=Y))
  {
    iRet = Create(X,Y);
    if(iRet!=RETURN_OK)
      return iRet;
  }
  
  SetZero();
  
  int i,j,k;
  
  for( i = 0; i < X; i++ )
  {
    for( j = 0; j < Y; j++)
    {
      for( k = 0; k < Z; k++ )
      {
        da_ta[i * size_y + j] += A->da_ta[ i * A->size_y + k] * B->da_ta[ k *B->size_y + j];
      }
    }
  }  
  return RETURN_OK;
}

int MATRIX::_MultAtBA_Raw(MATRIX *A, MATRIX *B)
{
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::MultAtBA() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::MultAtBA() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  int iRet;
  
  int _size_x,_size_y;
  
  _size_x=A->size_y;
  _size_y=A->size_y;
  
  if((size_x!=_size_x)||(size_y!=_size_y))
  {
    iRet = Create(_size_x,_size_y);
    if(iRet!=RETURN_OK)
      return iRet;
  }
  int i,j,k,l;
  
  
  double a_ij;
  
  
  for( i = 0; i < size_x; i++ )
  {
    for( j = 0; j < size_y; j++)
    {
      da_ta[i * size_y + j] = 0.0;
      
      for( l = 0; l < A->size_x; l++)
      {
        a_ij = 0.0;
        for( k = 0; k < B->size_x; k++ )
        {
          a_ij+= A->da_ta[k * A->size_y + i] * B->da_ta[k * B->size_y + l];
        }
        da_ta[i * size_y + j] += a_ij * A->da_ta[l*A->size_y + j];
      }
    }
  }  
  /**/
  return RETURN_OK;
}

int MATRIX::MultAtB(MATRIX *A, MATRIX *B)
{
  int iRet;
  
#ifdef ___MKL___  
  iRet = _MultAtB_IntGen(A,B);
#else
  iRet = _MultAtB_Gen(A,B);
#endif      
  
  return iRet;
}

int MATRIX::_MultAtB_Gen(MATRIX *A, MATRIX *B)
{
  int iRet;
  
  if(A->size_x!=B->size_x)
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::MultAB() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::MultAtB() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y);
    return RETURN_FAIL;
  }
  
  int _size_x,_size_y;
  
  _size_x=A->size_y;
  _size_y=B->size_y;
  
  if((size_x!=_size_x)||(size_y!=_size_y))
  {
    iRet = Create(_size_x,_size_y);
    if(iRet!=RETURN_OK)
      return iRet;
  }
  int i,j,k,I,J,K;
  
  I = _size_x;
  J = _size_y;
  K = A->size_x;
  
  SetZero();
  
  for( i = 0; i < I; i++ )
  {
    for( j = 0; j < J; j++)
    {
      for( k = 0; k < K; k++ )
      {
        da_ta[i * size_y + j] += A->da_ta[ k * A->size_y + i] * B->da_ta[ k *B->size_y + j];
      }
    }
  }
  
  return RETURN_OK;
}

int MATRIX::Mult_tr(MATRIX *A)   // vynasobi this s matici A ale transponovanou HUSTY !!!
{
  if(size_y!=A->size_y)
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::Mult() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::Mult() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",size_x,size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  int _size_x,_size_y,_size_a;
  
  _size_x = size_x;
  _size_y = A->size_x;
  _size_a = _size_x * _size_y;
  
  double *d=NULL;
  
  d= new double [_size_a];
  
  if(d==NULL)
    return RETURN_NOT_ENOUGH_MEMORY;
  
  int i,j,k;
  
  for( i = 0; i < _size_x; i++ )
  {
    for( j = 0; j < _size_y; j++)
    {
      d[i*_size_y + j] = 0.0;
      for( k = 0; k < size_y; k++ )
      {
        d[i*_size_y + j] += da_ta[i * size_y + k] * A->da_ta[ j * A->size_x + k];
      }
    }
  }  
  
  delete [] da_ta;
  
  da_ta    = d;
  size_x  = _size_x;
  size_y  = _size_y;
  size_a  = _size_a;
  
  return RETURN_OK;
  
}
int MATRIX::MultAtBA(MATRIX *A, MATRIX *B)
{
  int iRet;
  
  if(A->size_y==1)
  {
    switch( B->TYPE ) 
    {
    case VEC:
      {
        iRet = _MultAtBA_GenVectVect(A,B);
        break;
      }
    case GEN:
      {
        iRet = _MultAtBA_IntVect(A,B);
        break;
      }
    case SYM:
      {
        iRet = _MultAtBA_IntSymVect(A,B);
        break;
      }
    case DIA:
      {
        iRet = _MultAtBA_GenDiaVect(A,B);
        break;
      }
    default:
      {
        printf("SHIT!!\n");
        /*char s[5];
        gets(s);*/
      }
    }
    
    return iRet;
  }
  
  
  switch( B->TYPE ) 
  {
  case GEN:
    {
#ifdef ___MKL___  
      iRet = _MultAtBA_IntGen(A,B);
#else
      iRet = _MultAtBA_Gen(A,B);
#endif      
      break;
    }
  case SYM:
    {
#ifdef ___MKL___  
      iRet = _MultAtBA_IntSym(A,B);
#else
      iRet = _MultAtBA_Gen(A,B);
#endif      
      break;
    }
  case DIA:
    {
#ifdef ___MKL___  
      iRet = _MultAtBA_IntDiag(A,B);
#else
      iRet = _MultAtBA_Gen(A,B);
#endif      
      break;
    }
  default :
    {      
#ifdef __VYHOD_CHYBY__  
      REPORT_ERROR( "MATRIX::MultAtBA() : Incompatible matrix type!");
#endif
      printf("MATRIX::MultAtBA() : Incompatible matrix type\n");
      return RETURN_FAIL;      
    }
  }
  
  return iRet;
}



int MATRIX::_MultAtBA_Gen(MATRIX *A, MATRIX *B)
{
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::MultAtBA() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::MultAtBA() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  int iRet;
  
  iRet = MultAtB(A,B);
  if(iRet!=RETURN_OK)
    return iRet;
  
  iRet = Mult(A);
  if(iRet!=RETURN_OK)
    return iRet;
  /**/
  
  return RETURN_OK;
}


int MATRIX::Trans()
{
  int x,y;
  
  if((size_x==1)||(size_y==1))
  {
    if(size_x==1) { size_x = size_y; size_y=1;return RETURN_OK; }
    if(size_y==1) { size_y = size_x; size_x=1;return RETURN_OK; }
  }
  else
  { 
    MATRIX pom;
    
	// LMa 29.5.2009
    int iRet = pom.Create(size_y,size_x);
	if(iRet != RETURN_OK)
		return iRet;
    
    for(x=0;x<size_x;x++)
    {
      for(y=0;y<size_y;y++)
      {
        pom[y][x] = da_ta[x * size_y + y];
      }
    }
    
    Delete();
    
    size_x  = pom.size_y;
    size_y  = pom.size_x;
    size_a  = pom.size_x * pom.size_y;  
    da_ta   = pom.da_ta; 
    
    return RETURN_OK;
  }
  return RETURN_OK;
}


int MATRIX::Rand()
{
  for(int a=0;a<size_a;a++)  da_ta[a]=rand()/(double)(RAND_MAX);
  return RETURN_OK;
}


double MATRIX::Norm(int type)
{
  double N=0.0;
  
  if(type==0)
  {
    for(int a=0;a<size_a;a++) N+= da_ta[a];
  }
  
  return N;
}




////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      zkurveny operatory
//
////////////////////////////////////////////////////////////////////////////////////////////////////



double& MATRIX::operator()(int i, int j)
{
#ifdef __VYHOD_CHYBY__
  if((i>=size_x)||(j>=size_y))    
    REPORT_ERROR( "MATRIX::operator(i,j): Index out of range!");  
#endif
  
  return da_ta[i * size_y + j];
}

double* MATRIX::operator[](int i)
{
#ifdef __VYHOD_CHYBY__
  if(i>=size_x)    
    REPORT_ERROR( "MATRIX::operator[i][?]: Index out of range!");
#endif
  
  return &da_ta[i*size_y];
}

MATRIX& MATRIX::operator=(MATRIX L)
{  
  if((size_x!=L.size_x)||(size_y!=L.size_y))
  {    
    Delete();
    
    size_x=L.size_x;
    size_y=L.size_y;
    
    Create(size_x,size_y);    
  }
  
  //for(int a=0;a<size_a;a++) da_ta[a] = L.da_ta[a];
  
  memcpy( da_ta , L.da_ta , sizeof(double) * size_a);
  
  
  if(L.kill)
    L.Delete();
  
  return *this;
}

MATRIX& MATRIX::operator+=(MATRIX L)
{ 
  
  if((size_x!=L.size_x)||(size_y!=L.size_y))
  {    
    Delete();
    
    size_x=0;
    size_y=0;
    
    Create(size_x,size_y);    
  }
  
  for(int a=0;a<size_a;a++) da_ta[a] += L.da_ta[a];  
  
  if(L.kill)
    L.Delete();
  
  return *this;
}


MATRIX& MATRIX::operator*=(double c)
{  
  for(int a=0;a<size_a;a++)  da_ta[a] *= c;    
  return *this;
}

MATRIX MATRIX::operator+(double c)
{
  MATRIX X;
  
  X.Create(size_x,size_y);
  
  for(int a=0;a<size_a;a++)  X.da_ta[a] = da_ta[a]+c;
  
  return X;  
}

MATRIX MATRIX::operator-(double c)
{
  return operator +(-c);  
}

MATRIX MATRIX::operator*(double c)
{  
  int a;
  
  MATRIX X;
  
  X.Create(size_x,size_y);
  
  for(a=0;a<size_a;a++) X.da_ta[a] = da_ta[a]*c;
  
  if(kill)
    Delete();
  
  X.kill = true;
  return X;  
}

MATRIX MATRIX::operator/(double c)
{
  return operator*(1/c);  
}


MATRIX MATRIX::operator+(int c)
{
  double C=(double)c;
  
  return operator +(C);  
}

MATRIX MATRIX::operator-(int c)
{
  return operator +(-c);  
}

MATRIX MATRIX::operator*(int c)
{
  double C=(double)c;
  
  return operator *(C);  
}

MATRIX MATRIX::operator/(int c)
{
  double C=1/(double)c;
  return operator *(C);  
}

MATRIX& MATRIX::operator /= (double c)
{
  for (int a=0; a < size_a; a++)
    da_ta[a] /= c;
  
  return *this;
}

MATRIX& MATRIX::operator -= (MATRIX m)
{
  if (size_x != m.size_x || size_y != m.size_y)
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::operator-= : Inconsistent MATRIX sizes in subtraction!");
#endif
    printf( "MATRIX::operator-= : Inconsistent MATRIX sizes in subtraction!");
    return *this;
  }
  
  for (int a=0; a < size_a; a++)
    da_ta[a] -= m.da_ta[a];
  
  if(m.kill)
    m.Delete();
  
  return *this;
}


MATRIX MATRIX::operator+(MATRIX A)
{
  MATRIX X;
  
  
  if((size_y!=A.size_y)||(A.size_y!=size_y))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::operator + : Incompatible matrix sizes!");
#endif
    printf("MATRIX:: operator + : Incompatible matrix sizes {%d,%d} {%d,%d}\n",size_x,size_y,A.size_x,A.size_y);
    return A;
  }
  
  
  X.Create(size_x,size_y);
  X.kill = true;
  
  for(int a = 0; a < size_a; a++ )  X.da_ta[a] = da_ta[a] + A.da_ta[a];      
  
  if(A.kill)
    A.Delete();
  if(kill)
    Delete();
  
  
  return X;
}

MATRIX MATRIX::operator-(MATRIX A)
{
  MATRIX X;
  
  if((size_y!=A.size_y)||(A.size_y!=size_y))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::operator - : Incompatible matrix sizes!");
#endif
    printf("MATRIX:: operator - : Incompatible matrix sizes {%d,%d} {%d,%d}\n",size_x,size_y,A.size_x,A.size_y);
    return A;
  }
  
  X.Create(size_x,size_y);
  X.kill = true;
  
  for(int a = 0; a < size_a; a++ )  X.da_ta[a] = da_ta[a] - A.da_ta[a];
  
  if(A.kill)
    A.Delete();
  if(kill)
    Delete();
  
  return X;
}
/*
MATRIX operator*(double c, const MATRIX &A)
{
int a;

  MATRIX X;
  X.Create(A.size_x,A.size_y);
  
    for(a=0;a<A.size_a;a++)
    X.da_ta[a]=A.da_ta[a]*c;
    
      return X;
      }
/**/

MATRIX MATRIX::operator*( MATRIX A)
{
  MATRIX X;
  
  
  if(size_y!=A.size_x)
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::operator * : Incompatible matrix sizes!");
#endif
    printf("MATRIX::operator * : Incompatible matrix sizes {%d,%d}{%d,%d}\n",size_x,size_y,A.size_x,A.size_y);
    return X;
  }
  
  int _size_x,_size_y;
  
  _size_x=size_x;
  _size_y=A.size_y;
  
  X.Create(_size_x,_size_y);
  X.kill = true;
  
  int i,j,k;
  
  for( i = 0; i < _size_x; i++ )
  {
    for( j = 0; j < _size_y; j++)
    {
      X.da_ta[i * X.size_y + j] = 0.0;
      for( k = 0; k < size_y; k++ )
      {
        X.da_ta[i * X.size_y + j] += da_ta[i * size_y + k] * A.da_ta[k * A.size_y + j];
      }
    }
  }
  
  if(A.kill)
    A.Delete();
  if(kill)
    Delete();
  
  return X;
}

MATRIX MATRIX::operator/(MATRIX A)
{  
  return operator *(!(MATRIX)A);
}

MATRIX  MATRIX::operator~()
{
  MATRIX X;
  
  X.Create(size_y,size_x);
  
  int x,y;
  
  for( x = 0; x < size_x; x++ )
  {
    for( y = 0; y < size_y; y++)
    {
      X[y][x] = da_ta[ x * size_y + y];
    }
  }
  X.kill =true; 
  
  return X;
}

MATRIX  MATRIX::operator!()
{
  MATRIX X;
  X.Copy(this);
  X.Invert();
  X.kill = true;
  
  return X;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//     vybery dat
//
////////////////////////////////////////////////////////////////////////////////////////////////////



MATRIX * MATRIX::GetRow(int x)
{
  if(x>=size_x) 
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::GetRow(int x) : Index out of range!");
#endif
    return NULL;
  }
  
  MATRIX *r;
  r=new MATRIX();
  
  r->Create(1,size_y);
  
  for(int i=0;i<size_y;i++)
    r->da_ta[i] = da_ta[ x * size_y + i];
  
  return r;
}

double * MATRIX::GetRowDouble(int x)
{
  if(x>=size_x) return NULL;
  return &da_ta[x * size_y];
}

/*
double & MATRIX::_data(int i, int j)
{
return da_ta[i * size_y + j];
}
/**/

int MATRIX::GetRowR(int kerej, MATRIX *x)
{
  if(kerej>=size_x)
    return RETURN_FAIL;  
  
  if(x==NULL)
    x = new MATRIX();  
  else
  {
    if((x->size_x!=1)||(x->size_y!=size_y))
    {
      x->Delete();
      x->Create(1,size_y);
    }
  }
  
  if(x->da_ta==NULL)
    return RETURN_FAIL;
  
  for(int i=0;i<size_y;i++)
    x->da_ta[i] = da_ta[kerej * size_y + i];  
  
  return RETURN_OK;
}




////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      nacitani a ukladani matic
//
////////////////////////////////////////////////////////////////////////////////////////////////////


int MATRIX::LoadBin(char *filename, int _size_x)
{
  FILE *in=NULL;
  int x,y,iRet;
  int _size_y=0;
  
  if ((in = fopen(filename, "rb"))== NULL) 
    return RETURN_CANNOT_OPEN_FILE;
  
  if(_size_x<=0)
    return RETURN_FAIL;
  
  
  //
  // zjistovani velikosti
  // 
  
 	float *tmp = new float [_size_x];
  while(fread(tmp, sizeof(tmp), _size_x, in) == (size_t) _size_x) _size_y++;
  delete tmp;
  
  rewind(in);
  
  iRet = Create(_size_y,_size_x);
  if(iRet!=RETURN_OK)
    return iRet;
  
  float pom=0;
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      if(fread((void*)&pom,sizeof(float),1,in)!=1)
        return RETURN_FAIL;
      
      da_ta[x * size_y + y] = (double)pom;
    }
  }
  fclose(in);   
  return RETURN_OK;
  
}

int MATRIX::LoadBinD(char *filename)
{
  FILE *in=NULL;
  int x,y,iRet;
  
  if ((in = fopen(filename, "rb"))== NULL) 
    return RETURN_CANNOT_OPEN_FILE;
  
  if(fread((void*)&size_x,sizeof(int),1,in)!=1)
    return RETURN_FAIL;
  
  if(fread((void*)&size_y,sizeof(int),1,in)!=1)
    return RETURN_FAIL;
  
  iRet = Create(size_x,size_y);
  if(iRet!=RETURN_OK)
    return iRet;
  
  double pom=0;
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      if(fread((void*)&pom,sizeof(double),1,in)!=1)
        return RETURN_FAIL;
      
      da_ta[x*size_y + y] = pom;
    }
  }
  fclose(in);   
  return RETURN_OK;
}

int MATRIX::LoadBinF(char *filename)
{
  FILE *in=NULL;
  int x,y,iRet;
  
  if ((in = fopen(filename, "rb"))== NULL) 
    return RETURN_CANNOT_OPEN_FILE;
  
  
  if(fread((void*)&size_x,sizeof(int),1,in)!=1)
    return RETURN_FAIL;
  
  if(fread((void*)&size_y,sizeof(int),1,in)!=1)
    return RETURN_FAIL;
  
  iRet = Create(size_x,size_y);
  if(iRet!=RETURN_OK)
    return iRet;
  
  float pom=0;
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      if(fread((void*)&pom,sizeof(float),1,in)!=1)
        return RETURN_FAIL;      
      da_ta[ x*size_y + y ] = (double) pom;
    }
  }
  fclose(in);   
  return RETURN_OK;
}

int MATRIX::SaveBinF(char *filename)
{
  
  FILE *out=NULL;
  int x,y;
  
  
  if ((out = fopen(filename, "wb"))== NULL)
    return RETURN_FAIL;	
  
  if(fwrite((void*)&size_x,sizeof(int),1,out)!=1)
    return RETURN_FAIL;	
  
  if(fwrite((void*)&size_y,sizeof(int),1,out)!=1)
    return RETURN_FAIL;	
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      float pom=(float)da_ta[x * size_y + y];
      
      if(fwrite((void*)&pom,sizeof(float),1,out)!=1)
      {     
        return RETURN_FAIL;
      }
    }
  }  
  fclose(out);
  
  return RETURN_OK;
}

int MATRIX::SaveBinD(char *filename)
{
  
  FILE *out=NULL;
  int x,y;
  
  
  if ((out = fopen(filename, "wb"))== NULL)
  {
    return RETURN_FAIL;	
  } 
  
  if(fwrite((void*)&size_x,sizeof(int),1,out)!=1)
    return RETURN_FAIL;	
  
  if(fwrite((void*)&size_y,sizeof(int),1,out)!=1)
    return RETURN_FAIL;	
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      double pom=(double)da_ta[x * size_y + y];
      
      if(fwrite((void*)&pom,sizeof(double),1,out)!=1)
      {     
        return RETURN_FAIL;
      }
    }
  }  
  fclose(out);
  
  return RETURN_OK;
}

int MATRIX::LoadBin(char *filename, int size_x, int size_y)
{
  FILE *in=NULL;
  int x,y,iRet;
  
  if ((in = fopen(filename, "rb"))== NULL) 
    return RETURN_CANNOT_OPEN_FILE;
  
  if((size_x<=0)||(size_y<=0))
    return RETURN_FAIL;
  
  
  iRet = Create(size_x,size_y);
  if(iRet!=RETURN_OK)
    return iRet;
  
  double pom=0;
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      if(fread((void*)&pom,sizeof(double),1,in)!=1)
        return RETURN_FAIL;
      
      da_ta[x * size_y + y] = pom;
    }
  }
  fclose(in);   
  return RETURN_OK;
  
}

int MATRIX::LoadHTK(char *filename)
{
  
  FILE *in=NULL;
  
  if ((in = fopen(filename, "rb"))== NULL) 
    return RETURN_CANNOT_OPEN_FILE;
  
  if(in==NULL)
    return RETURN_FAIL;
  
  
  
  long  delka;
  long  perioda;
  short pbytu;
  short kind;
  
  char *d;
  
  
  d=(char *)&delka;
  
  
  fread(&d[3],sizeof(char),1,in);
  fread(&d[2],sizeof(char),1,in);
  fread(&d[1],sizeof(char),1,in);
  fread(&d[0],sizeof(char),1,in);
  
  
  d=(char *)&perioda;
  
  fread(&d[3],sizeof(char),1,in);
  fread(&d[2],sizeof(char),1,in);
  fread(&d[1],sizeof(char),1,in);
  fread(&d[0],sizeof(char),1,in);
  
  d=(char *)&pbytu;
  
  fread(&d[1],sizeof(char),1,in);
  fread(&d[0],sizeof(char),1,in);
  
  
  d=(char *)&kind;
  
  fread(&d[1],sizeof(char),1,in);
  fread(&d[0],sizeof(char),1,in);
  
  
  int vect_size=0;
  
  vect_size=pbytu/4;
  
  
  Create(delka,vect_size);
  
  float pom=0;
  
  for(long j=0;j<delka;j++)
  {
    for(int i=0;i<vect_size;i++)
    {
      d=(char *)&(pom);
      
      fread(&d[3],sizeof(char),1,in);
      fread(&d[2],sizeof(char),1,in);
      fread(&d[1],sizeof(char),1,in);
      fread(&d[0],sizeof(char),1,in);
      
      da_ta[j * size_y + i] = (double) pom;
    }
  }
  
  fclose(in);
  return RETURN_OK;
}

int MATRIX::LoadTxt(char *filename)
{
#define MAX_LINE 10000
  
  int iRet;
  FILE *in=NULL;
  
  if ((in = fopen(filename, "r"))== NULL) 
    return RETURN_CANNOT_OPEN_FILE;  
  
  long iPom=0;
  char pom[MAX_LINE];
  
  while(1)
  {
    if(fgets(pom,MAX_LINE,in)==NULL) break;
    iPom++;
  }
  fclose(in);
  
  if ((in = fopen(filename, "r"))== NULL)
  {
    printf("soubor se nepodarilo zalozit\n");
    return RETURN_CANNOT_OPEN_FILE;	
  }
  
  int radka=0;
  double fpom;
  char *_s = pom;
  char ss[50];
  
  fgets(pom,MAX_LINE,in);
  while(1)
  {
    if(sscanf(_s,"%f",&fpom)!=1) break;
    radka++;
    sscanf(_s,"%s",ss);   
    _s+=strlen(ss)+1;
  }
  fclose(in);
  
  iRet=Create(iPom,radka);
  if(iRet!=RETURN_OK)
    return iRet;
  
  if ((in = fopen(filename, "r"))== NULL)
  {
    printf("soubor se nepodarilo zalozit\n");
    return RETURN_FAIL;	
  }
  
  
  for(long x=0;x<iPom;x++)
  {
    fgets(pom,MAX_LINE,in);
    
    double fpom;
    int srac=0;
    _s=pom;
    
    for(int y=0;y<radka;y++)
    {
      if(sscanf(_s,"%lf",&fpom)!=1) break;
      srac++;
      sscanf(_s,"%s",ss);   
      _s+=strlen(ss)+1;
      da_ta[x * size_y + y] = fpom;
    }
    if(srac!=radka)
      return RETURN_FAIL;
  }
  
  return RETURN_OK;
}


int MATRIX::SaveTxt(char *filename)
{
  FILE *out=NULL;
  int x,y;
  
  if ((out = fopen(filename, "w"))== NULL)
  {
    return RETURN_FAIL;	
  }  
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      fprintf(out,"%e ",da_ta[x * size_y + y]);
    }
    fprintf(out,"\n");
    
  }  
  fclose(out);
  
  return RETURN_OK;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      tisk matic
//
////////////////////////////////////////////////////////////////////////////////////////////////////



void MATRIX::PrintDiag()
{
  int x;
  if((size_x==size_y)&&(size_x==0))
  {
    printf("CHYBA\n");
    return;
  }
  
  for(x=0;x<size_x;x++)
  {
    printf("%10.5f ",da_ta[x * size_y + x]);  
  }
  printf("\n");
  
}

void MATRIX::PrintDiagE()
{
  int x;
  if((size_x==size_y)&&(size_x==0))
  {
    printf("CHYBA\n");
    return;
  }
  
  for(x=0;x<size_x;x++)
  {
    printf("%+e ",da_ta[x * size_y + x]);  
  }
  printf("\n");
  
}

void MATRIX::Print()
{
  int x,y;
  if((size_x==size_y)&&(size_x==0))
    REPORT_ERROR( "MATRIX::Print(): No matrix!");
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      printf("%10.5f ",da_ta[x * size_y + y]);
    }
    printf("\n");
  }  
  printf("\n");
}

void MATRIX::PrintE()
{
  int x,y;
  
  for(x=0;x<size_x;x++)
  {
    for(y=0;y<size_y;y++)
    {
      printf("%+e ",da_ta[x * size_y + y]);
    }
    printf("\n");
  }
  printf("\n");
  
}

void MATRIX::PrintM()
{
  int x,y;
  
  printf("[ ");
  for(x=0;x<size_x;x++)
  {  
    if(x!=0) printf("  ");
    for(y=0;y<size_y;y++)
    {
      printf("%+e ",da_ta[x * size_y + y]);
    }
    printf(";\n");
  }
  printf("]\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      slozitejsi fce
//
////////////////////////////////////////////////////////////////////////////////////////////////////









double MATRIX::LogDetSPD()
{
  double D=0.0;
  
  if(size_x==1)
  {
    for (int j=0; j<size_y; j++)  D += log(da_ta[j]);
    return D;
  }
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::LogDet(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::LogDet(): Determinant a non-square MATRIX!");
    return 0.0;
  }  
  
  
  MATRIX l;  // Lower Tri Choleski Matrix
  int j,n;
  int iRet;
  
  n = size_x;
  l.Create(n,n);
  
  iRet  = Choleski(l);
  if (iRet==RETURN_OK)
  {
    for (j=0; j<n; j++)
    {
      D += 2*log(l[j][j]);
    }
  }
  else
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::LogDet(): LU decomposition not succesful!");
#endif
    printf("MATRIX::LogDet(): LU decomposition not succesful!");
    return 0.0;
  }
  
  l.Delete();
  return D;
}

double MATRIX::Cofact (int row, int col) // calculate the cofactor of a MATRIX for a given element
{
  int i,i1,j,j1;
  static MATRIX tempA,tempB;
  
  //MATRIX temp;
  
  if((row==-1)&&(col==-1))
  {
    tempA.Delete();
    tempB.Delete();
    return 0.0;
  }
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::Cofact(): Cofactor of a non-square MATRIX!");
#endif
    printf("MATRIX::Cofact(): Cofactor of a non-square MATRIX!");
    return 0.0;
  }
  
  if (row > size_x || col > size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::Cofact(): Index out of range!");
#endif
    printf("MATRIX::Cofact(): Index out of range!");
    return 0.0;
  }
  
  double  cof;
  
  //if(GetCurrentThreadId()==_threadA)
//  if(ManagedThreadId()==_threadA)
  {
    /**/
    
    if((tempA.size_x!=size_x-1)||(tempA.size_y!=size_y-1))
      tempA.Create(Row()-1,Col()-1);
    
    for (i=i1=0; i < size_x; i++)
    {
      if (i == row)
        continue;
      for (j=j1=0; j < size_y; j++)
      {
        if (j == col)
          continue;
        tempA.da_ta[i1*tempA.size_y + j1] = da_ta[i * size_y + j];
        j1++;
      }
      i1++;
    }
    //temp.PrintM();
    //temp.SaveTxt("temp.txt");
    
    cof = tempA.Det();
    if ((row+col)%2 == 1)
      cof = -cof;
    
  }
/*  else
  {
    if((tempB.size_x!=size_x-1)||(tempB.size_y!=size_y-1))
      tempB.Create(Row()-1,Col()-1);
    
    for (i=i1=0; i < size_x; i++)
    {
      if (i == row)
        continue;
      for (j=j1=0; j < size_y; j++)
      {
        if (j == col)
          continue;
        tempB.da_ta[i1*tempB.size_y + j1] = da_ta[i * size_y + j];
        j1++;
      }
      i1++;
    }
    //temp.PrintM();
    //temp.SaveTxt("temp.txt");
    
    cof = tempB.Det();
    if ((row+col)%2 == 1)
      cof = -cof;
    
  }
  /**/
  //temp.Delete();
  
  return cof;
}
double MATRIX::Det(void)
{   
  double det=1.0;
  
  if(size_x==1)     /// ha ha tu to je tu uplne blbe tady je to diagonalni srac
  {
    for (int i=0; i<size_y; i++)
    {
      det*=da_ta[i];
    }
    return det;
  }
  
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::Det(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::Det(): Determinant a non-square MATRIX!\n");
    return 0.0;
  }
  
  switch( TYPE ) 
  {
  case GEN:
    {
#ifdef ___MKL___  
      det = _Det_IntGen();
#else
      det = _Det_Gen();
#endif      
      break;
    }
  case SYM:
    {
#ifdef ___MKL___  
      det = _Det_IntGen();
#else
      det = _Det_Sym(); // svine nefunguje det = _Det_IntelSym();
#endif      
      break;
    }
  case DIA:
    {
      det = 1.0;
      for(int i=0;i<size_x;i++)  det*=da_ta[i * size_y + i];
      break;
    }
  default :
    {      
#ifdef __VYHOD_CHYBY__  
      REPORT_ERROR( "MATRIX::Det() : Incompatible matrix type!");
#endif
      printf("MATRIX::Det() : Incompatible matrix type\n");
      return RETURN_FAIL;      
    }
  }
  
  return det;
}

double MATRIX::_Det_IntGen() 
{
  double det=1.0;
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_DetIntGen(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::_DetIntGen(): Determinant a non-square MATRIX!\n");
    return 0.0;
  }
#ifdef ___MKL___    
  //  static double _pom[___MAX_SIZE___];
  
  //  double _pom[___MAX_SIZE___];
  //  int   _ipiv[100000];
  
  //  int *ipiv = NULL;
  int info; 
  
  //ipiv = new int[16*size_x];
  
  if(GetCurrentThreadId()==_threadA)
  {
    memcpy( _pomA , da_ta , sizeof(double) * size_a);  /// zkopiruje matici
    
    dgetrf(&size_x,&size_y,_pomA,&size_y,_ipivA,&info);
    
    if(info!=0)
      return 0.0;
    
    // for(int ii=0;ii<2*size_x;ii++) printf("%3d ",_ipiv[ii]);printf("\n\n");
    
    
    for(int i=0;i<size_x;i++)
    {
      if(_ipivA[i]!=i+1) 
        det = -det * _pomA[ i * size_y + i];
      else
        det = det * _pomA[ i * size_y + i];
    }
  }
  else  
  {
    memcpy( _pomB , da_ta , sizeof(double) * size_a);  /// zkopiruje matici
    
    dgetrf(&size_x,&size_y,_pomB,&size_y,_ipivB,&info);
    
    if(info!=0)
      return 0.0;
    
    // for(int ii=0;ii<2*size_x;ii++) printf("%3d ",_ipiv[ii]);printf("\n\n");
    
    
    for(int i=0;i<size_x;i++)
    {
      if(_ipivB[i]!=i+1) 
        det = -det * _pomB[ i * size_y + i];
      else
        det = det * _pomB[ i * size_y + i];
    }
  }
  /**/
  
  //delete []ipiv;
  if(info!=0)
  {
    return 0.0;     // nesystemove ale funkcni !! 
    
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_Det_IntelGen(): Unsuccesful LU decomposition!");
#endif
    printf("MATRIX::_Det_IntelGen():  Unsuccesful LU decomposition!\n");
    
  } 
  
  
  
  
  
#else
  det = _Det_Gen();
#endif     
  
  
  return det;
}

double MATRIX::_Det_Gen() // tuto je determinant buh vi vodkud ale umi asi i ty co neumi choleskyho rozklad
{
  int i,j,k;
  double piv,detVal =1.0;
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_DetGen(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::_DetGen(): Determinant a non-square MATRIX!\n");
    return 0.0;
  }
  
  MATRIX temp;
  temp.Create(this);
  
  for (k=0; k < size_x; k++)
  {
    int indx = temp.pivot(k);
	if (indx == -1) {
		temp.Delete();
		return 0;
	}
    if (indx != 0)
      detVal = - detVal;
    detVal = detVal * temp.da_ta[k * temp.size_y + k];
    for (i=k+1; i < size_x; i++)
    {
      piv = temp.da_ta[i * temp.size_y + k] / temp.da_ta[k* temp.size_y + k];
      
      for (j=k+1; j < size_x; j++)
        temp.da_ta[i*temp.size_y + j] -= piv * temp.da_ta[k*temp.size_y + j];
    }
  }  
  temp.Delete();
  return detVal;
}

double MATRIX::LogDet() // tuto je determinant buh vi vodkud ale umi asi i ty co neumi choleskyho rozklad
{
  double det=0.0;
  
  if(size_x==1)     /// ha ha tu to je tu uplne blbe tady je to diagonalni srac
  {    
    for (int i=0; i<size_y; i++)
    {
      det+=log(da_ta[i]);
    }
    return det;
  }
  

  //LOGDET by KW  //nekde to zlobi... !!
  ////////////////////////////////////////////////////////////////////
/*  int i,j,k;
  double piv,detVal = 0.0;
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::LogGen(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::LogGen(): Determinant a non-square MATRIX!\n");
    return 0.0;
  }
  
  MATRIX temp;
  temp.Create(this);
  
  for (k=0; k < size_x; k++)
  {
    //int indx = temp.pivot(k);
 //   if (indx == -1)
 //     return 0;
    //if (indx != 0)
      //detVal = - detVal;
    detVal = detVal + log(fabs(temp.da_ta[k * temp.size_y + k]));
    for (i=k+1; i < size_x; i++)
    {
      piv = temp.da_ta[i * temp.size_y + k] / temp.da_ta[k* temp.size_y + k];
      
      for (j=k+1; j < size_x; j++)
        temp.da_ta[i*temp.size_y + j] -= piv * temp.da_ta[k*temp.size_y + j];
    }
  }  
  temp.Delete();
  return detVal;


  */

//Zkusime prescholeskyho rozklad:

  MATRIX l; 
  double LogDet = 0.0;
  int j,n;
  int iRet;
  
  n = size_x;
  l.Create(n,n);
  iRet  = Choleski(l);  // Lower Tri Choleski Matrix 
  if (iRet==RETURN_OK)
  {
    for (j=0; j<n; j++)
      LogDet += log(fabs(l.da_ta[j *l.size_y + j])) + log(fabs(l.da_ta[j * l.size_y + j]));
  }
  else
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::LogDet(): LU decomposition not succesful!");
#endif
    printf("MATRIX::LogDet(): LU decomposition not succesful!\n");
    return 0.0;
  }
  
  
  l.Delete();
  return (double)(LogDet);  
  
}
//////////////////////////////////////////////////


double MATRIX::_Det_Sym()
{
  double D=0.0;
  if(size_x!=size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_Det_Sym(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::_Det_Sym(): Determinant a non-square MATRIX!\n");
    return 0.0;
  }
  
  
  MATRIX l; 
  double det = 1.0;
  int j,n;
  int iRet;
  
  n = size_x;
  l.Create(n,n);
  
  iRet  = Choleski(l);  // Lower Tri Choleski Matrix 
  if (iRet==RETURN_OK)
  {
    for (j=0; j<n; j++)
      det *=l.da_ta[j *l.size_y + j]*l.da_ta[j * l.size_y + j];
  }
  else
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_Det_Sym(): LU decomposition not succesful!");
#endif
    printf("MATRIX::_Det_Sym(): LU decomposition not succesful!\n");
    return 0.0;
  }
  
  
  l.Delete();
  return (double)(det);  
}


int MATRIX::pivot (int row)  // spocte pivota pro vypocet determinantu
{
  int i, k = int(row);
  double amax,temp;
  
  amax = -1;
  for (i=row; i < size_x; i++)
  {
    if ( (temp = abs( da_ta[i*size_y+ row])) > amax && temp != 0.0)
    {
      amax = temp;
      k = i;
    }
  }
  if (da_ta[k * size_y + row] == double(0))
    return -1;
  
  if (k != int(row))
  {
    double pom;
    for(i=0;i<size_y;i++)
    {
      pom                   = da_ta[k * size_y + i];
      da_ta[k * size_y + i] = da_ta[row * size_y + i];
      da_ta[row * size_y+i] = pom;
    }
    return k;
  }
  return 0;
}

int MATRIX::Invert()    // ha ha tuto asi invertuje matici
{
  int iRet=0;
  
  if (size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::Invert(): Determinant a non-square MATRIX!!");
#endif
    printf("MATRIX::Invert(): Determinant a non-square MATRIX!\n");
    return RETURN_FAIL;
  }
  
  switch( TYPE ) 
  {
  case GEN:
    {
#ifdef ___MKL___  
      iRet = _Invert_IntGen();
#else
      iRet = _Invert_Gen();
#endif      
      break;
    }
  case SYM:
    {
#ifdef ___MKL___  
      iRet = _Invert_IntGen();
#else
      iRet = _Invert_Gen(); // svine nefunguje pro symetrickou
#endif      
      break;
    }
  case DIA:
    {
      for(int i=0;i<size_x;i++)  da_ta[i * size_y + i] = 1/da_ta[i * size_y + i];
      break;
    }
  default :
    {      
#ifdef __VYHOD_CHYBY__  
      REPORT_ERROR( "MATRIX::Invert() : Incompatible matrix type!");
#endif
      printf("MATRIX::Invert() : Incompatible matrix type\n");
      return RETURN_FAIL;      
    }
  }
  
  return iRet;
}

int MATRIX::_Invert_IntGen()    // ha ha tuto asi invertuje matici
{
  if( size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_Invert_IntGen(): Invert a non-square MATRIX!!");
#endif
    printf("MATRIX::_Invert_IntGen(): Invert a non-square MATRIX!\n");
    return RETURN_FAIL; 
  } 
  
#ifdef ___MKL___  
  
  int info;
  int lwork=-1;
  
  
  //  static double _pom[___MAX_SIZE___];
  //double _pom[___MAX_SIZE___];
  //int    ipiv[___MAX_SIZE___];
  
  lwork = 16*size_x;
  
  if(GetCurrentThreadId()==_threadA)
  {
    dgetrf(&size_x,&size_y,da_ta,&size_y,_ipivA,&info);
    
    if(info!=0)
      return RETURN_FAIL;
    
    dgetri(&size_x,da_ta,&size_x,_ipivA,_pomA,&lwork,&info);
  }
  else
  {
    dgetrf(&size_x,&size_y,da_ta,&size_y,_ipivB,&info);
    
    if(info!=0)
      return RETURN_FAIL;
    
    dgetri(&size_x,da_ta,&size_x,_ipivB,_pomB,&lwork,&info);
    
  }
  
  if(info!=0)
    return RETURN_FAIL;
  
  return RETURN_OK;
  
#else
  
  int iRet;
  iRet = _Invert_Gen();
  return iRet;
  
#endif
  
  
}

int MATRIX::_Invert_Gen()    // ha ha tuto asi invertuje matici
{
  int a;
  long k, j, i, n;
  double D = 1.0;
  
  if( size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::_Invert_Gen(): Invert a non-square MATRIX!!");
#endif
    printf("MATRIX::_Invert_Gen(): Invert a non-square MATRIX!\n");
    return RETURN_FAIL;
  }
  
  //double *pom=NULL; pom=new double [size_a];
  
  // static double pom[___MAX_SIZE___];
  
  //double _pom[___MAX_SIZE___];
  
  
//  if(GetCurrentThreadId()==_threadA)
//if(ManagedThreadId()==_threadA)
  {
    
    for(a=0;a<size_a;a++)  _pomA[a]=da_ta[a];
    
    n = size_x;
    for(k=0; k<n; k++)
    {
      for(i=0; i<n; i++)
        for(j=0; j<n; j++)
          if( i!=k && j!=k )
            _pomA[i*n+j] -= _pomA[k*n+j]*_pomA[i*n+k]/_pomA[k*n+k];
          
          for(i=0; i<n; i++)
            for(j=0; j<n; j++)
              if( i==k && i!=j )
                _pomA[i*n+j]= -_pomA[i*n+j]/_pomA[k*n+k];
              
              for(i=0; i<n; i++)
                for(j=0; j<n; j++)
                  if( j==k && i!=j )
                    _pomA[i*n+j]= _pomA[i*n+j]/_pomA[k*n+k];
                  D *=(double)( _pomA[k*n+k]);
                  _pomA[k*n+k]=(double)( 1.0/_pomA[k*n+k]);
    }
    
    
    
    for(a=0;a<size_a;a++) da_ta[a]=_pomA[a];
  }
/*  else
  {
    for(a=0;a<size_a;a++)  _pomB[a]=da_ta[a];
    
    n = size_x;
    for(k=0; k<n; k++)
    {
      for(i=0; i<n; i++)
        for(j=0; j<n; j++)
          if( i!=k && j!=k )
            _pomB[i*n+j] -= _pomB[k*n+j]*_pomB[i*n+k]/_pomB[k*n+k];
          
          for(i=0; i<n; i++)
            for(j=0; j<n; j++)
              if( i==k && i!=j )
                _pomB[i*n+j]= -_pomB[i*n+j]/_pomB[k*n+k];
              
              for(i=0; i<n; i++)
                for(j=0; j<n; j++)
                  if( j==k && i!=j )
                    _pomB[i*n+j]= _pomB[i*n+j]/_pomB[k*n+k];
                  D *=(double)( _pomB[k*n+k]);
                  _pomB[k*n+k]=(double)( 1.0/_pomB[k*n+k]);
    }

    
    
    for(a=0;a<size_a;a++) da_ta[a]=_pomB[a];
  }
      */
  //delete []pom;
  
  return RETURN_OK;
}

int MATRIX::Choleski(MATRIX &L)// Choleski: Place lower triangular choleski factor of A in L  Return false if matrix singular or not +definite 
{ 
  
  if( size_x != size_y)
  {
#ifdef __VYHOD_CHYBY__
    REPORT_ERROR( "MATRIX::Choleski(): Invert a non-square MATRIX!!");
#endif
    printf("MATRIX::Choleski(): Invert a non-square MATRIX!\n");
    return RETURN_FAIL;
  }
  
  MATRIX *A=this;  
  
  int size,i,j,k;
  double sum;
  
  if(size_x!=size_y)
    return RETURN_FAIL;
  
  size = size_x;
  for (i=1; i<=size; i++)
    
    for (j=1; j<=i; j++)
    {
      sum=da_ta[(i-1)*size_y + j-1];
      for (k=1; k<j; k++)
        sum -= (L[i-1][k-1]*L[j-1][k-1]);
      if ((i==j)&&(sum<=0.0)) 
        return RETURN_FAIL;
      else if (i==j)
        sum = sqrt(sum);
      else if (L[j-1][j-1]==0.0)
        return RETURN_FAIL;
      else
        sum /= L[j-1][j-1];
      L[i-1][j-1] = sum;
    }
    
    for (i=1; i<=size; i++)
      for (j=i+1; j<=size; j++) 
        L[i-1][j-1] = 0.0;
      
      return RETURN_OK;      
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      katastrofalni funkce
//
////////////////////////////////////////////////////////////////////////////////////////////////////




/*
Computes all eigenvalues and eigenvectors of a real symmetric
matrix a[0..n-1][0..n-1]. On output, elements of a above the diagonal
are destroyed. d[0..n-1] returns the eigenvalues of a.
v[0..n-1][0..n-1] is a matrix whose columns contain, on output,
the normalized eigenvectors of a.
nrot returns the number of Jacobi rotations that were required.
*/

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau);
#define MAX_ITER_JACOBI 50


int MATRIX::Jacobi(MATRIX *D, MATRIX *V, long *nrot)
{
  long j,iq,ip,i;
  double tresh,theta,tau,t,sm,s,h,g,c,*b=NULL,*z=NULL;
  void nrerror();
  
  
  int iSize = size_x;
  long n = iSize;
  
  double **a;
  double **v;
  
  a = new double * [iSize];
  v = new double * [iSize];
  
  for(i=0;i<iSize;i++)
  {
    a[i] = new double [iSize];
    v[i] = new double [iSize];
  }
  
  for(i=0;i<iSize;i++)
  {
    for(j=0;j<iSize;j++)
    {
      a[i][j] = (*this)[i][j];
    }
  }
  
  
  double *d;
  d=new double [iSize];
  D->SetZero();
  
  
  
  ///////////////////////////////////////////////
  
  b= new double [n];
  z= new double [n];
  
  /* Initialize to the identity matrix */
  for (ip=0;ip<n;ip++) 
  {
    for (iq=0;iq<n;iq++) v[ip][iq]=0.0;
    v[ip][ip]=1.0;
  }
  for (ip=0;ip<n;ip++) 
  {
    b[ip]=d[ip]=a[ip][ip];
    z[ip]=0.0;
  }
  *nrot=0;
  for (i=0; i<MAX_ITER_JACOBI; i++) 
  {
    sm=0.0;
    for (ip=0;ip<n-1;ip++) 
    {
      for (iq=ip+1;iq<n;iq++)
        sm +=(double) fabs(a[ip][iq]);
    }
    if (sm == 0.0) 
    {
      delete []b;
      delete []z;
      for(i=0;i<iSize;i++)
      {
        (*D)[i][i] = d[i];
        for(j=0;j<iSize;j++)
        {
          (*V)[i][j] = v[i][j];
        }
      }      
      return RETURN_OK;
    }
    if (i < 4)
      tresh=(double)(0.2*sm/(n*n));
    else
      tresh=0.0;
    for (ip=0;ip<n-1;ip++) 
    {
      for (iq=ip+1;iq<n;iq++) 
      {
        g=(double)(100.0*fabs(a[ip][iq]));
        if (i > 4 && (double)(fabs(d[ip])+g) == (double)fabs(d[ip])&& (double)(fabs(d[iq])+g) == (double)fabs(d[iq]))
        {
          a[ip][iq]=0.0;
        }
        else 
          
          if (fabs(a[ip][iq]) > tresh) 
          {
            h=d[iq]-d[ip];
            if ((double)(fabs(h)+g) == (double)fabs(h))
            {
              t=(a[ip][iq])/h;
            }
            else 
            {
              theta=(double)(0.5*h/(a[ip][iq]));
              t=(double)(1.0/(fabs(theta)+sqrt(1.0+theta*theta)));
              if (theta < 0.0) t = -t;
            }
            c=(double)(1.0/sqrt(1+t*t));
            s=t*c;
            tau=(double)(s/(1.0+c));
            h=(double)(t*a[ip][iq]);
            z[ip] -= h;
            z[iq] += h;
            d[ip] -= h;
            d[iq] += h;
            a[ip][iq]=0.0;
            for (j=0;j<=ip-1;j++) {
              ROTATE(a,j,ip,j,iq)
            }
            for (j=ip+1;j<=iq-1;j++) {
              ROTATE(a,ip,j,j,iq)
            }
            for (j=iq+1;j<n;j++) {
              ROTATE(a,ip,j,iq,j)
            }
            for (j=0;j<n;j++) {
              ROTATE(v,j,ip,j,iq)
            }
            ++(*nrot);
          }
      }
      
    }
    for (ip=0;ip<n;ip++) {
      b[ip] += z[ip];
      d[ip]=b[ip];
      z[ip]=0.0;
    }
  }
  printf("Too many iterations in routine JACOBI");
  return RETURN_FAIL;
}


#define RADIX 2.0


int MATRIX::Balance()
{
  long n, last,j,i;
  double s,r,g,f,c,sqrdx;
  double **a = NULL;
  
  
  if(size_x!=size_y)
    return RETURN_FAIL;
  
  
  n = this->size_x; 
  
  a = new double * [n];
  for(i=0;i<n;i++)
  {
    a[i] = new double [n];
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      a[i][j] = (*this)[i][j];
    }
  }
  
  sqrdx=RADIX*RADIX;
  last=0;
  while (last == 0) 
  {
    last=1;
    for (i=1;i<=n;i++) 
    {
      r=c=0.0;
      for (j=1;j<=n;j++)
        if (j != i) 
        {
          c +=(double)( fabs(a[j-1][i-1]));
          r +=(double)( fabs(a[i-1][j-1]));
        }
        if (c && r) 
        {
          g=(double)(r/RADIX);
          f=1.0f;
          s=c+r;
          while (c<g) 
          {
            f *= RADIX;
            c *= sqrdx;
          }
          g=(double)(r*RADIX);
          while (c>g) {
            f /= RADIX;
            c /= sqrdx;
          }
          if ((c+r)/f < 0.95*s) 
          {
            last=0;
            g=1.0f/f;
            for (j=1;j<=n;j++) a[i-1][j-1] *= g;
            for (j=1;j<=n;j++) a[j-1][i-1] *= f;
          }
        }
    }
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      (*this)[i][j] = (double) a[i][j];
    }
  }
  
  for( i = 0; i < n; i++ )
  {
    delete [] a[i];
  }
  delete  []a;
  return RETURN_OK;
}

int MATRIX::Elmhes()
{
  long n,m,j,i;
  double y,x,tmp;
  double **a = NULL;
  
  if(size_x!=size_y)
    return RETURN_FAIL;
  
  
  n = this->size_x; 
  
  a = new double * [n];
  for(i=0;i<n;i++)
  {
    a[i] = new double [n];
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      a[i][j] = (*this)[i][j];
    }
  }
  
  
  for (m=2;m<n;m++) 
  {
    x=0.0;
    i=m;
    for (j=m;j<=n;j++) 
    {
      if (fabs(a[j-1][m-2]) > fabs(x)) 
      {
        x=a[j-1][m-2];
        i=j;
      }
    }
    if (i != m) 
    {
      for (j=m-1;j<=n;j++)
      {
        tmp=a[i-1][j-1]; 
        a[i-1][j-1]=a[m-1][j-1];
        a[m-1][j-1]=tmp;
      }
      for (j=1;j<=n;j++)
      {
        tmp=a[j-1][i-1]; 
        a[j-1][i-1]=a[j-1][m-1]; 
        a[j-1][m-1]=tmp;
      }
    }
    if (x) 
    {
      for (i=m+1;i<=n;i++)
      {
        y=a[i-1][m-2];
        if (y) 
        {
          y /= x;
          a[i-1][m-2]=y;
          for (j=m;j<=n;j++)
            a[i-1][j-1] -= y*a[m-1][j-1];
          for (j=1;j<=n;j++)
            a[j-1][m-1] += y*a[j-1][i-1];
        }
      }
    }
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      (*this)[i][j] = (double) a[i][j];
    }
  }
  
  for( i = 0; i < n; i++ )
  {
    delete [] a[i];
  }
  delete  []a;
  return RETURN_OK;
}

#define SIGN(a,b) ((b) > 0 ? fabs(a) : -fabs(a))
#define HQR_MAX_ITERATIONS 30	/* Original default is 30 */

int MATRIX::Hqr(double *wr, double *wi)
{
  long n,nn,m,l,k,j,its,i,mmin;
  double z,y,x,w,v,u,t,s,r,q,p,anorm;
  double **a = NULL;
  
  
  if(size_x!=size_y)
    return RETURN_FAIL;
  
  
  n = this->size_x; 
  
  a = new double * [n];
  for(i=0;i<n;i++)
  {
    a[i] = new double [n];
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      a[i][j] = (*this)[i][j];
    }
  }
  
  
  anorm=(double)fabs(a[0][0]);
  for (i=2;i<=n;i++)
    for (j=(i-1);j<=n;j++)
      anorm +=(double)fabs(a[i-1][j-1]);
    nn=n;
    t=0.0;
    while (nn >= 1) 
    {
      its=0;
      do 
      {
        for (l=nn;l>=2;l--) 
        {
          s=(double)(fabs(a[l-2][l-2])+fabs(a[l-1][l-1]));
          if (s == 0.0) s=anorm;
          if ((double)(fabs(a[l-1][l-2]) + s) == s) break;
        }
        x=a[nn-1][nn-1];
        if (l == nn) 
        {
          wr[nn-1]=x+t;
          wi[nn-1]=0.0; nn--;
        } else {
          y=a[nn-2][nn-2];
          w=a[nn-1][nn-2]*a[nn-2][nn-1];
          if (l == (nn-1))
          {
            p=0.5f*(y-x);
            q=p*p+w;
            z=double(sqrt(fabs(q)));
            x += t;
            if (q >= 0.0) 
            {
              z=(double)(p+SIGN(z,p));
              wr[nn-2]=wr[nn-1]=x+z;
              if (z) wr[nn-1]=x-w/z;
              wi[nn-2]=wi[nn-1]=0.0;
            } else 
            {
              wr[nn-2]=wr[nn-1]=x+p;
              wi[nn-2]= -(wi[nn-1]=z);
            }
            nn -= 2;
          } else 
          {
            if (its == HQR_MAX_ITERATIONS)
            { 
              printf("Too many iterations in HQR\n"); 
              return RETURN_FAIL; 
            }
            if (its == 10 || its == 20) 
            {
              t += x;
              for (i=1;i<=nn;i++) a[i-1][i-1] -= x;
              s=(double)(fabs(a[nn-1][nn-2])+fabs(a[nn-2][nn-3]));
              y=x=0.75f*s;
              w = -0.4375f*s*s;
            }
            ++its;
            for (m=(nn-2);m>=l;m--) 
            {
              z=a[m-1][m-1];
              r=x-z;
              s=y-z;
              p=(r*s-w)/a[m][m-1]+a[m-1][m];
              q=a[m][m]-z-r-s;
              r=a[m+1][m];
              s=(double)(fabs(p)+fabs(q)+fabs(r));
              p /= s;
              q /= s;
              r /= s;
              if (m == l) break;
              u=(double)(fabs(a[m-1][m-2])*(fabs(q)+fabs(r)));
              v=(double)(fabs(p)*(fabs(a[m-2][m-2])+fabs(z)+fabs(a[m][m])));
              if ((double)(u+v) == v) break;
            }
            for (i=m+2;i<=nn;i++) 
            {
              a[i-1][i-3]=0.0;
              if  (i != (m+2)) 
                a[i-1][i-4]=0.0;
            }
            for (k=m;k<=nn-1;k++) 
            {
              if (k != m) 
              {
                p=a[k-1][k-2];
                q=a[k][k-2];
                r=0.0;
                if (k != (nn-1)) r=a[k+1][k-2];
                x=(double)(fabs(p)+fabs(q)+fabs(r));
                if (x) {
                  p /= x;
                  q /= x;
                  r /= x;
                }
              }
              s=(double)(SIGN(sqrt(p*p+q*q+r*r),p));
              if (s) 
              {
                if (k == m) 
                {
                  if (l != m)
                    a[k-1][k-2] = -a[k-1][k-2];
                } else
                  a[k-1][k-2] = -s*x;
                p += s;
                x=p/s;
                y=q/s;
                z=r/s;
                q /= p;
                r /= p;
                for (j=k;j<=nn;j++)
                {
                  p=a[k-1][j-1]+q*a[k][j-1];
                  if (k != (nn-1)) 
                  {
                    p += r*a[k+1][j-1];
                    a[k+1][j-1] -= p*z;
                  }
                  a[k][j-1] -= p*y;
                  a[k-1][j-1] -= p*x;
                }
                mmin = nn<k+3 ? nn : k+3;
                for (i=l;i<=mmin;i++)
                {
                  p=x*a[i-1][k-1]+y*a[i-1][k];
                  if (k != (nn-1)) 
                  {
                    p += z*a[i-1][k+1];
                    a[i-1][k+1] -= p*r;
                  }
                  a[i-1][k] -= p*q;
                  a[i-1][k-1] -= p;
                }
              }
            }
          }
        }
    } 
    while (l < nn-1);
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      (*this)[i][j] = (double) a[i][j];
    }
  }
  
  for( i = 0; i < n; i++ )
  {
    delete [] a[i];
  }
  delete  []a;
  return RETURN_OK;
  
  
}


#define _INFINITY_ (1000000000000.0f)
#define MIN_INITIAL_GROWTH (1000.0)
#define MAX_ITER 8
#define MAX_L_UPDATE 5
#define MAX_INITIAL_ITER 10
#define MIN_DECREASE (0.01)
#define EPSILON (0.00001)
#define MY_MAXINT 32000


double MATRIX::Euclidian_Distance(double* S1,double * S2,long dim )
{
  long i;
  double aux, D = 0.0;
  
  /// showFV( dim, S1 ); showFV( dim, S2 ); 
  for( i = 0; i < dim; i++ )
  {
    aux = S1[i] - S2[i];
    D += aux * aux;
  }
  return( (double)sqrt((double)D) );
}

void MATRIX::get_random(long max,long * ran )
{
  long i;
  double r;
  
  r = (double)rand();
  
  r = r / RAND_MAX  * (double)max;
  i = (long)(r);
  if( i < 0 || i > max )
  {
    printf("get_random> Problems with the random");
    printf(" generator - Check setup! Exit...\n"); exit(1);
  }
  *ran = i;
}

double MATRIX::vector_len(double *V,long dim )
{
  long i;
  double L = 0.0;
  
  for( i = 0; i < dim; i++ )
    L += V[i] * V[i];
  return( (double)sqrt((double)L) );
}

void MATRIX::norm_vector(double * V,long dim )
{
  long i; double len;
  
  len = vector_len( V, dim );
  if( len != 0.0 )
  {
    for( i = 0; i < dim; i++ )
      V[i] /= len;
  }
}

void MATRIX::random_unit_vector(double * x,long dim ) // generate a random vector of length dim with unit size

{
  long i, rand;
  
  for( i = 0; i < dim; i++ )
  {
    get_random( MY_MAXINT, &rand );
    x[i] = ((double)rand - (double)(MY_MAXINT/2)) / (double)MY_MAXINT;
  }
  norm_vector( x, dim );
}

void MATRIX::copy_FV(double * src,double * dest,long dim )
{
  for(long i = 0; i < dim; i++ ) dest[i] = src[i];
}



/*
*  Given a matrix 'M', an Eigenvalue 'eigVal', calculate the
*  the Eigenvector 'eigVec'
*  'nr' is only used to indicate the global rank of the Eigenvalue
*/

int MATRIX::GetEigVect(double *eigVal, MATRIX *eigVect, long nr)
{
  
  double lambdaI, lambdaI1;
  double *xi = NULL, *xi1 = NULL, *y = NULL, *y_init_max = NULL;
  double deltaL, di, di1, len_y, sum_nom, sum_denom, maxGF, decrease;
  long n, i, nrIter = 0, nrInitialIter = 0, nrLambdaUpdates = 0;
  bool done = false, betterEigVal = false;
#ifdef VERIFY
  double eigenDiff;
  Matrix aux1, aux2, aux3;
  init_Matrix( &aux1 ); init_Matrix( &aux2 ); init_Matrix( &aux3 );
  Copy_Matrix( M, &aux1 );
  aux2.row = M->row; aux2.col = 1; Matrix_Alloc( &aux2 );
#endif
  
  
  if(size_x!=size_y)
    return RETURN_FAIL;
  
  n = size_x;;
  
  double *eigVec =NULL;
  
  eigVec = new double [n];
  
  
  /*
  xi = (double*) malloc( n * sizeof(double) ); CHKPTR(xi);
  xi1 = (double*) malloc( n * sizeof(double) ); CHKPTR(xi1);
  y = (double*) malloc( n * sizeof(double) ); CHKPTR(y);
  y_init_max = (double*) malloc( n * sizeof(double) ); CHKPTR(y_init_max);
  
  /**/
  
  xi  = new double [n];
  xi1 = new double [n];
  y   = new double [n];
  y_init_max = new double [n];
  
  
  
  lambdaI = *eigVal;
  
  maxGF = -_INFINITY_;
  do	/* first case */
  {
    nrInitialIter++;
    random_unit_vector( xi, n );
    //linearSolve( M, lambdaI, y, xi );
    LinearSolve(lambdaI,y,xi);
    len_y =(double) vector_len( y, n );
#ifdef DEBUG
    printf("Initial growth factor of y = %f\n", len_y); /**/
#endif
    if( len_y > maxGF )
    {
      maxGF = len_y;
      copy_FV( y, y_init_max, n );
    }
    /* printf("Current maximum initial growth = %f\n", maxGF ); /**/
  }
  while( len_y < MIN_INITIAL_GROWTH && nrInitialIter < MAX_INITIAL_ITER );
  copy_FV( y_init_max, y, n );
  
  nrIter = 0;
  di = 0.0;
  for( i = 0; i < n; i++ )
    xi1[i] = y[i] / len_y;
  
  do
  {
    nrIter++;
    di1 = Euclidian_Distance( xi1, xi, n );
    /* printf("Vector Xi:\n"); debug_vec( xi, n );
    printf("Vector Xi1:\n"); debug_vec( xi1, n );
    printf("di1 = %.30f\n\n", di1 ); /**/
    
    if( di1 < EPSILON || nrIter >= MAX_ITER || nrLambdaUpdates >= MAX_L_UPDATE )
      done = true;
    else
    {
      decrease = (double) fabs( di1 - di);
      if( decrease < MIN_DECREASE )
      {
        /* update lambda */
        nrLambdaUpdates++;
        sum_nom = 0.0; sum_denom = 0.0;
        for( i = 0; i < n; i++ )
        {
          sum_nom += xi[i] * xi[i];
          sum_denom += xi[i] * y[i];
        }
        deltaL = sum_nom / (double)fabs(sum_denom);
        lambdaI1 = lambdaI + deltaL;
#ifdef DEBUG
        printf("UPDATE deltaL=%.40f\n", deltaL );
#endif
        betterEigVal = true;
        lambdaI = lambdaI1;
        if( deltaL == 0.0 )
          done = true;
        else
          nrIter = 0;	/* start new cycle with new eigenvalue */
      }
      if( ! done )
      {
        copy_FV( xi1, xi, n );
        di = di1;
        /* Solve the set of linear equations : (M-lambda*UnitMatrix)*y = xi  */
        //linearSolve( M, lambdaI, y, xi );
        LinearSolve(lambdaI,y,xi);
        len_y =(double) vector_len( y, n );
        for( i = 0; i < n; i++ )
          xi1[i] = y[i] / len_y;
      }
    }
  }
  while( ! done );
  
  copy_FV( xi1, eigVec, n );
  if( betterEigVal )
    *eigVal = lambdaI;
  
#ifdef VERIFY
  for( i = 0; i < n; i++ )
    aux2.Elem[i] = xi1[i];
  
  Mult_Matrix( M, &aux2, &aux3 );
  ScalarMult_Matrix( 1.0/lambdaI, &aux3 );
  
  for( i = 0; i < n; i++ )
    y[i] =(double) aux3.Elem[i];
  eigenDiff = Euclidian_Distance( xi1, y, n );
#ifdef DEBUG
  if( eigenDiff > MAX_DIFF_VERIFY )
  {
    printf("Had a big difference: %f\n", eigenDiff );
    printf("Number of iterations: --- %d ---\nVector X:\n", nrIter);
    for( i = 0; i < n; i++ )
      printf("%.30f\n", xi1[i] );
    printf("Vector should be X:\n"); 
    for( i = 0; i < n; i++ )
      printf("%.30f\n", aux3.Elem[i] );
  }
  else
    printf("\n --------------  V E R I F I E D\n");
  printf(" REASON:\n");
  printf("di1=%f  decrease=%f  nrIter=%d  deltaL=%.20f\n  nrLambdaUpdates=%d\n",
    di1, decrease, nrIter, deltaL, nrLambdaUpdates );
  if( di1 < EPSILON ) printf("di1 < EPSILON\n");
  if( nrIter >= MAX_ITER ) printf("nrIter >= MAX_ITER\n");
  if( deltaL == 0.0 ) printf("deltaL == 0.0\n");
  if( nrLambdaUpdates >= MAX_L_UPDATE )printf("LambdaUpdates >= MAX_L_UPDATE\n");
  printf("-------------------------------------------------------\n");
#endif
  if( eigenDiff > MAX_DIFF_VERIFY )
  { printf("Had trouble calculating eigenvector nr. %d\n", nr+1 ); }
  Matrix_Free( &aux1 ); Matrix_Free( &aux2 ); Matrix_Free( &aux3 );
#endif
  
  //FREE(xi); FREE(xi1); FREE(y); FREE(y_init_max);
  
  delete []xi;
  delete []xi1;
  delete []y;
  delete []y_init_max;
  
  for(i=0;i<n;i++)
  {
    (*eigVect)[i][0] = eigVec[i];
  }
  
  
  delete []eigVec;
  
  return RETURN_OK;
}




#define TINY 1.0e-20;

int MATRIX::ludcmp(double **a,long n,long *indx,double *d)
{
  long i,imax,j,k;
  double big,dum,sum,temp;
  double *vv = NULL;
  
  //if( num_recip_error ) return;
  
  vv = new double [n];
  //vv = (double*) malloc( n * sizeof(double) );
  
  *d=1.0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if ((temp=(double)fabs(a[i][j])) > big) big=temp;
      if (big == 0.0) 
      { 
        printf("Singular matrix in routine LUDCMP");
        return RETURN_FAIL;
      }
      vv[i]=1.0f/big;
  }
  for (j=0;j<n;j++) {
    for (i=0;i<j;i++) {
      sum=a[i][j];
      for (k=0;k<i;k++) sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
    }
    big=0.0;
    for (i=j;i<n;i++) {
      sum=a[i][j];
      for (k=0;k<j;k++)
        sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
      if ( (dum=vv[i]*(double)fabs(sum)) >= big) {
        big=dum;
        imax=i;
      }
    }
    if (j != imax) {
      for (k=0;k<n;k++) {
        dum=a[imax][k];
        a[imax][k]=a[j][k];
        a[j][k]=dum;
      }
      *d = -(*d);
      vv[imax]=vv[j];
    }
    indx[j]=imax;
    if (a[j][j] == 0.0f) a[j][j]=(double)TINY;
    if (j != n-1) {
      dum=1.0f/(a[j][j]);
      for (i=j+1;i<n;i++) a[i][j] *= dum;
    }
  }
  //FREE( vv );
  delete []vv;
  return RETURN_OK;
}
#undef TINY


static void lubksb(double **a,long n,long *indx,double b[])
{
  long i,ii=0,ip,j;
  double sum;
  
  //if( num_recip_error ) return;
  
  for (i=0;i<n;i++) {
    ip=indx[i];
    sum=b[ip];
    b[ip]=b[i];
    if (ii)
      for (j=ii;j<=i-1;j++) sum -= a[i][j]*b[j];
      else if (sum) ii=i;
      b[i]=sum;
  }
  for (i=n-1;i>=0;i--) {
    sum=b[i];
    for (j=i+1;j<n;j++) sum -= a[i][j]*b[j];
    b[i]=sum/a[i][i];
  }
}


//#define GAUSS



void MATRIX::gaussj(double **a,int n,double **b,int m)
{
  int *indxc=NULL,*indxr=NULL,*ipiv=NULL;
  int i,icol,irow,j,k,l,ll;
  double big,dum,pivinv,tmp;
  
  // if( num_recip_error ) return;
  
  //indxc = (int*) malloc( n * sizeof(int) ); CHKPTR( indxc );
  indxc = new int[n];
  
  //indxr = (int*) malloc( n * sizeof(int) ); CHKPTR( indxr );
  indxr = new int[n];
  
  //ipiv = (int*) malloc( n * sizeof(int) ); CHKPTR( ipiv );
  ipiv = new int[n];
  
  
  for (j=0;j<n;j++) ipiv[j]=0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if (ipiv[j] != 1)
        for (k=0;k<n;k++) {
          if (ipiv[k] == 0) {
            if (fabs(a[j][k]) >= big) {
              big=fabs(a[j][k]);
              irow=j;
              icol=k;
            }
          } else if (ipiv[k] > 1) {printf("GAUSSJ: Singular Matrix-1");return;}
        }
        ++(ipiv[icol]);
        if (irow != icol) {
          for (l=0;l<n;l++)
          {tmp=a[irow][l]; a[irow][l]=a[icol][l]; a[icol][l]=tmp;}
          for (l=0;l<m;l++)
          {tmp=b[irow][l]; b[irow][l]=b[icol][l]; b[icol][l]=tmp;}
        }
        indxr[i]=irow;
        indxc[i]=icol;
        if (a[icol][icol] == 0.0) {printf("GAUSSJ: Singular Matrix-2");return;}
        pivinv=1.0/a[icol][icol];
        a[icol][icol]=1.0;
        for (l=0;l<n;l++) a[icol][l] *= pivinv;
        for (l=0;l<m;l++) b[icol][l] *= pivinv;
        for (ll=0;ll<n;ll++)
          if (ll != icol) {
            dum=a[ll][icol];
            a[ll][icol]=0.0;
            for (l=0;l<n;l++) a[ll][l] -= a[icol][l]*dum;
            for (l=0;l<m;l++) b[ll][l] -= b[icol][l]*dum;
          }
  }
  for (l=n-1;l>=0;l--) {
    if (indxr[l] != indxc[l])
      for (k=0;k<n;k++)
      {tmp=a[k][indxr[l]]; a[k][indxr[l]]=a[k][indxc[l]]; a[k][indxc[l]]=tmp;}
  }
  //FREE( ipiv ); FREE( indxr ); FREE( indxc );
  delete []ipiv;
  delete []indxr;
  delete []indxc;
}





#define LU_DECOMP


int MATRIX::LinearSolve(double eigVal, double *y, double *x)
{
  
  double **a = NULL, **b = NULL, d, *bVec = NULL;
  long n, m, i, j, *indx = NULL;
  
  
  n = size_x; 
  m = 1;
  
  a = new double * [n];
  for(i=0;i<n;i++)
  {
    a[i] = new double [n];
  }
  
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      a[i][j] = (*this)[i][j];
    }
  }
  /*
  a = (double**) malloc( n * sizeof(double*) );
  for( i = 0; i < n; i++ )
  a[i] = (double*) malloc( n * sizeof(double) );
  for( i = 0; i < n; i++ )
  for( j = 0; j < n; j++ )
  a[i][j] = (double)M->Elem[i*n+j];
  /**/
  
  for( i = 0; i < n; i++ )
    a[i][i] -= eigVal;
  
  
#ifdef GAUSS 
  // b = (double**) malloc( n * sizeof(double*) );
  b =new double *[n];
  
  for( i = 0; i < n; i++ )
  {
    //b[i] = (double*) malloc( m * sizeof(double) );
    b[i] = new double[m];
  }
  for( i = 0; i < n; i++ )
    for( j = 0; j < m; j++ )
      b[i][j] = x[i];
    
    // debug_a( a, n, n );
    // printf("Vector X:\n"); debug_a( b, n, m );
    gaussj(a,n,b,m);
    /// printf("Solution vector Y:\n"); debug_a( b, n, m );
    
    for( i = 0; i < n; i++ )
      for( j = 0; j < m; j++ )
        y[i] = b[i][j];
      
      for( i = 0; i < n; i++ )
      {
        //FREE(a[i]); FREE(b[i]);
        delete []a[i];
        delete []b[i];
      }
      //FREE(a); FREE(b);
      delete []a;
      delete []b;
#endif
      
#ifdef LU_DECOMP
      //bVec = (double*) malloc( n * sizeof(double) ); CHKPTR(bVec);
      
      bVec =  new double [n];
      
      //indx = (long*) malloc( n * sizeof(long) ); CHKPTR(indx);
      indx = new long [n];
      
      copy_FV( x, bVec, n );
      
      /* debug_a( a, n, n );
      printf(" @@@@@@@@ Input vector X:\n"); debug_vec( bVec, n ); /**/
      ludcmp(a,n,indx,&d);
      lubksb(a,n,indx,bVec);
      /* printf("Solution vector Y:\n"); debug_vec( bVec, n ); /**/
      
      copy_FV( bVec, y, n );
      //FREE( bVec ); FREE( indx );
      delete []bVec;
      delete []indx;
#endif
      
      for(i=0;i<n;i++)
      {
        delete []a[i];
      }
      delete []a;
      
      return RETURN_OK;
}



int MATRIX::Dim()
{
  if(size_x!=size_y)
    return -1;
  else
    return size_x;   
  
}


#define EPSILON_WR (0.0001)   // nejmensi hodnota realny slozky
#define EPSILON_WI (0.0001)   // nejmensi hodnota imaginarni slozky



int MATRIX::Eig(MATRIX *V, MATRIX *D)
{
  if(IsDiag())
  {
    int i,j,iRet;

    int iSize = this->Dim();

    double *sracka;
    
    sracka = new double[iSize];

    iRet = V->Create(iSize,iSize);
    if(iRet !=RETURN_OK)
      return iRet;

    V->SetZero();    
    
    iRet = D->Create(iSize,iSize);
    if(iRet !=RETURN_OK)
      return iRet;

    D->SetZero();    
    
    for( i = 0; i < iSize; i++ )
      sracka[i] = GetXY(i,i);

    qsort((void *)sracka, iSize, sizeof(double), CompareDouble);

    for( i = 0; i < iSize; i++ )
      (*D)[i][i] = sracka[i];

    for(i=0;i<iSize;i++)
    {     
      for( j = 0; j < iSize; j++ )
      {
        if(sracka[i]==GetXY(j,j))
          (*V)[i][j] = 1;
      }
    }
    

  }
  else
  {
    double *wr;
    double *wi;
    int i,j,iRet;
    
    MATRIX mPom;
    mPom.Copy(this);
    
    int iSize = this->Dim();
    
    if(iSize<=0)
      return RETURN_WRONG_MATRIX;
    
    
    wr = new double [iSize];
    wi = new double [iSize];
    
    
    if( iSize == 1 )
    {
      wr[0] =(double) mPom[0][0];
      wi[0] = 0.0;
    }
    else
    { 
      iRet = mPom.Balance(); 
      if(iRet!=RETURN_OK)
        return iRet;
      
      iRet = mPom.Elmhes(); 
      if(iRet!=RETURN_OK)
        return iRet;
      
      iRet = mPom.Hqr( wr, wi );
      if(iRet!=RETURN_OK)
        return iRet;
    }
    
    
    for( i = 0; i < iSize; i++ )
    {
      if( fabs(wi[i]) > EPSILON_WI && fabs(wr[i]) > EPSILON_WR )
      {
        return RETURN_EIG_IMAG;
      }
    }
    
    qsort((void *)wr, iSize, sizeof(double), CompareDouble);
    
    
    iRet = V->Create(iSize,iSize);
    if(iRet !=RETURN_OK)
      return iRet;
    
    
    iRet = D->Create(iSize,iSize);
    if(iRet !=RETURN_OK)
      return iRet;
    
    
    for( i = 0; i < iSize; i++ )
      (*D)[i][i] = wr[i];
    
    
    MATRIX eigVect;
    eigVect.Create(iSize,1);
    
    for(i=0;i<iSize;i++)
    {
      GetEigVect(&(wr[i]),&eigVect,i);    
      for( j = 0; j < iSize; j++ )
      {
        (*V)[j][i] = eigVect[j][0];
      }
    }
    
    
    delete []wr;
    delete []wi; 
  }
  
  return RETURN_OK;
}

int MATRIX::CompareDouble(const void *FirstArgument, const void *SecondArgument)
{
  return (*(double *)FirstArgument) < (*(double *)SecondArgument) ? 1 : (*(double *)FirstArgument) > (*(double *)SecondArgument) ? -1 : 0;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//
//      optimalizace na intel
//
////////////////////////////////////////////////////////////////////////////////////////////////////



int MATRIX::_MultAB_IntGen(MATRIX *A, MATRIX *B)
{
  
#ifdef ___MKL___
  
  int             iRet;
  double           alpha, beta;
  CBLAS_ORDER     order;
  CBLAS_TRANSPOSE transA, transB;
  
  if(A->size_y!=B->size_x)  
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAB() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAB() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_x,A->size_y,B->size_x,B->size_y);
    return RETURN_FAIL;
  }
  
  int X,Y,Z;
  
  X = A->size_x;
  Y = B->size_y;
  Z = A->size_y;
  
  if((size_x!=X)||(size_y!=Y))
  {
    iRet = Create(X,Y);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  
  transA = CblasNoTrans;
  transB = CblasNoTrans;
  order  = CblasRowMajor;
  alpha = 1.0;
  beta  = 0.0;    
  
  
  cblas_dgemm(order, transA, transB, X, Y, Z, alpha, A->da_ta, Z, B->da_ta, Y, beta, da_ta, Y);
  
#else
  int iRet;
  
  iRet = MultAB(B,A);
  return iRet;
  
#endif
  
  return RETURN_OK;
  
}


int MATRIX::_MultAtB_IntGen(MATRIX *A, MATRIX *B)
{
  
#ifdef ___MKL___
  
  int             iRet;
  double           alpha, beta;
  CBLAS_ORDER     order;
  CBLAS_TRANSPOSE transA, transB;
  
  if(A->size_x!=B->size_x)  
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAB() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAB() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y);
    return RETURN_FAIL;
  }
  
  int _size_x,_size_y;
  
  _size_x=A->size_y;
  _size_y=B->size_y;
  
  if((size_x!=_size_x)||(size_y!=_size_y))
  {
    iRet = Create(_size_x,_size_y);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  
  transA = CblasTrans;
  transB = CblasNoTrans;
  order  = CblasRowMajor;
  alpha = 1.0;
  beta  = 0.0;    
  
  
  cblas_dgemm(order, transA, transB, A->size_y, B->size_y, A->size_x, alpha, A->da_ta, A->size_y, B->da_ta, B->size_y, beta, da_ta, B->size_y);
  
#else
  int iRet;
  
  iRet = MultAtB(B,A);
  return iRet;
  
#endif
  
  return RETURN_OK;
  
}



int MATRIX::_MultAtBA_IntGen(MATRIX *A, MATRIX *B)
{
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
#ifdef ___MKL___
  
  int             iRet;
  
  int X,Y;
  
  X=A->size_y;
  Y=B->size_y;
  
  if((size_x!=X)||(size_y!=X))
  {
    iRet = Create(X,X);
    if(iRet!=RETURN_OK)
      return iRet;
    SetZero();
  }  
  
  //double *pom;  pom=new double[X * Y];
  // static double _pom[___MAX_SIZE___];
  //double  _pom[___MAX_SIZE___];
  
  
  
  if(GetCurrentThreadId()==_threadA)
  {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, X, Y, Y, 1.0, A->da_ta, X, B->da_ta, Y, 0.0, _pomA,  Y);
    
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X, X, Y, 1.0, _pomA, Y, A->da_ta, X, 0.0, da_ta,X);
  }
  else
  {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, X, Y, Y, 1.0, A->da_ta, X, B->da_ta, Y, 0.0, _pomB,  Y);
    
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X, X, Y, 1.0, _pomB, Y, A->da_ta, X, 0.0, da_ta,X);
    
    
  }
  
  //delete []pom;
  
  
#else
  int iRet;
  
  iRet = MultAtBA(A,B);
  if(iRet!=RETURN_OK)
    return iRet;
  
#endif
  
  return RETURN_OK;
}


int MATRIX::_MultAtBA_IntSym(MATRIX *A, MATRIX *B)
{
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA_IntSym() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA_IntSym() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
#ifdef ___MKL___
  
  int             iRet;
  double           alpha, beta;
  
  alpha = 1.0;
  beta  = 0.0;
  
  
  int X,Y;
  
  X=A->size_y;
  Y=B->size_y;
  
  if((size_x!=X)||(size_y!=X))
  {
    iRet = Create(X,X);
    if(iRet!=RETURN_OK)
      return iRet;
    SetZero();
  }  
  
  //double *pom;  pom=new double[X * Y];
  
  // static double _pom[___MAX_SIZE___];
  
  //double _pom[___MAX_SIZE___];
  
  /* 
  //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Y, X, Y, alpha, B->da_ta, Y, A->da_ta, X, beta, _pom,  X);
  
    
      cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, Y, X, alpha, B->da_ta, Y,  A->da_ta, X, beta, _pom,  X); 
      
        
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, X, X, Y, alpha, A->da_ta, X, _pom, X, beta, da_ta,X);
  /**/
  
  /*
  
    cblas_dsymm(CblasRowMajor, CblasRight  , CblasLower  , X, Y,   alpha, A->da_ta, Y, B->da_ta, Y, beta, _pom ,  Y); 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X, X, Y, 1.0 , _pom,     Y, A->da_ta, X, 0.0 , da_ta,  X);
    
  /**/
  
  if(GetCurrentThreadId()==_threadA)
  {
    
    cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, X, Y, Y, 1.0, A->da_ta, X, B->da_ta, Y, 0.0, _pomA, Y);  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X, X, Y, 1.0, _pomA,     Y, A->da_ta, X, 0.0, da_ta,X);
  }
  else
  {
    cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, X, Y, Y, 1.0, A->da_ta, X, B->da_ta, Y, 0.0, _pomB, Y);  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, X, X, Y, 1.0, _pomB,     Y, A->da_ta, X, 0.0, da_ta,X);
    
    
  }
  
  /**/ 
  //delete []pom;
  
#else
  
  int iRet;  
  iRet = MultAtBA(A,B);
  if(iRet!=RETURN_OK)
    return iRet;
  
#endif
  
  return RETURN_OK;
}

int MATRIX::_MultAB_IntSym(MATRIX *A, MATRIX *B)
{
  
#ifdef ___MKL___
  
  int             iRet;
  double           alpha, beta;
  
  if(A->size_y!=B->size_x)  
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAB_IntSym() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAB_IntSym() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_x,A->size_y,B->size_x,B->size_y);
    return RETURN_FAIL;
  }
  
  beta  = 0.0;
  alpha = 1.0;
  
  int X,Y,Z;
  
  X = A->size_x;
  Y = B->size_y;
  Z = A->size_y;
  
  if((size_x!=X)||(size_y!=Y))
  {
    iRet = Create(X,Y);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  
  cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, X, Y, alpha, A->da_ta, Z,  B->da_ta, Y, beta, da_ta,  Y);
  
  
#else
  int iRet;
  
  iRet = _MultAB_Gen(A,B);
  if(iRet!=RETURN_OK)
    return iRet;
  
#endif
  
  return RETURN_OK;
}



int MATRIX::_MultAtBA_IntDiag(MATRIX *A, MATRIX *B)
{
  
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA_IntDiag() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA_IntDiag() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  
  int iRet = 0;
  
  
  int size_x_Ab = A->size_x;
  int size_y_Ab = A->size_y;
  
  //  static double _pom[___MAX_SIZE___];
  //double _pom[___MAX_SIZE___];
  
#ifdef ___MKL___

  int i,j;
  int X,Y;
  
  X = A->size_x;
  Y = A->size_y;
  
  
  if((size_x!=Y)||(size_y!=Y))
  {
    iRet = Create(Y,Y);
    if(iRet!=RETURN_OK)
      return iRet;
  }
  
  
  if(GetCurrentThreadId()==_threadA)
  {
    
    memcpy( _pomA , A->da_ta , sizeof(double) * A->size_a);
    
    
    for( i = 0; i < size_x_Ab; i++ )
    {
      for( j = 0; j < size_y_Ab; j++)
      {
        _pomA[i*size_y_Ab +j] *= B->da_ta[i *B->size_y +i];
      }
    }   
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Y, Y, X, 1.0, A->da_ta , Y, _pomA , Y, 0.0, da_ta,Y);
  }
  else
  {
    
    memcpy( _pomB , A->da_ta , sizeof(double) * A->size_a);
    
    
    for( i = 0; i < size_x_Ab; i++ )
    {
      for( j = 0; j < size_y_Ab; j++)
      {
        _pomB[i*size_y_Ab +j] *= B->da_ta[i *B->size_y +i];
      }
    }   
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Y, Y, X, 1.0, A->da_ta , Y, _pomB , Y, 0.0, da_ta,Y);
    
  }
  
  return RETURN_OK;
  
#else
  
  return RETURN_FAIL;
  
#endif
  
  
  
}



int MATRIX::_MultAtBA_IntVect(MATRIX *A, MATRIX *B)
{
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x)||(A->size_y!=1))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA_IntVect() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA_IntVect() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
#ifdef ___MKL___
  
  int             iRet;
  double           alpha, beta;
  CBLAS_ORDER     order;
  CBLAS_TRANSPOSE transA, transB;
  
  
  int X,Y;
  
  X=A->size_y;
  Y=B->size_y;
  
  if((size_x!=X)||(size_y!=X))
  {
    iRet = Create(X,X);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  // double *pom; pom=new double[X * Y];
  // static double _pom[___MAX_SIZE___];
  
  // double _pom[___MAX_SIZE___];
  
  
  transA = CblasTrans;
  transB = CblasNoTrans;
  order  = CblasRowMajor;
  alpha = 1.0;
  beta  = 0.0;    
  
  if(GetCurrentThreadId()==_threadA)
  { 
    cblas_dgemv(CblasRowMajor,CblasNoTrans, B->size_x, B->size_y,alpha, B->da_ta , B->size_x ,A->da_ta,1,beta,_pomA,1);
    
    //  for(int i=0;i<X*Y;i++) printf("%10.5f  ",pom[i]);
    
    da_ta[0] = cblas_ddot(A->size_x, A->da_ta, 1, _pomA,1);  
  }
  else
  {
    cblas_dgemv(CblasRowMajor,CblasNoTrans, B->size_x, B->size_y,alpha, B->da_ta , B->size_x ,A->da_ta,1,beta,_pomB,1);
    
    //  for(int i=0;i<X*Y;i++) printf("%10.5f  ",pom[i]);
    
    da_ta[0] = cblas_ddot(A->size_x, A->da_ta, 1, _pomB,1);  
  }
  
  
  // delete []pom;
  
  
#else
  int iRet;
  iRet = _MultAtB_IntGen(A,B);
  if(iRet!=RETURN_OK)
    return iRet;
  
#endif
  
  return RETURN_OK;
}

int MATRIX::_MultAtBA_IntSymVect(MATRIX *A, MATRIX *B)
{
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x)||(A->size_y!=1))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA_IntSymVect() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA_IntSymVect() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
#ifdef ___MKL___
  
  int             iRet;
  double           alpha, beta;
  
  
  int X,Y;
  
  X=A->size_y;
  Y=B->size_y;
  
  if((size_x!=X)||(size_y!=X))
  {
    iRet = Create(X,X);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  //double *_pom;  _pom=new double[X * Y];
  //  static double _pom[___MAX_SIZE___];
  
  
  
  
  // double _pom[___MAX_SIZE___];  
  
  
  alpha = 1.0;
  beta  = 0.0;    
  
  
  if(GetCurrentThreadId()==_threadA)
  {
    cblas_dgemv(CblasRowMajor,CblasNoTrans, B->size_x, B->size_y,alpha, B->da_ta , B->size_x ,A->da_ta,1,beta,_pomA,1);
    da_ta[0] = cblas_ddot(A->size_x, A->da_ta, 1, _pomA,1);  
  }
  else
  {
    cblas_dgemv(CblasRowMajor,CblasNoTrans, B->size_x, B->size_y,alpha, B->da_ta , B->size_x ,A->da_ta,1,beta,_pomB,1);
    da_ta[0] = cblas_ddot(A->size_x, A->da_ta, 1, _pomB,1);  
  }
  
  // je to f v prdeli nechapu jak muze nasobeni obecnou matici je rychlejsi nez symetrickou ... nechapu
  
  
  //  cblas_dsymv(CblasRowMajor, CblasUpper, B->size_x, alpha, B->da_ta, B->size_x, A->da_ta,1,beta, _pom,1);
  //  da_ta[0] = cblas_ddot(A->size_x, A->da_ta, 1, _pom,1);  
  
  
  //delete []_pom;
  
  
#else
  int iRet;
  
  iRet = MultAtBA(A,B);
  if(iRet!=RETURN_OK)
    return iRet;
  
#endif
  
  return RETURN_OK;
}



int MATRIX::Add_MultAtA(MATRIX *A, double alpha)
{
  int iRet;
  
#ifdef ___MKL___  
  iRet = _Add_MultAtA_IntGen(A,alpha);
#else
  iRet = _Add_MultAtA_Gen(A,alpha);
#endif      
  
  return iRet;
}

int MATRIX::_Add_MultAtA_IntGen(MATRIX *A, double alpha)
{
#ifdef ___MKL___
  
  int iRet;
  int X,Y;
  
  X = A->size_x;
  Y = A->size_y;
  
  if(size_a==0)
  {
    iRet = Create(Y,Y);
    if(iRet!=RETURN_OK)
      return iRet;
  }
  
  if((size_x!=size_y)||(Y!=size_x))  
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_Add_MultAtA_IntGen() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_Add_MultAtA_IntGen() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_x,A->size_y,A->size_y,A->size_x);
    return RETURN_FAIL;
  }
  
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Y, Y, X, alpha, A->da_ta, X, A->da_ta, Y, 1.0, da_ta, Y);
  
#else
  int iRet;  
  iRet = _Add_MultAtA_Gen(A,alpha);
  return iRet;
  
#endif
  
  return RETURN_OK;
  
}

int MATRIX::_Add_MultAtA_Gen(MATRIX *A, double alpha)
{
  
  int iRet;
  int X,Y;
  
  X = A->size_x;
  Y = A->size_y;
  
  if(size_a==0)
  {
    iRet = Create(Y,Y);
    if(iRet!=RETURN_OK)
      return iRet;
  }
  
  if((size_x!=size_y)||(Y!=size_x))  
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_Add_MultAtA_Gen() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_Add_MultAtA_Gen() : Incompatible matrix sizes {%d,%d}{%d,%d}\n",A->size_x,A->size_y,A->size_y,A->size_x);
    return RETURN_FAIL;
  }
  
  int i,j;
  
  for(i=0;i<Y;i++)
  {
    for(j=0;j<Y;j++)
    {
      da_ta[i * size_y + j] +=  alpha * A->da_ta[i] * A->da_ta[j];
    }
  }
  
  
  return RETURN_OK;
}


int MATRIX::_GetRowR(int kerej, MATRIX *x)    // tuto pofrci odkazem to sem na to zvedavej estli se to nigde neprepise
{
  if(kerej>=size_x)
    return RETURN_FAIL;  
  
  if(x->da_ta==NULL)
  {   
    x->size_x = 1;
    x->size_y = size_y;
    x->size_a = size_y;
  }
  else
  {
    if((x->size_x!=1)||(x->size_y!=size_y))
    {
      return RETURN_FAIL;
    }
  }  
  
  x->da_ta = &da_ta[kerej * size_y];
  
  
  
  return RETURN_OK;
  
  
}

double MATRIX::GetXY(int x, int y)
{
  return da_ta[x * size_y + y]; // LMa 29.5.2009
}

bool MATRIX::IsDiag()
{
  int x,y;
  int iSize = min(size_x,size_y);
  
  for(x=0;x<iSize;x++)
  {
    for(y=0;y<iSize;y++)
      if(x!=y)
        if(da_ta[x * size_y + y]!=0.0)
          return false;
        
  } 
  
  return true;
}

int MATRIX::_MultAtBA_GenDiaVect(MATRIX *A, MATRIX *B)
{
  
  if((B->size_y!=B->size_x)||(A->size_x!=B->size_x)||(A->size_y!=1))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA_GenDiaVect() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA_GenDiaVect() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_x,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  int X,Y;
  int iRet;
  
  X=A->size_y;
  Y=B->size_y;
  
  if((size_x!=X)||(size_y!=X))
  {
    iRet = Create(X,X);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  
  double fPom = 0;
  
  for (int i=0;i<Y;i++)	
    fPom += A->da_ta[i]* A->da_ta[i] * B->da_ta[i*Y+i];
  
  
  da_ta[0] = fPom;
  
  
  
  return RETURN_OK;
  
}


void MATRIX::SetXY(int i, int j, double co)
{
  da_ta[ i * size_y + j ] = co;
}

int MATRIX::_MultAtBA_GenVectVect(MATRIX *A, MATRIX *B)
{
  
  if((B->size_x!=1)||(A->size_x!=B->size_y)||(A->size_y!=1))
  {
#ifdef __VYHOD_CHYBY__  
    REPORT_ERROR( "MATRIX::_MultAtBA_GenVectVect() : Incompatible matrix sizes!");
#endif
    printf("MATRIX::_MultAtBA_GenVectVect() : Incompatible matrix sizes {%d,%d}{%d,%d}{%d,%d}\n",A->size_y,A->size_x,B->size_y,B->size_y,A->size_x,A->size_y);
    return RETURN_FAIL;
  }
  
  int X,Y;
  int iRet;
  
  X=A->size_y;
  Y=B->size_y;
  
  if((size_x!=X)||(size_y!=X))
  {
    iRet = Create(X,X);
    if(iRet!=RETURN_OK)
      return iRet;
  }  
  
  
  double fPom = 0;
  
  for (int i=0;i<Y;i++)	
    fPom += A->da_ta[i]* A->da_ta[i] * B->da_ta[i];
  
  
  da_ta[0] = fPom;
  
  
  
  return RETURN_OK;
  
}

//vypocte Floatpole * MATRIX
//MATRIX musi byt ctvercova, vysledek se ulozi zpet do Floatpole
void MATRIX::_MultFloatArray(float **p, int N, int M) {
	if(size_x!=size_y) {
		printf("MATRIX::_MultFloatArray error: MATRIX must be square.\n");
		return;
	}
	if(p==NULL) {
		printf("MATRIX::_MultFloatArray error: FloatArray is empty.\n");
		return;
	}
	int i, j;
	MATRIX buff, res, tmp;
	buff.Create(M, 1);
	res.Create(M, 1);
	tmp.Create(this);
	for(i=0;i<N;i++){
		for(j=0;j<M;j++){
			buff.da_ta[j] = p[i][j];
		}
		res.MultAB(&tmp, &buff);
		for(j=0;j<M;j++){
			p[i][j] = (float) res.da_ta[j];
		}
	}
}


//vypocte Floatpole * MATRIX
//MATRIX musi byt ctvercova, vysledek se ulozi zpet do Floatpole
//pred nasobenim odecte vektor vM a po vynasobeni pricte vektor vTM
void MATRIX::_MultFloatArrayT(float **p, int N, int M, float *vM, float *vTM) {
	if(size_x!=size_y) {
		printf("MATRIX::_MultFloatArrayT error: MATRIX must be square.\n");
		return;
	}
	if(p==NULL) {
		printf("MATRIX::_MultFloatArrayT error: FloatArray is empty.\n");
		return;
	}
	int i, j;
	MATRIX buff, res, tmp;
	buff.Create(M, 1);
	res.Create(M, 1);
	tmp.Create(this);
	for(i=0;i<N;i++){
		for(j=0;j<M;j++){
			buff.da_ta[j] = p[i][j] - vM[j];
		}
		res.MultAB(&tmp, &buff);
		for(j=0;j<M;j++){
			p[i][j] = (float) res.da_ta[j] + vTM[j];
		}
	}
}
