#include "general/Exception.h"
#include <stdio.h>

/** Exception type strings */
static const char *pchExceptionType[] = {"Class-CModelAdapt","Model-Main","ModelConstruction","Multi class parametrization",
										 "Packages handling","Speaker Verification","Model-SVM","StatsHandling"};

/*======================================================================*/
//
//  Exception(const char *pchFileName, int lFileLine, eExceptionType eType, std::string strDescription, bool bPrint)
//
//  Constructor
//
/*======================================================================*/

Exception::Exception(const char *pchFileName, int lFileLine, eExceptionType eType, std::string strDescription, bool bPrint) :
m_pchFileName(pchFileName), m_lFileLine(lFileLine), m_eType(eType), m_strDescription(strDescription)
{

#ifdef EXCEPTION_PRINTING

  // print exception
  if (bPrint == true)
    Print();

#else //EXCEPTION_PRINTING

  (void)bPrint;

#endif //EXCEPTION_PRINTING

}

/*======================================================================*/
//
//  ~Exception(void)
//
//  Destructor
//
/*======================================================================*/

Exception::~Exception(void)
{
}

/*======================================================================*/
//
//  void Print(void)
//
//  Prints exception
//
/*======================================================================*/

void Exception::Print(void)
{
  time_t Time;
  tm *pTime;

  // print exception
  Time = time(&Time);
  pTime = localtime(&Time);
  //printf("\nException raised %d.%d.%d %d:%d:%d\n  Exception location:    %s, line %ld\n  Exception type:        %s\n  Exception description: %s\n", pTime->tm_mday, pTime->tm_mon + 1, pTime->tm_year + 1900, pTime->tm_hour, pTime->tm_min, pTime->tm_sec, m_pchFileName, m_lFileLine, pchExceptionType[m_eType], m_strDescription.c_str());
  printf(" ERROR: %s\n", m_strDescription.c_str());
}
