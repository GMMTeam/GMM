#ifndef __H_EXCEPTION
#define __H_EXCEPTION

//#define EXCEPTION_PRINTING
#define INTERFACEDLLEXPORT

#include <time.h>
#include <sstream>

/** Exception types */
enum eExceptionType{ModelAdaptException,MainException,ModelConstruction,MultiParam,DataPackages,ModulSpeakerVerify,ModelSVM,StatsHandling};

/*======================================================================*/
//
//  class Exception
//
/*======================================================================*/

/**
 * This class implements methods for exception.
*/
class Exception
{

  private:

  /** Exception file name */
  const char *m_pchFileName;
  /** Exception line number */
  int m_lFileLine;

  public:

  /** Exception type */
  eExceptionType m_eType;
  /** Exception description */
  std::string m_strDescription;

  /**
   * Constructor
   * @param pchFileName File name of exception
   * @param lFileLine File line of exception
   * @param eType Type of exception
   * @param strDescription Description of exception
   * @param bPrint Exception print indicator
  */
  INTERFACEDLLEXPORT Exception(const char *pchFileName, int lFileLine, eExceptionType eType, std::string strDescription, bool bPrint);

  /**
   * Destructor
  */
  INTERFACEDLLEXPORT ~Exception(void);

  /**
   * Prints exception
  */
  INTERFACEDLLEXPORT void Print(void);
};

/** Exception mechanism macros */
#ifdef EXCEPTION_SUPPORT

#define EXCEPTION_THROW(TYPE, DESCRIPTION, PRINT) {std::ostringstream __abcxyz__; __abcxyz__ << DESCRIPTION; throw new Exception(__FILE__, __LINE__, (TYPE), __abcxyz__.str(), (PRINT));}
#define EXCEPTION_TRY try
#define EXCEPTION_CATCH(EXCEPTION) catch(EXCEPTION)

#else //EXCEPTION_SUPPORT

#define EXCEPTION_THROW(TYPE, DESCRIPTION, PRINT) {std::ostringstream __abcxyz__; __abcxyz__ << DESCRIPTION; delete new Exception(__FILE__, __LINE__, (TYPE), __abcxyz__.str(), (PRINT)); abort();}
#define EXCEPTION_TRY
#define EXCEPTION_CATCH(EXCEPTION) EXCEPTION; if (false)

#endif //EXCEPTION_SUPPORT

#endif //__H_EXCEPTION
