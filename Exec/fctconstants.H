#ifndef FCTCONSTANTS_H_
#define FCTCONSTANTS_H_
#include <AMReX_Config.H>

#include <AMReX_REAL.H>

#define ro 	0
#define rou 1 
#define rov	2
#if AMREX_SPACEDIM==2
#define roE 3
#define pre 4
#define mac 5
#define conscomp 4
#endif
#if AMREX_SPACEDIM==3
#define row 3
#define roE 4
#define pre 5
#define mac 6
#define ccomp 5
#endif
#define gamma   1.4
#define c1		1.0/(gamma-1.0)
// #define five    5.0
// #define six     6.0
// #define seven   BL_REAL(7.0)
// #define eight   BL_REAL(8.0)
// #define nine    BL_REAL(9.0)
// #define ten     BL_REAL(10.0)
// #define twelve  BL_REAL(12.0)
// #define fifteen BL_REAL(15.0)
// #define sixteen BL_REAL(16.0)
// #define twenty  BL_REAL(20.0)
// #define seventy BL_REAL(70.0)
// #define ninety  BL_REAL(90.0)
// #define tenth   BL_REAL(0.1)
// #define eighth  BL_REAL(0.125)
#define sixth   0.16666666666666667
// #define fifth   BL_REAL(0.2)
// #define fourth  BL_REAL(0.25)
#define third   0.33333333333333333
#define half    0.5
#define one12	0.08333333333333333
#define epsilon		1E-16
// #define two3rd  BL_REAL(0.66666666666666667)
// #define Pi      BL_REAL(3.1415926535897932)

#endif /*BL_CONSTANTS_H*/
