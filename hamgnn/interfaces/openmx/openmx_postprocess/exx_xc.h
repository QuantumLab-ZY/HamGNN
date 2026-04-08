/*----------------------------------------------------------------------
  exx_xc.h

  semi-local XC potenitals for EXX and hybrid functional calculations 

  MT, 14/JAN/2010
----------------------------------------------------------------------*/
#ifndef EXX_XC_H_INCLUDED
#define EXX_XC_H_INCLUDED

/* C part of Ceparly-Alder XC */
double EXX_XC_CA_withoutX(double den, int P_switch);

/* C part of CA-LSDA */
void EXX_XC_CA_LSDA(double den0, double den1, double XC[2],int P_switch);

#endif /* EXX_XC_H_INCLUDED */
