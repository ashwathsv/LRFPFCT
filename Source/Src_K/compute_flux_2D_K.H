#ifndef _compute_flux_2d_H_
#define _compute_flux_2d_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_conv_flux_x(int i, int j, int k,
            amrex::Array4<amrex::Real> const& flcx,
            amrex::Array4<amrex::Real> const& fab,
            amrex::Array4<amrex::Real> const& prearr,
            amrex::Array4<amrex::Real> const& vx,
            int fx_lox, int fx_hix)
{
  if(i > fx_lox && i < fx_hix){
    flcx(i,j,k,ro)  = half*( fab(i-1,j,k,rou) + fab(i,j,k,rou) );
    flcx(i,j,k,rou) = half*( (pow(fab(i-1,j,k,rou),2)/fab(i-1,j,k,ro)) + (pow(fab(i,j,k,rou),2)/fab(i,j,k,ro)) )
                    + half*( prearr(i-1,j,k) + prearr(i,j,k) );
    flcx(i,j,k,rov) = half*( (fab(i-1,j,k,rou)*fab(i-1,j,k,rov)/fab(i-1,j,k,ro)) + (fab(i,j,k,rou)*fab(i,j,k,rov)/fab(i,j,k,ro)) );
    flcx(i,j,k,roE) = half*( ( (fab(i-1,j,k,roE)*fab(i-1,j,k,rou))/fab(i-1,j,k,ro) ) 
                    +        ( (fab(i,j,k,roE)*fab(i,j,k,rou))/fab(i,j,k,ro) )   )   
                    + half*vx(i,j,k)*( prearr(i-1,j,k) + prearr(i,j,k) );
                    // + half*( (prearr(i-1,j,k)*fab(i-1,j,k,rou)/fab(i-1,j,k,ro)) + (prearr(i,j,k)*fab(i,j,k,rou)/fab(i,j,k,ro)) );
  }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_conv_flux_y(int i, int j, int k,
            amrex::Array4<amrex::Real> const& flcy,
            amrex::Array4<amrex::Real const> const& fab,
            amrex::Array4<amrex::Real> const& prearr,
            amrex::Array4<amrex::Real> const& vy,
            int fy_loy, int fy_hiy)
{
  if(j > fy_loy && j < fy_hiy){
    flcy(i,j,k,ro)  = half*( fab(i,j-1,k,rov) + fab(i,j,k,rov) );
    flcy(i,j,k,rou) = half*( (fab(i,j-1,k,rou)*fab(i,j-1,k,rov)/fab(i,j-1,k,ro)) + (fab(i,j,k,rou)*fab(i,j,k,rov)/fab(i,j,k,ro)) );
    flcy(i,j,k,rov) = half*( (pow(fab(i,j-1,k,rov),2)/fab(i,j-1,k,ro)) + (pow(fab(i,j,k,rov),2)/fab(i,j,k,ro)) )
                    + half*( prearr(i,j-1,k) + prearr(i,j,k) );
    flcy(i,j,k,roE) = half*( ( (fab(i,j-1,k,roE)*fab(i,j-1,k,rov))/fab(i,j-1,k,ro) ) 
                    +        ( (fab(i,j,k,roE)*fab(i,j,k,rov))/fab(i,j,k,ro) )   )   
                    + half*vy(i,j,k)*( prearr(i,j-1,k) + prearr(i,j,k) );  
                    // + half*( (prearr(i,j-1,k)*fab(i,j-1,k,rov)/fab(i,j-1,k,ro)) + (prearr(i,j,k)*fab(i,j,k,rov)/fab(i,j,k,ro)) ); 
  }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_x_bc(int i, int j, int k, int n,
               amrex::Array4<amrex::Real> const& fx, int fx_lox, int fx_hix)
{
  if(i == fx_lox){      fx(i,j,k,n) = fx(i+1,j,k,n);  }
  else if(i == fx_hix){ fx(i,j,k,n) = fx(i-1,j,k,n);  }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_y_bc(int i, int j, int k, int n,
               amrex::Array4<amrex::Real> const& fy, int fy_loy, int fy_hiy)
{
  if(j == fy_loy){      fy(i,j,k,n) = fy(i,j+1,k,n);  } 
  else if(j == fy_hiy){ fy(i,j,k,n) = fy(i,j-1,k,n);  }   
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_diff_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fldx,
             amrex::Array4<amrex::Real> const& vx, 
             amrex::Array4<amrex::Real> const& fab, 
             int fx_lox, int fx_hix, amrex::Real dtdx)
{
    // if(i > fx_lox && i < fx_hix){
    //   fldx(i,j,k,n) = (sixth + third*pow(dtdx*vx(i,j,k),2.0))*(fab(i-1,j,k,n) - fab(i,j,k,n))/dtdx; }

    if(i > fx_lox && i < fx_hix){
      fldx(i,j,k,n) = (one12 + third*pow(dtdx*vx(i,j,k),2.0))*(fab(i-1,j,k,n) - fab(i,j,k,n))/dtdx; }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_diff_flux_y(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fldy,
             amrex::Array4<amrex::Real> const& vy, 
             amrex::Array4<amrex::Real> const& fab, 
             int fy_loy, int fy_hiy, amrex::Real dtdy)
{
    // if(j > fy_loy && j < fy_hiy){
    //   fldy(i,j,k,n) = (sixth + third*pow(dtdy*vy(i,j,k),2.0))*(fab(i,j-1,k,n) - fab(i,j,k,n))/dtdy;    }

    if(j > fy_loy && j < fy_hiy){
      fldy(i,j,k,n) = (one12 + third*pow(dtdy*vy(i,j,k),2.0))*(fab(i,j-1,k,n) - fab(i,j,k,n))/dtdy;    }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_ad_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flax,
             amrex::Array4<amrex::Real> const& vx, 
             // amrex::Array4<amrex::Real> const& vy, 
             amrex::Array4<amrex::Real> const& fab,
             amrex::Array4<amrex::Real> const& fabcx,
             int fx_lox, int fx_hix, 
             // int fx_loy, int fx_hiy, 
             amrex::Real dtdx, 
             // amrex::Real dtdy,
             amrex::Real diff1)
{
    using namespace amrex;
    
    if((i > fx_lox && i < fx_hix)){
      flax(i,j,k,n) = diff1*((sixth - sixth*pow(dtdx*vx(i,j,k),2.0))*(fabcx(i,j,k,n) - fabcx(i-1,j,k,n))
                    - one12*(fab(i,j,k,n) - fab(i-1,j,k,n)));      }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_ad_flux_y(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flay,
             // amrex::Array4<amrex::Real> const& vx,
             amrex::Array4<amrex::Real> const& vy, 
             amrex::Array4<amrex::Real> const& fab,
             amrex::Array4<amrex::Real> const& fabcy, 
             // int fy_lox, int fy_hix, 
             int fy_loy, int fy_hiy, 
             // amrex::Real dtdx, 
             amrex::Real dtdy, 
             amrex::Real diff1)
{
    using namespace amrex;

    if(j > fy_loy && j < fy_hiy){
      flay(i,j,k,n) = diff1*((sixth - sixth*pow(dtdy*vy(i,j,k),2.0))*(fabcy(i,j,k,n) - fabcy(i,j-1,k,n))
                    - one12*(fab(i,j,k,n) - fab(i,j-1,k,n)));      }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void prelimit_ad_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flax,
             amrex::Array4<amrex::Real> const& fabcx,
             int ilo, int jlo, int ihi, int jhi)
{
    if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi)){
      amrex::Real fltmp = amrex::Math::abs(flax(i,j,k,n));
      amrex::Real sgn   = amrex::Math::copysign(1.0,fabcx(i,j,k,n)-fabcx(i-1,j,k,n));
      flax(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp,sgn*(fabcx(i-1,j,k,n)-fabcx(i-2,j,k,n)),
                      sgn*(fabcx(i+1,j,k,n)-fabcx(i,j,k,n)))); }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void prelimit_ad_flux_y(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flay,
             amrex::Array4<amrex::Real> const& fabcy,
             int ilo, int jlo, int ihi, int jhi)
{
    if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi)){
      amrex::Real fltmp = amrex::Math::abs(flay(i,j,k,n));
      amrex::Real sgn   = amrex::Math::copysign(1.0,fabcy(i,j,k,n)-fabcy(i,j-1,k,n));
      flay(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp,sgn*(fabcy(i,j-1,k,n)-fabcy(i,j-2,k,n)),
                      sgn*(fabcy(i,j+1,k,n)-fabcy(i,j,k,n)))); }
}
//----------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void corrected_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fltx, amrex::Array4<amrex::Real> const& frin,
             amrex::Array4<amrex::Real> const& frout, 
             int ilo, int jlo, int ihi, int jhi)
{
  if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi)){
    amrex::Real temp = fltx(i,j,k,n);
    if(temp >= 0.0)
      fltx(i,j,k,n) = temp*amrex::min(frout(i-1,j,k,n),frin(i,j,k,n),1.0);
    else
      fltx(i,j,k,n) = temp*amrex::min(frin(i-1,j,k,n),frout(i,j,k,n),1.0);
  }
}
//----------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void corrected_flux_y(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flty, amrex::Array4<amrex::Real> const& frin,
             amrex::Array4<amrex::Real> const& frout, 
             int ilo, int jlo, int ihi, int jhi)
{
  using namespace amrex;
  if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi)){
    Real temp = flty(i,j,k,n);
    if(temp >= 0.0)
      flty(i,j,k,n) = temp*amrex::min(frout(i,j-1,k,n),frin(i,j,k,n),1.0);
    else
      flty(i,j,k,n) = temp*min(frin(i,j-1,k,n),frout(i,j,k,n),1.0);
  }
}


#endif
