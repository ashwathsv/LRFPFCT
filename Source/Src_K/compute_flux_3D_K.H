#ifndef _compute_flux_3d_H_
#define _compute_flux_3d_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

using namespace amrex;

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
    flcx(i,j,k,rou) = half*( (pow(fab(i-1,j,k,rou),2.0)/fab(i-1,j,k,ro)) + (pow(fab(i,j,k,rou),2.0)/fab(i,j,k,ro)) )
                    + half*( prearr(i-1,j,k) + prearr(i,j,k) );
                    // + half*( fab(i-1,j,k,pre) + fab(i,j,k,pre) );
    flcx(i,j,k,rov) = half*( (fab(i-1,j,k,rou)*fab(i-1,j,k,rov)/fab(i-1,j,k,ro)) + (fab(i,j,k,rou)*fab(i,j,k,rov)/fab(i,j,k,ro)) );
    flcx(i,j,k,row) = half*( (fab(i-1,j,k,rou)*fab(i-1,j,k,row)/fab(i-1,j,k,ro)) + (fab(i,j,k,rou)*fab(i,j,k,row)/fab(i,j,k,ro)) );
    // flcx(i,j,k,ro)  = half*( fab(i-1,j,k,rou) + fab(i,j,k,rou) );
    // flcx(i,j,k,rou) = half*( (fab(i-1,j,k,rou)*fab(i-1,j,k,rov)/fab(i-1,j,k,ro)) 
    //                 + (fab(i,j,k,rou)*fab(i,j,k,rov)/fab(i,j,k,ro)) )
    //                 + half*( fab(i-1,j,k,pre) + fab(i,j,k,pre) );
    // flcx(i,j,k,rov) = half*( (fab(i-1,j,k,rou)*fab(i-1,j,k,rov)/fab(i-1,j,k,ro)) + (fab(i,j,k,rou)*fab(i,j,k,rov)/fab(i,j,k,ro)) );
    // flcx(i,j,k,row) = half*( (fab(i-1,j,k,rou)*fab(i-1,j,k,row)/fab(i-1,j,k,ro)) + (fab(i,j,k,rou)*fab(i,j,k,row)/fab(i,j,k,ro)) );
    flcx(i,j,k,roE) = half*  ( (fab(i-1,j,k,rou)*fab(i-1,j,k,roE)/fab(i-1,j,k,ro)) 
                    +          (fab(i,j,k,roE)*fab(i,j,k,rou)/fab(i,j,k,ro)) )   
                    + half*vx(i,j,k)*( prearr(i-1,j,k) + prearr(i,j,k) );
                    // + half*( (fab(i-1,j,k,pre)*fab(i-1,j,k,rou)/fab(i-1,j,k,ro)) + (fab(i,j,k,pre)*fab(i,j,k,rou)/fab(i,j,k,ro)) );
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
    flcy(i,j,k,rov) = half*( (pow(fab(i,j-1,k,rov),2.0)/fab(i,j-1,k,ro)) + (pow(fab(i,j,k,rov),2.0)/fab(i,j,k,ro)) )
                    + half*( prearr(i,j-1,k) + prearr(i,j,k) );
                    // + half*( fab(i,j-1,k,pre) + fab(i,j,k,pre) );
    flcy(i,j,k,row) = half*( (fab(i,j-1,k,row)*fab(i,j-1,k,rov)/fab(i,j-1,k,ro)) + (fab(i,j,k,row)*fab(i,j,k,rov)/fab(i,j,k,ro)) );
    flcy(i,j,k,roE) = half*( ( (fab(i,j-1,k,roE)*fab(i,j-1,k,rov))/fab(i,j-1,k,ro) ) 
                    +        ( (fab(i,j,k,roE)*fab(i,j,k,rov))/fab(i,j,k,ro) )   )   
                    + half*vy(i,j,k)*( prearr(i,j-1,k) + prearr(i,j,k) );
                    // + half*( (fab(i,j-1,k,pre)*fab(i,j-1,k,rov)/fab(i,j-1,k,ro)) + (fab(i,j,k,pre)*fab(i,j,k,rov)/fab(i,j,k,ro)) );   
  }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_conv_flux_z(int i, int j, int k,
            amrex::Array4<amrex::Real> const& flcz,
            amrex::Array4<amrex::Real const> const& fab,
            amrex::Array4<amrex::Real> const& prearr,
            amrex::Array4<amrex::Real> const& vz,
            int fz_loz, int fz_hiz)
{
  if(k > fz_loz && k < fz_hiz){
    flcz(i,j,k,ro)  = half*( fab(i,j,k-1,row) + fab(i,j,k,row) );
    flcz(i,j,k,rou) = half*( (fab(i,j,k-1,rou)*fab(i,j,k-1,row)/fab(i,j,k-1,ro)) + (fab(i,j,k,rou)*fab(i,j,k,row)/fab(i,j,k,ro)) );
    flcz(i,j,k,rov) = half*( (fab(i,j,k-1,rov)*fab(i,j,k-1,row)/fab(i,j,k-1,ro)) + (fab(i,j,k,rov)*fab(i,j,k,row)/fab(i,j,k,ro)) );
    flcz(i,j,k,row) = half*( (pow(fab(i,j,k-1,row),2.0)/fab(i,j,k-1,ro)) + (pow(fab(i,j,k,row),2.0)/fab(i,j,k,ro)) )
                    + half*( prearr(i,j,k-1) + prearr(i,j,k) );
                    // + half*( fab(i,j,k-1,pre) + fab(i,j,k,pre) );
    flcz(i,j,k,roE) = half*( ( (fab(i,j,k-1,roE)*fab(i,j,k-1,row))/fab(i,j,k-1,ro) ) 
                    +        ( (fab(i,j,k,roE)*fab(i,j,k,row))/fab(i,j,k,ro) )   )   
                    + half*vz(i,j,k)*( prearr(i,j,k-1) + prearr(i,j,k) );
                    // + half*( (fab(i,j,k-1,pre)*fab(i,j,k-1,row)/fab(i,j,k-1,ro)) + (fab(i,j,k,pre)*fab(i,j,k,row)/fab(i,j,k,ro)) );       
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
void flux_z_bc(int i, int j, int k, int n,
               amrex::Array4<amrex::Real> const& fz, int fz_loz, int fz_hiz)
{
  if(k == fz_loz){      fz(i,j,k,n) = fz(i,j,k+1,n);  } 
  else if(k == fz_hiz){ fz(i,j,k,n) = fz(i,j,k-1,n);  }   
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
    if(j > fy_loy && j < fy_hiy){
      fldy(i,j,k,n) = (one12 + third*pow(dtdy*vy(i,j,k),2.0))*(fab(i,j-1,k,n) - fab(i,j,k,n))/dtdy;    }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_diff_flux_z(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fldz,
             amrex::Array4<amrex::Real> const& vz, 
             amrex::Array4<amrex::Real> const& fab, 
             int fz_loz, int fz_hiz, amrex::Real dtdz)
{
    if(k > fz_loz && k < fz_hiz){
      fldz(i,j,k,n) = (one12 + third*pow(dtdz*vz(i,j,k),2.0))*(fab(i,j,k-1,n) - fab(i,j,k,n))/dtdz;    }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_ad_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flax,
             amrex::Array4<amrex::Real> const& vx, 
             amrex::Array4<amrex::Real> const& fab,
             amrex::Array4<amrex::Real> const& fabcx,
             int fx_lox, int fx_hix, 
             amrex::Real dtdx, 
             amrex::Real diff1)
{
    using namespace amrex;

    if((i > fx_lox && i < fx_hix)){
      flax(i,j,k,n) = diff1*((sixth - sixth*pow(dtdx*vx(i,j,k),2.0))*(fabcx(i,j,k,n) - fabcx(i-1,j,k,n))
                    - one12*(fab(i,j,k,n) - fab(i-1,j,k,n)));      
    }
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
                    - one12*(fab(i,j,k,n) - fab(i,j-1,k,n)));      
    }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_ad_flux_z(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flaz,
             // amrex::Array4<amrex::Real> const& vx,
             amrex::Array4<amrex::Real> const& vz, 
             amrex::Array4<amrex::Real> const& fab,
             amrex::Array4<amrex::Real> const& fabcz, 
             // int fy_lox, int fy_hix, 
             int fz_loz, int fz_hiz, 
             // amrex::Real dtdx, 
             amrex::Real dtdz, 
             amrex::Real diff1)
{
    using namespace amrex;

    if(k > fz_loz && k < fz_hiz){
      flaz(i,j,k,n) = diff1*((sixth - sixth*pow(dtdz*vz(i,j,k),2.0))*(fabcz(i,j,k,n) - fabcz(i,j,k-1,n))
                    - one12*(fab(i,j,k,n) - fab(i,j,k-1,n)));      
    }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void prelimit_ad_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flax,
             amrex::Array4<amrex::Real> const& fabcx,
             int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
    if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi) && (k>=klo && k<=khi)){
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
             int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
    if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi) && (k>=klo && k<=khi)){
      amrex::Real fltmp = amrex::Math::abs(flay(i,j,k,n));
      amrex::Real sgn   = amrex::Math::copysign(1.0,fabcy(i,j,k,n)-fabcy(i,j-1,k,n));
      flay(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp,sgn*(fabcy(i,j-1,k,n)-fabcy(i,j-2,k,n)),
                      sgn*(fabcy(i,j+1,k,n)-fabcy(i,j,k,n)))); }
}
//---------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void prelimit_ad_flux_z(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flaz,
             amrex::Array4<amrex::Real> const& fabcz,
             int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
    if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi) && (k>=klo && k<=khi)){
      amrex::Real fltmp = amrex::Math::abs(flaz(i,j,k,n));
      amrex::Real sgn   = amrex::Math::copysign(1.0,fabcz(i,j,k,n)-fabcz(i,j,k-1,n));
      flaz(i,j,k,n) = sgn*amrex::max(0.0,amrex::min(fltmp,sgn*(fabcz(i,j,k-1,n)-fabcz(i,j,k-2,n)),
                      sgn*(fabcz(i,j,k+1,n)-fabcz(i,j,k,n)))); }
}
//----------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void corrected_flux_x(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fltx, amrex::Array4<amrex::Real> const& frin,
             amrex::Array4<amrex::Real> const& frout, 
             int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
  if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi) && (k>=klo && k<=khi)){
    amrex::Real temp = fltx(i,j,k,n);
    if(temp >= 0.0)
      fltx(i,j,k,n) = temp*amrex::min(frout(i-1,j,k,n),frin(i,j,k,n),1.000000000);
    else
      fltx(i,j,k,n) = temp*amrex::min(frin(i-1,j,k,n),frout(i,j,k,n),1.000000000);
  }
}
//----------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void corrected_flux_y(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flty, amrex::Array4<amrex::Real> const& frin,
             amrex::Array4<amrex::Real> const& frout, 
             int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
  using namespace amrex;
  if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi) && (k>=klo && k<=khi) ){
    Real temp = flty(i,j,k,n);
    if(temp >= 0.0)
      flty(i,j,k,n) = temp*amrex::min(frout(i,j-1,k,n),frin(i,j,k,n),1.00000000);
    else
      flty(i,j,k,n) = temp*min(frin(i,j-1,k,n),frout(i,j,k,n),1.00000000);
  }
}
//----------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void corrected_flux_z(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fltz, amrex::Array4<amrex::Real> const& frin,
             amrex::Array4<amrex::Real> const& frout, 
             int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
  using namespace amrex;
  if((i>=ilo && i<=ihi) && (j>=jlo && j<=jhi) && (k>=klo && k<=khi) ){
    Real temp = fltz(i,j,k,n);
    if(temp >= 0.0)
      fltz(i,j,k,n) = temp*amrex::min(frout(i,j,k-1,n),frin(i,j,k,n),1.00000000);
    else
      fltz(i,j,k,n) = temp*min(frin(i,j,k-1,n),frout(i,j,k,n),1.00000000);
  }
}

#endif
