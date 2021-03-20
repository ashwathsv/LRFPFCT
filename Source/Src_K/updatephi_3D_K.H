#ifndef updatephi_3d_k_H_
#define updatephi_3d_k_H_

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void conv_updatephi(int i, int j, int k, int n,
            amrex::Array4<amrex::Real> const& fabnew, amrex::Array4<amrex::Real> const& fabold,
            amrex::Array4<amrex::Real> const& flcx, amrex::Array4<amrex::Real> const& flcy, amrex::Array4<amrex::Real> const& flcz,
            amrex::Array4<amrex::Real> const& fabcx, amrex::Array4<amrex::Real> const& fabcy, amrex::Array4<amrex::Real> const& fabcz, 
            amrex::Real dtdx, amrex::Real dtdy, amrex::Real dtdz)
{
    fabcx(i,j,k,n) = fabold(i,j,k,n) - dtdx*(flcx(i+1,j,k,n) - flcx(i,j,k,n));
    fabcy(i,j,k,n) = fabold(i,j,k,n) - dtdy*(flcy(i,j+1,k,n) - flcy(i,j,k,n));
    fabcz(i,j,k,n) = fabold(i,j,k,n) - dtdz*(flcz(i,j,k+1,n) - flcz(i,j,k,n));
    fabnew(i,j,k,n) = fabold(i,j,k,n) 
                    - dtdx*(flcx(i+1,j,k,n) - flcx(i,j,k,n))
                    - dtdy*(flcy(i,j+1,k,n) - flcy(i,j,k,n)) 
                    - dtdz*(flcz(i,j,k+1,n) - flcz(i,j,k,n));
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void phiconv_bc(int i, int j, int k, int n, 
             amrex::Array4<amrex::Real> const& fabnew,
             amrex::Array4<amrex::Real> const& fabcx, 
             amrex::Array4<amrex::Real> const& fabcy, amrex::Array4<amrex::Real> const& fabcz,
             int philox, int philoy, int philoz,
             int phihix, int phihiy, int phihiz )
{
    if(i==philox && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i+1,j,k,n);    
        fabcy(i,j,k,n) = fabcy(i+1,j,k,n);
        fabcz(i,j,k,n) = fabcz(i+1,j,k,n);    
        fabnew(i,j,k,n) = fabnew(i+1,j,k,n);  
    }else if(i==phihix && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i-1,j,k,n);
        fabcy(i,j,k,n) = fabcy(i-1,j,k,n);
        fabcz(i,j,k,n) = fabcz(i-1,j,k,n);
        fabnew(i,j,k,n) = fabnew(i-1,j,k,n);  }

    if(j==philoy && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i,j+1,k,n);    
        fabcy(i,j,k,n) = fabcy(i,j+1,k,n);    
        fabcz(i,j,k,n) = fabcz(i,j+1,k,n);
        fabnew(i,j,k,n) = fabnew(i,j+1,k,n);  
    }else if(j==phihiy && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i,j-1,k,n);
        fabcy(i,j,k,n) = fabcy(i,j-1,k,n);
        fabcz(i,j,k,n) = fabcz(i,j-1,k,n);
        fabnew(i,j,k,n) = fabnew(i,j-1,k,n);  }

    if(k==philoz){
        fabcx(i,j,k,n) = fabcx(i,j,k+1,n);    
        fabcy(i,j,k,n) = fabcy(i,j,k+1,n);
        fabcz(i,j,k,n) = fabcz(i,j,k+1,n);    
        fabnew(i,j,k,n) = fabnew(i,j,k+1,n);  
    }else if(k==phihiz){
        fabcx(i,j,k,n) = fabcx(i,j,k-1,n);
        fabcy(i,j,k,n) = fabcy(i,j,k-1,n);
        fabcz(i,j,k,n) = fabcz(i,j,k-1,n);
        fabnew(i,j,k,n) = fabnew(i,j,k-1,n);  }
}
//-----------------------------------------------------------------------------------------
 // Update the conserved variables at the end of diffusion step
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void diff_updatephi(int i, int j, int k, int n,
            amrex::Array4<amrex::Real> const& fabnew, 
            amrex::Array4<amrex::Real> const& fldx, amrex::Array4<amrex::Real> const& fldy, amrex::Array4<amrex::Real> const& fldz,
            amrex::Real dtdx, amrex::Real dtdy, amrex::Real dtdz)
{
    amrex::Real temp = fabnew(i,j,k,n);
    fabnew(i,j,k,n) = temp
                    - dtdx*(fldx(i+1,j,k,n) - fldx(i,j,k,n))
                    - dtdy*(fldy(i,j+1,k,n) - fldy(i,j,k,n)) 
                    - dtdz*(fldz(i,j,k+1,n) - fldz(i,j,k,n));
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void phidiff_bc(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fabnew,
             int philox, int philoy, int philoz,
             int phihix, int phihiy, int phihiz)
{
    using namespace amrex;

    if(i == philox && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabnew(i,j,k,n) = fabnew(i+1,j,k,n);
    }else if(i == phihix && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabnew(i,j,k,n) = fabnew(i-1,j,k,n);    }

    if(j == philoy && (k > philoz && k < phihiz)){
        fabnew(i,j,k,n) = fabnew(i,j+1,k,n);
    }else if(j == phihiy && (k > philoz && k < phihiz)){
        fabnew(i,j,k,n) = fabnew(i,j-1,k,n);
    }

    if(k == philoz){
        fabnew(i,j,k,n) = fabnew(i,j,k+1,n);
    }else if(k == phihiz){
        fabnew(i,j,k,n) = fabnew(i,j,k-1,n);
    }
}
//-----------------------------------------------------------------------------------------
// Update the partially convected variables (fabcx, fabcy, fabcz) before prelimiting fluxes
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void diff_updatephiconv(int i, int j, int k, int n,
            amrex::Array4<amrex::Real> const& fabcx, amrex::Array4<amrex::Real> const& fabcy, amrex::Array4<amrex::Real> const& fabcz,
            amrex::Array4<amrex::Real> const& fldx, amrex::Array4<amrex::Real> const& fldy, amrex::Array4<amrex::Real> const& fldz,
            amrex::Real dtdx, amrex::Real dtdy, amrex::Real dtdz)
{

    fabcx(i,j,k,n) = fabcx(i,j,k,n) + dtdx*(fldx(i+1,j,k,n) - fldx(i,j,k,n));
    fabcy(i,j,k,n) = fabcy(i,j,k,n) + dtdy*(fldy(i,j+1,k,n) - fldy(i,j,k,n));
    fabcz(i,j,k,n) = fabcz(i,j,k,n) + dtdz*(fldz(i,j,k+1,n) - fldz(i,j,k,n));
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void phidiffconv_bc(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fabcx, amrex::Array4<amrex::Real> const& fabcy, amrex::Array4<amrex::Real> const& fabcz,
             int philox, int philoy, int philoz, int phihix, int phihiy, int phihiz    )
{
    if(i == philox && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i+1,j,k,n); 
        fabcy(i,j,k,n) = fabcy(i+1,j,k,n);
        fabcz(i,j,k,n) = fabcz(i+1,j,k,n);  }
    else if(i == phihix && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i-1,j,k,n);
        fabcy(i,j,k,n) = fabcy(i-1,j,k,n);
        fabcz(i,j,k,n) = fabcz(i-1,j,k,n);  }

    if(j == philoy && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i,j+1,k,n);
        fabcy(i,j,k,n) = fabcy(i,j+1,k,n);
        fabcz(i,j,k,n) = fabcz(i,j+1,k,n);  }
    else if(j == phihiy && (k > philoz && k < phihiz)){
        fabcx(i,j,k,n) = fabcx(i,j-1,k,n); 
        fabcy(i,j,k,n) = fabcy(i,j-1,k,n);
        fabcz(i,j,k,n) = fabcz(i,j-1,k,n);  } 

    if(k == philoz){
        fabcx(i,j,k,n) = fabcx(i,j,k+1,n);
        fabcy(i,j,k,n) = fabcy(i,j,k+1,n);
        fabcz(i,j,k,n) = fabcz(i,j,k+1,n);  }
    else if(k == phihiz){
        fabcx(i,j,k,n) = fabcx(i,j,k-1,n); 
        fabcy(i,j,k,n) = fabcy(i,j,k-1,n);
        fabcz(i,j,k,n) = fabcz(i,j,k-1,n);  }          
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void get_flux_fracs(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fltx, 
             amrex::Array4<amrex::Real> const& flty, 
             amrex::Array4<amrex::Real> const& fltz,
             amrex::Array4<amrex::Real> const& fab, 
             amrex::Array4<amrex::Real> const& frin,
             amrex::Array4<amrex::Real> const& frout)
{
  amrex::Real umin = amrex::min(fab(i-1,j,k,n), fab(i,j-1,k,n), fab(i,j,k-1,n), fab(i,j,k,n), fab(i+1,j,k,n), fab(i,j+1,k,n), fab(i,j,k+1,n));
  amrex::Real umax = amrex::max(fab(i-1,j,k,n), fab(i,j-1,k,n), fab(i,j,k-1,n), fab(i,j,k,n), fab(i+1,j,k,n), fab(i,j+1,k,n), fab(i,j,k+1,n));
      
  frin(i,j,k,n)  = (umax-fab(i,j,k,n))/(epsilon
                 + amrex::max(fltx(i,j,k,n),0.0) - amrex::min(fltx(i+1,j,k,n),0.0)
                 + amrex::max(flty(i,j,k,n),0.0) - amrex::min(flty(i,j+1,k,n),0.0)
                 + amrex::max(fltz(i,j,k,n),0.0) - amrex::min(fltz(i,j,k+1,n),0.0));
  frout(i,j,k,n) =  (fab(i,j,k,n) - umin)/(epsilon
                 + amrex::max(fltx(i+1,j,k,n),0.0) - amrex::min(fltx(i,j,k,n),0.0)
                 + amrex::max(flty(i,j+1,k,n),0.0) - amrex::min(flty(i,j,k,n),0.0)
                 + amrex::max(fltz(i,j,k+1,n),0.0) - amrex::min(fltz(i,j,k,n),0.0));
}
//-------------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void correct_phi (int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fabnew, amrex::Array4<amrex::Real> const& fabtemp,
             amrex::Array4<amrex::Real> const& fltx, amrex::Array4<amrex::Real> const& flty, amrex::Array4<amrex::Real> const& fltz,
             amrex::Real romin)
{
  fabnew(i,j,k,n) = fabtemp(i,j,k,n) 
                  - (fltx(i+1,j,k,n) - fltx(i,j,k,n)) 
                  - (flty(i,j+1,k,n) - flty(i,j,k,n))
                  - (fltz(i,j,k+1,n) - fltz(i,j,k,n));
  if(n == ro && fabnew(i,j,k,n) < romin){fabnew(i,j,k,n) = romin;}
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void phicorr_bc(int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& fabnew,
             int philox, int philoy, int philoz, int phihix, int phihiy, int phihiz    )
{
    if((i == philox) && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabnew(i,j,k,n) = fabnew(i+1,j,k,n);  }
    else if(i == phihix && (j > philoy && j < phihiy) && (k > philoz && k < phihiz)){
        fabnew(i,j,k,n) = fabnew(i-1,j,k,n); }

    if(j == philoy){
        fabnew(i,j,k,n) = fabnew(i,j+1,k,n);  }
    else if(j == phihiy){
        fabnew(i,j,k,n) = fabnew(i,j-1,k,n);   } 

    if(k == philoz){
        fabnew(i,j,k,n) = fabnew(i,j,k+1,n);  }
    else if(k == phihiz){
        fabnew(i,j,k,n) = fabnew(i,j,k-1,n);   }            
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void scale_x_flux (int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flxx, amrex::Array4<amrex::Real> const& fltx,
             amrex::Real dxdt, amrex::Real dt, amrex::Real /*dx*/, amrex::Real dy, amrex::Real dz )
{
  flxx(i,j,k,n) = (flxx(i,j,k,n) + dxdt*fltx(i,j,k,n))*dy*dz*dt;
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void scale_y_flux (int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flxy, amrex::Array4<amrex::Real> const& flty,
             amrex::Real dydt, amrex::Real dt, amrex::Real dx, amrex::Real /*dy*/, amrex::Real dz)
{
  flxy(i,j,k,n) = (flxy(i,j,k,n) + dydt*flty(i,j,k,n))*dx*dz*dt;
}
//-----------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void scale_z_flux (int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& flxz, amrex::Array4<amrex::Real> const& fltz,
             amrex::Real dzdt, amrex::Real dt, amrex::Real dx, amrex::Real dy, amrex::Real /*dz*/)
{
  flxz(i,j,k,n) = (flxz(i,j,k,n) + dzdt*fltz(i,j,k,n))*dx*dy*dt;
}
#endif


