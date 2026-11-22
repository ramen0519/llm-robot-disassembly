/* Orthographic per-pixel depth test for original triangles. No decimation or occlusion guessing. */
#include <math.h>
#include <stdint.h>
void paint(int n, int nf, const double *xy, const double *world, const int32_t *faces,
           const double *rgb, double *depth, uint8_t *image) {
    for (int t=0;t<nf;t++) {
        const int32_t *f=faces+3*t;
        const double *a=xy+3*f[0], *b=xy+3*f[1], *c=xy+3*f[2];
        double ux=b[0]-a[0], uy=b[1]-a[1], vx=c[0]-a[0], vy=c[1]-a[1];
        double det=ux*vy-vx*uy;
        if (fabs(det)<1.e-10) continue;
        int x0=fmax(0,floor(fmin(a[0],fmin(b[0],c[0]))));
        int x1=fmin(n-1,ceil(fmax(a[0],fmax(b[0],c[0]))));
        int y0=fmax(0,floor(fmin(a[1],fmin(b[1],c[1]))));
        int y1=fmin(n-1,ceil(fmax(a[1],fmax(b[1],c[1]))));
        const double *wa=world+3*f[0], *wb=world+3*f[1], *wc=world+3*f[2];
        double u[3],v[3];
        for(int i=0;i<3;i++){u[i]=wb[i]-wa[i];v[i]=wc[i]-wa[i];}
        double nx=u[1]*v[2]-u[2]*v[1], ny=u[2]*v[0]-u[0]*v[2], nz=u[0]*v[1]-u[1]*v[0];
        double sh=.57+.43*fabs(.28*nx+.39*ny+.877*nz)/fmax(1.e-12,sqrt(nx*nx+ny*ny+nz*nz));
        for(int y=y0;y<=y1;y++) for(int x=x0;x<=x1;x++){
            double dx=x+.5-a[0],dy=y+.5-a[1];
            double s=(dx*vy-dy*vx)/det,h=(ux*dy-uy*dx)/det;
            double z=a[2]+s*(b[2]-a[2])+h*(c[2]-a[2]);
            int idx=y*n+x;
            if(s>=0 && h>=0 && s+h<=1 && z>depth[idx]){
                depth[idx]=z;
                for(int ch=0;ch<3;ch++) image[3*idx+ch]=(uint8_t)fmin(255,rgb[ch]*sh);
            }
        }
    }
}
