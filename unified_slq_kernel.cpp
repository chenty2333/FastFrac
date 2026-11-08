// Compiled CRN-SLQ kernel, common flags and OpenMP thread count as unified_kernel.cpp.
// Each candidate/probe uses h Lanczos steps. Probe columns are SIMD batched;
// no shared low-rank approximation, sampling shortcut, or surrogate substitution.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>

static double quadrature(std::vector<double> d,std::vector<double> e,double s){
 int h=d.size(); std::vector<double> z(h,0.);z[0]=1.;e[h-1]=0.;
 for(int l=0;l<h;l++)for(int its=0;its<100;its++){
  int m=l;for(;m<h-1;m++){double dd=std::abs(d[m])+std::abs(d[m+1]);if(std::abs(e[m])<=4*std::numeric_limits<double>::epsilon()*dd)break;}
  if(m==l)break;
  double g=(d[l+1]-d[l])/(2*e[l]),rr=std::hypot(g,1.);g=d[m]-d[l]+e[l]/(g+std::copysign(rr,g));
  double sn=1.,cs=1.,p=0.;
  for(int i=m-1;i>=l;i--){double f=sn*e[i],bb=cs*e[i];
   if(std::abs(f)>=std::abs(g)){cs=g/f;rr=std::hypot(cs,1.);e[i+1]=f*rr;sn=1/rr;cs*=sn;}
   else{sn=f/g;rr=std::hypot(sn,1.);e[i+1]=g*rr;cs=1/rr;sn*=cs;}
   g=d[i+1]-p;rr=(d[i]-g)*sn+2*cs*bb;p=sn*rr;d[i+1]=g+p;g=cs*rr-bb;
   f=z[i+1];z[i+1]=sn*z[i]+cs*f;z[i]=cs*z[i]-sn*f;
  }
  d[l]-=p;e[l]=g;e[m]=0.;
 }
 double answer=0.;for(int j=0;j<h;j++){if(!(d[j]>0.)||!std::isfinite(d[j]))return std::numeric_limits<double>::quiet_NaN();answer+=z[j]*z[j]*std::pow(d[j],s);}return answer;
}
extern "C" int unified_slq_family(int n,const int64_t* ptr,const int32_t* col,const double* aval,
 const double* probes,int p,int h,double exponent,const int64_t* edges,int count,
 double* output,int64_t* counts,int threads,int blocksize){
 std::atomic<int64_t> finished(0),probe_count(0),mv_count(0),padded_mv(0);
 std::atomic<int> failure(0);double invnorm=1/std::sqrt((double)n);
 #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
 for(int candidate=0;candidate<count;candidate++){
  int u=edges[2*candidate],v=edges[2*candidate+1];double total=0.;int64_t local_mv=0,local_probe=0,local_padded=0;
  for(int start=0;start<p;start+=blocksize){
   int B=std::min(blocksize,p-start);
   std::vector<double> cur((size_t)n*B),prev((size_t)n*B,0.),z((size_t)n*B),diag(B*h),off(B*h),alpha(B),beta(B,0.),next(B);
   std::vector<int> actual(B,h),done(B,0);
   for(int i=0;i<n;i++)for(int b=0;b<B;b++)cur[(size_t)i*B+b]=probes[(int64_t)(start+b)*n+i]*invnorm;
   for(int step=0;step<h;step++){
    std::fill(z.begin(),z.end(),0.);
    for(int i=0;i<n;i++){
     double* zi=z.data()+(size_t)i*B;
     for(int64_t a=ptr[i];a<ptr[i+1];a++){
      const double* ci=cur.data()+(size_t)col[a]*B;double av=aval[a];
      #pragma omp simd
      for(int b=0;b<B;b++)zi[b]+=av*ci[b];
     }
    }
    if(u>=0)for(int b=0;b<B;b++){double diff=cur[(size_t)u*B+b]-cur[(size_t)v*B+b];z[(size_t)u*B+b]+=diff;z[(size_t)v*B+b]-=diff;}
    std::fill(alpha.begin(),alpha.end(),0.);
    for(int i=0;i<n;i++)for(int b=0;b<B;b++){size_t a=(size_t)i*B+b;z[a]-=beta[b]*prev[a];alpha[b]+=cur[a]*z[a];}
    std::fill(next.begin(),next.end(),0.);
    for(int i=0;i<n;i++)for(int b=0;b<B;b++){size_t a=(size_t)i*B+b;z[a]-=alpha[b]*cur[a];next[b]+=z[a]*z[a];}
    for(int b=0;b<B;b++){
     if(done[b]){next[b]=1.;continue;}
     local_mv++;next[b]=std::sqrt(next[b]);diag[b*h+step]=alpha[b];off[b*h+step]=next[b];
     if(next[b]<1e-13){actual[b]=step+1;done[b]=1;next[b]=1.;}
    }
    local_padded+=B;
    prev.swap(cur);
    for(int i=0;i<n;i++)for(int b=0;b<B;b++){size_t a=(size_t)i*B+b;cur[a]=done[b]?0.:z[a]/next[b];}
    for(int b=0;b<B;b++)beta[b]=done[b]?0.:next[b];
   }
   for(int b=0;b<B;b++){
    std::vector<double>d(diag.begin()+b*h,diag.begin()+b*h+actual[b]),e(off.begin()+b*h,off.begin()+b*h+actual[b]);
    double value=n*quadrature(d,e,exponent);if(!std::isfinite(value))failure=1;total+=value;local_probe++;
   }
  }
  output[candidate]=total/p;finished.fetch_add(1,std::memory_order_relaxed);
  probe_count.fetch_add(local_probe,std::memory_order_relaxed);mv_count.fetch_add(local_mv,std::memory_order_relaxed);padded_mv.fetch_add(local_padded,std::memory_order_relaxed);
 }
 counts[0]=finished;counts[1]=probe_count;counts[2]=mv_count;counts[3]=padded_mv;return failure;
}
