#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <omp.h>

static double now_s(){timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

extern "C" double unified_spmv_benchmark(int n,const int* row,const int* col,const double* data,
 int repetitions,int threads,double* checksum){
 std::vector<double> x(n),y(n);
 for(int i=0;i<n;i++)x[i]=std::sin(.017*i)+.1;
 omp_set_num_threads(std::max(1,threads));
 // Warm-up and timed calls use exactly the same CSR arithmetic.
 for(int repeat=0;repeat<3;repeat++){
  #pragma omp parallel for schedule(static)
  for(int i=0;i<n;i++){double v=0;for(int a=row[i];a<row[i+1];a++)v+=data[a]*x[col[a]];y[i]=v;}
 }
 const double start=now_s();
 #pragma omp parallel
 {
  for(int repeat=0;repeat<repetitions;repeat++){
   #pragma omp for schedule(static)
   for(int i=0;i<n;i++){double v=0;for(int a=row[i];a<row[i+1];a++)v+=data[a]*x[col[a]];y[i]=v;}
  }
 }
 const double elapsed=now_s()-start;
 *checksum=0;for(double v:y)*checksum+=v;
 return elapsed;
}

// All modes use the same CSR, independent-RHS batching, three-term Lanczos,
// shifted LDL factorization, and stopping rule.  mode=0: scalar resistance;
// mode=1: additionally differentiate LDL to obtain projected ||x||^2;
// mode=2/3: additionally form every n-vector solution by Q L^{-T} updates,
// and check its actual residual.  mode=3 returns vectors (count must be 1).
extern "C" int unified_batch(int n,const int* row,const int* col,const double* data,
 int count,const long long* edges,int q,const double* shifts,double tol,int threads,
 int blocksize,int mode,int maxiter,double deadline,double* out,double* norm_out,
 double* residual_out,int* iterations,long long* padded_steps,long long* true_checks,
 double* vector_out){
 std::atomic<int> failure{0};
 omp_set_num_threads(std::max(1,threads));
 const int blocks=(count+blocksize-1)/blocksize;
 #pragma omp parallel for schedule(static) if(blocks>1)
 for(int block=0;block<blocks;block++){
  if(failure.load())continue;
  int first=block*blocksize,B=std::min(blocksize,count-first);
  std::vector<double> v((size_t)n*B,0),prev((size_t)n*B,0),z((size_t)n*B);
  std::vector<double> beta(B,0),diag(B),next(B),pivot(B*q),forward(B*q),d(B*q,0),rr(B*q);
  std::vector<double> dpivot(B*q,0),dforward(B*q,0),dd(B*q,0),mult(B*q,0),coef(B*q,0);
  std::vector<double> X,dir;
  if(mode>=2){X.assign((size_t)n*B*q,0);dir.assign((size_t)n*B*q,0);}
  std::vector<int> done(B,0);
  for(int c=0;c<B;c++){
   v[(size_t)edges[2*(first+c)]*B+c]=1/std::sqrt(2.0);
   v[(size_t)edges[2*(first+c)+1]*B+c]=-1/std::sqrt(2.0);
  }
  int remaining=B;
  for(int step=0;step<maxiter;step++){
   if(deadline>0 && now_s()>deadline){failure.store(2);break;}
   padded_steps[block]+=B;
   std::fill(z.begin(),z.end(),0.0);
   for(int i=0;i<n;i++){
    double* zi=z.data()+(size_t)i*B;
    for(int a=row[i];a<row[i+1];a++){
     const double* vi=v.data()+(size_t)col[a]*B;double av=data[a];
     #pragma omp simd
     for(int c=0;c<B;c++)zi[c]+=av*vi[c];
    }
   }
   std::fill(diag.begin(),diag.end(),0.0);
   for(int i=0;i<n;i++)for(int c=0;c<B;c++){
    size_t a=(size_t)i*B+c;z[a]-=prev[a]*beta[c];diag[c]+=v[a]*z[a];
   }
   std::fill(next.begin(),next.end(),0.0);
   for(int i=0;i<n;i++)for(int c=0;c<B;c++){
    size_t a=(size_t)i*B+c;z[a]-=v[a]*diag[c];next[c]+=z[a]*z[a];
   }
   for(int c=0;c<B;c++){
    if(done[c]){next[c]=1;continue;}
    next[c]=std::sqrt(next[c]);
    for(int j=0;j<q;j++){
     int a=c*q+j;
     if(step==0){pivot[a]=diag[c]+shifts[j];forward[a]=std::sqrt(2.0);dpivot[a]=1;}
     else {
      const double oldp=pivot[a],oldf=forward[a],oldpd=dpivot[a],oldfd=dforward[a];
      mult[a]=beta[c]/oldp;
      pivot[a]=diag[c]+shifts[j]-beta[c]*mult[a];
      forward[a]=-mult[a]*oldf;
      if(mode==1){
       dpivot[a]=1+beta[c]*beta[c]*oldpd/(oldp*oldp);
       dforward[a]=-mult[a]*oldfd+mult[a]*oldf*oldpd/oldp;
      }
     }
     if(!(pivot[a]>0)||!std::isfinite(pivot[a])){failure.store(1);break;}
     const double term=forward[a]*forward[a]/pivot[a];d[a]+=term;
     if(mode==1)dd[a]+=2*forward[a]*dforward[a]/pivot[a]-term*dpivot[a]/pivot[a];
     coef[a]=forward[a]/pivot[a];
     rr[a]=next[c]*std::abs(coef[a]);
    }
    iterations[first+c]=step+1;
   }
   if(failure.load())break;
   if(mode>=2){
    for(int i=0;i<n;i++)for(int c=0;c<B;c++)if(!done[c]){
     const double vc=v[(size_t)i*B+c];size_t base=((size_t)i*B+c)*q;
     #pragma omp simd
     for(int j=0;j<q;j++){
      size_t a=base+j;int cj=c*q+j;
      dir[a]=vc-mult[cj]*dir[a];X[a]+=coef[cj]*dir[a];
     }
    }
   }
   for(int c=0;c<B;c++)if(!done[c]){
    double maxrr=0;for(int j=0;j<q;j++)maxrr=std::max(maxrr,rr[c*q+j]);
    bool stop=maxrr<=std::sqrt(2.0)*tol || next[c]<1e-14;
    if(stop && mode>=2){
     true_checks[block]+=q;
     std::vector<double> tr(q,0),nx(q,0),xr(q,0);
     const int u=edges[2*(first+c)],vtx=edges[2*(first+c)+1];
     for(int i=0;i<n;i++){
      for(int j=0;j<q;j++){
       const size_t a=((size_t)i*B+c)*q+j;double ax=shifts[j]*X[a];
       for(int kk=row[i];kk<row[i+1];kk++)ax+=data[kk]*X[((size_t)col[kk]*B+c)*q+j];
       const double ri=(i==u?1.0:(i==vtx?-1.0:0.0))-ax;
       tr[j]+=ri*ri;nx[j]+=X[a]*X[a];xr[j]+=X[a]*ri;
      }
     }
     maxrr=0;for(int j=0;j<q;j++)maxrr=std::max(maxrr,std::sqrt(tr[j]));
     stop=maxrr<=std::sqrt(2.0)*tol*1.1;
     if(stop){
      for(int j=0;j<q;j++){
       const size_t oi=(first+c)*q+j;
       out[oi]=X[((size_t)u*B+c)*q+j]-X[((size_t)vtx*B+c)*q+j]+xr[j];
       norm_out[oi]=nx[j];residual_out[oi]=std::sqrt(tr[j]);
      }
     }
    }
    if(stop){
     done[c]=1;remaining--;
     if(mode<2)for(int j=0;j<q;j++) {
      const size_t oi=(first+c)*q+j;out[oi]=d[c*q+j];residual_out[oi]=rr[c*q+j];
      norm_out[oi]=mode==1?-dd[c*q+j]:0;
     }
     if(mode==3)for(int i=0;i<n;i++)for(int j=0;j<q;j++)vector_out[(size_t)i*q+j]=X[((size_t)i*B+c)*q+j];
     next[c]=1;
    }
   }
   if(failure.load()||remaining==0)break;
   for(int i=0;i<n;i++)for(int c=0;c<B;c++){
    size_t a=(size_t)i*B+c;prev[a]=v[a];v[a]=done[c]?0:z[a]/next[c];
   }
   for(int c=0;c<B;c++)beta[c]=done[c]?0:next[c];
  }
  if(remaining && !failure.load())failure.store(3);
 }
 return failure.load();
}
