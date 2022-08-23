#include <iostream>
#include <stdio.h>
#include <cuda.h>
using namespace std;
__global__ void kernel(int starting_point,int *d_b_train,int *d_b_result,int *d_ll,int *d_src,int *d_des,int *d_size,int *d_off,
int volatile *d_vol,int *d_class_off,int *d_b_classs,int *d_b_scr,int *d_b_des, int *d_b_seat)
{
    //counter to ensure that it will not stop in the middle in a block
    __shared__ unsigned counter;
    int train_no,s_details,temp,s,temp1;	

    //index which is going to be different for different batches.
    int index=starting_point+blockIdx.x*1024+threadIdx.x;
    
    train_no=d_b_train[index];
    s=d_class_off[train_no]+d_b_classs[index];
    s_details=d_off[train_no]+d_b_classs[index]*d_size[train_no];
    temp1=1;
    temp = 0;
    __syncthreads();
    
    int d,s1,s2,temp2;	
    if(d_b_scr[index]>d_b_des[index])
    {
      if(d_b_des[index]>d_des[train_no])
        s1=d_b_des[index]-d_des[train_no];
      else
        s1=d_des[train_no]-d_b_des[index];
      if(d_b_scr[index]>d_des[train_no])
        d=d_b_scr[index]-d_des[train_no];
      else
        d=d_des[train_no]-d_b_scr[index];
      if(s1>d)
        s2=s1-d;
      else
        s2=d-s1;				    			
    }
    else 
    {
      if(d_b_scr[index]>d_src[train_no])
        s1=d_b_scr[index]-d_src[train_no];
      else
        s1=d_src[train_no]-d_b_scr[index];
      if(d_b_des[index]>d_src[train_no])
        d=d_b_des[index]-d_src[train_no];
      else
        d=d_src[train_no]-d_b_des[index];
      if(s1>d)
        s2=s1-d;
      else
        s2=d-s1;
    }
    counter=1;	
    do
    {
      counter = 1;			
      if(temp1)
        atomicMin(&(d_ll[s]),index);
      __syncthreads();	
      if(temp1)
      {					
        if(d_ll[s]==index)
        {
          for(int i=s_details+s1;i<s_details+d;i++)
          {
              d_vol[i]=d_vol[i]-d_b_seat[index];				
              if(d_vol[i]<0)
              {	
                  temp=1;
                  temp2=i;
                  break;						
              }
          }				
          if(temp)
          {
            for(int i=s_details+s1;i<=temp2;i++)
                d_vol[i] += d_b_seat[index];
            d_b_result[index]=0;
            temp=0;
          }
          else
            d_b_result[index] = s2 * d_b_seat[index];
          temp1=0;
          d_ll[s]=10000;	       
        }
        else if(temp1)
          counter=0;
      }
      __syncthreads();
          
    }while(!counter);    
    __syncthreads();
}
int main()
{
	  // variable declarations...
    int Num,*train,*classs,*src,*des,*src_des,*off,*seat,*capacity,*class_off,*d_train,
    *d_classs,*d_src,*d_des,*d_src_des,*d_off,*d_seat,*d_capacity,*d_class_off;
	  cin>>Num;
	
    //Allocating memory for the Host arrays
    train=(int*)malloc(Num * sizeof (int));
    classs=(int*)malloc(Num * sizeof (int));
    src=(int*)malloc(Num * sizeof (int));
    des=(int*)malloc(Num * sizeof (int)); 
      
    // Allocate memory on gpu
    cudaMalloc(&d_train, (Num) * sizeof(int));
    cudaMalloc(&d_classs, (Num) * sizeof(int));
    cudaMalloc(&d_src, (Num) * sizeof(int));
    cudaMalloc(&d_des, (Num) * sizeof(int));
    cudaMalloc(&d_src_des, (Num) * sizeof(int));
    cudaMalloc(&d_off, (Num) * sizeof(int));	
    cudaMalloc(&d_capacity, (25 * Num) * sizeof(int));
    cudaMalloc(&d_class_off, (25 * Num) * sizeof(int));
    cudaMalloc(&d_seat, (25 * Num * 50) * sizeof(int));
    
    //Some more needed variables
    off=(int*)malloc(Num * sizeof (int));
    src_des=(int*)malloc(Num * sizeof (int));
    seat=(int*)malloc(25 * Num * 50 * sizeof (int));
    capacity=(int*)malloc(25 * Num * sizeof (int)); 
    class_off=(int*)malloc(Num * sizeof (int));
    int t11=0, t12=0;
    for(int i=0;i<Num;i++)
    {
      cin>>train[i];
      cin>>classs[i];
      cin>>src[i];
      cin>>des[i];
      src_des[i]=abs(src[i]-des[i]);
      off[i]=t12;
      class_off[i]=t11;
      for(int j=0;j<classs[i];j++)
      {
        int temp,max_cap;
        cin>>temp>>max_cap;
        for(int k=t12;k<t12+src_des[i];k++)
          seat[k]=max_cap;
        t12+=src_des[i];
        capacity[t11]=max_cap;
        t11++;
      }
    }

	  //Transfer the input host arrays to the device 
    cudaMemcpy(d_train, train, Num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_classs, classs, Num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src, Num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_des, des, Num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_des, src_des, Num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, off, Num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity, t11 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seat, seat, t12 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_off, class_off, t12 * sizeof(int), cudaMemcpyHostToDevice);
    
    
    int *h_l,*d_l;
    h_l=(int*)malloc(t11*sizeof(int));
    cudaMalloc(&d_l, (t11) * sizeof(int));
    int ashok = ceil((float(t11)/1024));
    for(int j=0;j<ashok;j++)
    {
      for(int i=0;i<1024;i++)
      {
        int id =  j * 1024 + i;
        if(id<t11)
          h_l[i]=10000;
      }
    }
    cudaMemcpy(d_l,h_l,t11*sizeof(int),cudaMemcpyHostToDevice);

	  //2nd part of the input


    int noof_batches,noof_req,*b_train,*b_classs,*b_scr,*b_des,*b_seat,*b_result,*b_size;
    cin>>noof_batches;
    //main loop
    for(int i=0;i<noof_batches;i++)
    {
      cin>>noof_req;
      b_train=(int*)malloc(noof_req * sizeof(int));
      b_classs=(int*)malloc(noof_req * sizeof(int));
      b_scr=(int*)malloc(noof_req * sizeof(int));
      b_des=(int*)malloc(noof_req * sizeof(int));
      b_seat=(int*)malloc(noof_req * sizeof(int));
      b_result=(int*)malloc(noof_req * sizeof(int));
      b_size=(int*)malloc(noof_req * sizeof (int));

      int *d_b_train, *d_b_classs, *d_b_scr, *d_b_des, *d_b_size, *d_b_seat, *d_b_result, id;

      cudaMalloc(&d_b_train,noof_req * sizeof(int));
      cudaMalloc(&d_b_classs,noof_req * sizeof(int));
      cudaMalloc(&d_b_scr,noof_req * sizeof(int));
      cudaMalloc(&d_b_des,noof_req * sizeof(int));
      cudaMalloc(&d_b_seat,noof_req * sizeof(int));
      cudaMalloc(&d_b_result,noof_req * sizeof(int));
      cudaMalloc(&d_b_size,noof_req * sizeof(int));	

      for(int iz=0;iz<noof_req;iz++)
      {
        cin>>id;
        cin>>b_train[iz];
        cin>>b_classs[iz];
        cin>>b_scr[iz];
        cin>>b_des[iz];
        cin>>b_seat[iz];
        b_size[iz]=abs(b_scr[iz]-b_des[iz]);	
      }
      
      cudaMemcpy(d_b_train,b_train,noof_req * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_classs,b_classs,noof_req * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_scr,b_scr,noof_req * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_des,b_des,noof_req * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_seat,b_seat,noof_req * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_size,b_size,noof_req * sizeof(int),cudaMemcpyHostToDevice);
      
      //making requests into blocks of 1024
      int blocks=noof_req/1024;
      int reamining=noof_req%1024;
      
      int temp=blocks;
      while(temp)
      {	
        int temp1=blocks-temp;
        //each with 1024 elements
        dim3 grid1(1,1,1);
        dim3 block1(1024,1,1);
        kernel<<<grid1,block1>>>(1024*temp1,d_b_train,d_b_result,d_l,d_src,d_des,d_src_des,d_off,
        d_seat,d_class_off,d_b_classs,d_b_scr,d_b_des,d_b_seat);
        cudaDeviceSynchronize();
        temp--;
      }
      //For the remaining elements
      dim3 grid1(1,1,1);
      dim3 block1(reamining,1,1);
      kernel<<<grid1,block1>>>(1024*blocks,d_b_train,d_b_result,d_l,d_src,d_des,d_src_des,d_off,
      d_seat,d_class_off,d_b_classs,d_b_scr,d_b_des,d_b_seat);

      // copy the result back...
      cudaMemcpy(b_result,d_b_result,noof_req * sizeof(int),cudaMemcpyDeviceToHost);
      
      //Output
      int noof_succ=0,noof_seats=0;
      for(int k=0;k<noof_req;k++)
      {
        if(b_result[k])
        {
          cout<<"success"<<endl;
          noof_succ++;
          noof_seats+=b_result[k];
        }
        else
          cout<<"failure"<<endl;
      }
      cout<<noof_succ<<" "<<noof_req-noof_succ<<endl;
      cout<<noof_seats<<endl;
      noof_seats=0;

      // deallocate the memory...
      free(b_train);
      free(b_classs);
      free(b_scr);
      free(b_des);
      free(b_seat);
      free(b_result);
    }

}
