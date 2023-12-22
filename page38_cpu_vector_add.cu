#include <stdio.h>
#include <time.h>


#define N 100000

void add(int *a, int *b, int *c){

  int tid = 0;
  while(tid < N){

    c[tid] = a[tid] + b[tid];
    tid += 1;
  }

}


int main(void) {

  clock_t start,end;
  double time_used;

  
  int a[N], b[N], c[N];

  for (int i = 0; i < N; i++){
    a[i] = -i;
    b[i] = i + i-332;
  }

  start = clock();
  add(a, b, c);

  /*
  for (int i = 0; i < N; i++){
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }*/
  return 0;

}
