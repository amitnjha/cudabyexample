#include <stdio.h>

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( 1 );}}

void* big_random_block( int size ) {
    unsigned char *data = (unsigned char*)malloc( size );
    HANDLE_NULL( data );
    for (int i=0; i<size; i++)
        data[i] = rand();

    return data;
}

#define SIZE (100 * 1024 * 1024)

int main(void){

  unsigned char *buffer =  (unsigned char*)big_random_block(SIZE);
  unsigned int histo[256];

  for(int i = 0; i<256;i++){
    histo[i] = 0;
  }

  for(int i = 0; i < SIZE; i++){
    histo[buffer[i]]++;
  }

  long histoCount  = 0;

  for (int i = 0; i < 256; i++){
    histoCount += histo[i];
  }

  printf("Histogram Sum :%ld\n", histoCount);
  
}

