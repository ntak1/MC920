/*   Connected Components
 *
 *   Input: binary image ("raw pbm" format)
 *   Output: number of connected components
             binary image containing set of connected components (4-connected neighbors)
 *
 *   Helio Pedrini
 */
 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits.h>

#define BACKGROUND    1     /* White */
#define FOREGROUND    0     /* Black */

int **binary_image;

int row, col, width, height, num_comp,
    *label;


/* create 2D image containing integer values */
int **create_image(width, height)
int width, height;
{
 int i;
 int **prow, *pdata;
 
 /* allocate storage from the heap for the appropriate number of bytes */
 pdata=(int *) calloc(width*height, sizeof(int));

 if (pdata == (int*) NULL) {
   printf("No heap space for image\n");
   exit(1);
 }
 
 /* allocate storage for a pointer array on the heap */
 prow=(int **) calloc(height, sizeof(int *));

 if (prow == (int**) NULL) {
   printf("No heap space for image\n");
   exit(1);
 } 
 
 for (i=0; i < height; i++) {
  prow[i]=pdata;  /* store pointers to rows */
  pdata+=width;   /* move to next row */
 }
 
 return prow;     /* pointer to 2D image */
}


/* release heap storage for the image */
void free_image(pa)
int **pa;
{
 void free();
 free(*pa);    /* free the data */
 free(pa);     /* free the row pointers */
}


/* join label[p] and label[q] to the same equivalence class (iterative function) */
void equiv_iterative(p,q)
int p,q;
{
 int temp;
 
 while (label[p] != p) {
  p=label[p];
  if (p < q) {
    temp=p;
    p=q;
    q=temp;
  }
 }
	
 label[p]=q;
}


/* join label[p] and label[q] to the same equivalence class (recursive function) */
void equiv_recursive(p,q)
int p,q;
{
 int temp;
 
 if (label[p] == p)
   label[p]=q;
 else if (label[p] < q)
        equiv_recursive(q, label[p]);
      else equiv_recursive(label[p], q);
}


/* find connected components by using region labeling technique,
 * where the image pixels corresponding to different components 
 * receive different labels.
 */
void labeling()
{
 int k;

 label=(int *) calloc(height*width,sizeof(int));

 /* for each pixel, assign a label equal to its index */ 
 for (row=0; row<height; row++)
  for (col=0; col<width; col++)
   label[row*width+col]=row*width + col;

 /* find connected components, assigning labels to each component */
 for (row=0; row<height; row++)
  for (col=0; col<width; col++)
   if (binary_image[row][col] == FOREGROUND) {
     if ((row > 0) && (binary_image[row-1][col] == FOREGROUND))
       equiv_iterative(row*width+col, (row-1)*width+col);
     if ((col > 0) && (binary_image[row][col-1] == FOREGROUND))
       equiv_iterative(row*width+col, row*width+col-1);
   }
   else label[row*width+col]=-1;
 
 /* re-arrange labels */
 num_comp=0;
 for (row=0; row<height; row++)
  for (col=0; col<width; col++) {
   k=row*width + col;
   if (label[k] != -1) {
     if (label[k] == k) {
	num_comp++;      /* new connected component */
	label[k]=num_comp;
     }
     else label[k]=label[label[k]];
   }
  }
}


/* draw a bounding box (offset by one pixel) for each connected component */
void bounding_box()
{
 int k, *x_1, *y_1, *x_2, *y_2;
 
 printf("\nNumber of connected components: %d\n", num_comp);

 x_1=(int *) calloc(num_comp,sizeof(int));
 x_2=(int *) calloc(num_comp,sizeof(int));
 y_1=(int *) calloc(num_comp,sizeof(int));
 y_2=(int *) calloc(num_comp,sizeof(int));
      
 for (k=0; k<num_comp; k++) {
  x_1[k]=y_1[k]=INT_MAX;
  x_2[k]=y_2[k]=INT_MIN;
 }
 
 for (row=0; row<height; row++)
  for (col=0; col<width; col++) {
   k=label[row*width+col];
   if (k != -1) {
     k=k-1;
     if (x_1[k] > col) x_1[k]=col;
     if (y_1[k] > row) y_1[k]=row;
     if (x_2[k] < col) x_2[k]=col;
     if (y_2[k] < row) y_2[k]=row;
     binary_image[row][col]=FOREGROUND;
   }
   else binary_image[row][col]=BACKGROUND;
  }
 
 
 /* set bounding box pixels */
 for (k=0; k<num_comp; k++) {  
  for (row=y_1[k]-1; row<=y_2[k]+1; row++)
   if ((row >= 0) && (row < height) && (x_1[k]-1 >= 0))
     binary_image[row][x_1[k]-1]=FOREGROUND;

  for (row=y_1[k]-1; row<=y_2[k]+1; row++)
   if ((row >= 0) && (row < height) && (x_2[k]+1 < width))
     binary_image[row][x_2[k]+1]=FOREGROUND;
 
  for (col=x_1[k]-1; col<=x_2[k]+1; col++)
   if ((col >= 0) && (col < width) && (y_1[k]-1 >= 0))
     binary_image[y_1[k]-1][col]=FOREGROUND;

  for (col=x_1[k]-1; col<=x_2[k]+1; col++)
   if ((col >= 0) && (col < width) && (y_2[k]+1 < height))
     binary_image[y_2[k]+1][col]=FOREGROUND;
 }
}


/* read image file ("raw pbm"): every 8 pixels are combined to fill one 8 bit byte, which is
  written as an ascii character into the "pbm" file. At the end of a row, the remaining part 
  of a byte is filled with zeros, and each new row is started the beginning of a byte */
void read_image_pbm(argc, argv)
 int argc;
 char **argv;
{
 unsigned char ch;
 int width_bytes, bits;
 static int mask[8] = {128, 64, 32, 16, 8, 4, 2, 1};
 char str[80];
 FILE *fin;
  
 if (argc != 3) {
   printf("\nUsage: %s img_in[.pbm] img_out[.pbm]\n\n", argv[0]);
   exit(1);
 }
 else if ((fin = fopen(argv[1], "r")) == NULL) {
   printf("Unable to open input file: %s\n\n", argv[1]);
   exit(1);  
 }      
  
 /* read header */
 fgets(str, 80, fin);
 fgets(str, 80, fin);
 fscanf(fin, "%d %d\n", &width, &height);
 
 printf("\nImage size (width x height): %d x %d  pixels\n", width, height);
       
 binary_image=create_image(width, height);
  
 /* read image */
 width_bytes=(int) ceil((double) width/8);

 for (row=0; row<height; row++)
  for (col=0; col<width_bytes; col++) {
   fscanf(fin, "%c", &ch);
   for (bits=0; bits<8 && col*8+bits<width; bits++)
    if (ch & mask[bits]) binary_image[row][col*8+bits]=FOREGROUND;
    else binary_image[row][col*8+bits]=BACKGROUND;
  }
   
 fclose(fin);
}


/* save binary image ("raw pbm" format) */
void save_image_pbm(argc, argv)
int argc;
char **argv;
{
 unsigned char ch;
 int width_bytes, bits;
 static int mask[8] = {128, 64, 32, 16, 8, 4, 2, 1};
 FILE *fout;
 
 if ((fout = fopen(argv[2],"w")) == NULL) {
   printf("Unable to open output file\n");
   exit(1);
 }
  
 /* save "pbm" header */
 fprintf(fout, "P4\n");
 fprintf(fout, "# CREATOR: Connected components program written by Helio Pedrini\n");
 fprintf(fout, "%d %d\n", width, height);
 
 /* save binary image */
 width_bytes=(int) ceil((double) width/8);
 
 for (row=0; row<height; row++)
  for (col=0; col<width_bytes; col++) {
    ch=0;
    for (bits=0; bits<8 && col*8+bits<width; bits++)
     if (binary_image[row][col*8+bits] == FOREGROUND)
       ch=ch | mask[bits];
    fprintf(fout, "%c", ch);
  }
  
 fclose(fout);
}


int main(argc, argv)
int argc;
char **argv;
{  
 read_image_pbm(argc, argv);
 labeling();
 bounding_box();
 save_image_pbm(argc, argv);
 free_image(binary_image);
}
