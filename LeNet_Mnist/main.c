#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>  //max_pool
#include <time.h>
#include <stdint.h>

#include <nnType.h>
#include <nnInference.h>
#include <tensor.h>

int train_runs=60000;
int test_runs= 1;
int test_num = 0.0;
double test_success_count = 0.0;

void init(tensor * T)
{
        external(&T[0], 0, 0, 1, 784);  /* [1, 784], layer : 0,  */
        variable(&T[2], 32, 1, 5, 5, "input/Variable");  /* [32, 1, 5, 5], layer : 0,  */
        variable(&T[12], 0, 0, 1, 10, "input/Variable_7");  /* [1, 10], layer : 0,  */
        variable(&T[16], 0, 0, 3136, 1024, "input/Variable_4");  /* [3136, 1024], layer : 0,  */
        variable(&T[17], 0, 0, 1024, 10, "input/Variable_6");  /* [1024, 10], layer : 0,  */
        variable(&T[18], 0, 0, 1, 1024, "input/Variable_5");  /* [1, 1024], layer : 0,  */
        variable(&T[4], 0, 0, 1, 32, "input/Variable_1");  /* [1, 32], layer : 0,  */
        variable(&T[5], 64, 32, 5, 5, "input/Variable_2");  /* [64, 32, 5, 5], layer : 0,  */
        variable(&T[6], 0, 0, 1, 64, "input/Variable_3");  /* [1, 64], layer : 0,  */
}
void kernel(tensor * T)
{
        reshape(&T[3], &T[0], -1, 28, 28, 1);  /* [1, 28, 28, 1], layer : 1,  */
        transpose(&T[7], &T[3], 0, 3, 1, 2);  /* [1, 1, 28, 28], layer : 2,  */
        // Notice: convxbias-> convxt_bias modify by self
        convxt_bias(&T[8], &T[7], &T[2], &T[4], padding = 1, stride = 1, groups = 1);  /* [1, 32, 28, 28], layer : 3,  */
        relu(&T[9], &T[8]);  /* [1, 32, 28, 28], layer : 4,  */
        max_pool(&T[10], &T[9], size = 2, padding = 1, stride = 2);  /* [1, 32, 14, 14], layer : 5,  */
        convxt_bias(&T[11], &T[10], &T[5], &T[6], padding = 1, stride = 1, groups = 1);  /* [1, 64, 14, 14], layer : 6,  */
        relu(&T[13], &T[11]);  /* [1, 64, 14, 14], layer : 7,  */
        max_pool(&T[14], &T[13], size = 2, padding = 1, stride = 2);  /* [1, 64, 7, 7], layer : 8,  */
        transpose(&T[15], &T[14], 0, 2, 3, 1);  /* [1, 7, 7, 64], layer : 9,  */
        reshape(&T[19], &T[15], 0, 0, -1, 3136);  /* [1, 3136], layer : 10,  */
        matmul(&T[20], &T[19], &T[16]);  /* [1, 1024], layer : 11,  */
        add(&T[21], &T[20], &T[18]);  /* [1, 1024], layer : 12,  */
        relu(&T[22], &T[21]);  /* [1, 1024], layer : 13,  */
        matmul(&T[23], &T[22], &T[17]);  /* [1, 10], layer : 14,  */
        add(&T[24], &T[23], &T[12]);  /* [1, 10], layer : 15,  */
        softmax(&T[1], &T[24]);  /* [1, 10], layer : 16,  */
}

int main()
{
    extern tensor T[512];
    init(T);
    int target[10];
    //initialize the input by 28 x 28 (0,1)matrix of the images

    T[0].data = (float * ) calloc(784 * sizeof(float), sizeof(float));
    FILE *image_test;
    FILE *image_test_label;
    image_test = fopen("./mnist/t10k-images-idx3-ubyte", "rb");
    image_test_label = fopen("./mnist/t10k-labels-idx1-ubyte", "rb");
    int useless[1000];
    fread(useless, 1, 16, image_test);
    fread(useless, 1, 8, image_test_label);

    while (!feof(image_test) && !feof(image_test_label))
    { 

        if (image_test == NULL || image_test_label == NULL){
            assert(0);
        }

        unsigned char image_buf[784];
        unsigned char label_buf[10];

        memset(image_buf, 0, 784);
        memset(label_buf, 0, 10);
        fread(image_buf, 1, 784, image_test);
        fread(label_buf, 1, 1, image_test_label);

        for (int i = 0; i < 784; i++)
        {
            T[0].data[i] = ((float)(unsigned int)image_buf[i]) / 255;
        }

        //******** NNEF *********
        kernel(T);
        //***********************

        for (int k = 0; k < 10; k++){
            target[k] = 0;
        }
        int target_value = (unsigned int)label_buf[0];
        target[target_value] = 1;
        double max_value = -99999;
        int max_index = 0;

        for (int k = 0; k < 10; k++)
        {
            if (T[1].data[k] > max_value)
            {
                max_value = T[1].data[k];
                max_index = k;
            }
        }

        //output == target
        if (target[max_index] == 1){
            test_success_count ++;
        }

        test_num ++;

        if ((int)test_num % 1000 == 0){
            printf("test num: %d, success: %f\n", test_num, test_success_count);
        }

        if (test_num>=test_runs) break;
    }

    printf("The success rate: %f\n", test_success_count / test_num); 
    
    for (int i = 0; i < T[1].size; i++)
        printf(" %g", T[1].data[i]);
    printf("\n");
}
