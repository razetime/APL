#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <time.h>
#include <assert.h>
#include <limits>
#include <math.h>
#include <float.h>  //max_pool
#include <string>

using namespace std;
//#include <tensor.h>
#include <nnType.h>
//nnKernel nnCompileProgram(const char * str);
//void nnRun(nnKernel &kernel, int type);

#define DEBUG_TIME
#define im2colxGEMM

const int inputsize = 224*224*3;
#define DYNAMIC

#ifdef DYNAMIC
    #define TYPE 1
#else
    #define TYPE 0   
#endif

#include <nnInference.h>
#include <tensor.h>

// New version parser gen the code
float tensor_n;

#if 1
/*
void init(tensor * T)
{
	variable(&T[2], N = 0, C = 0, H = 1, W = 288, "Const");
	variable(&T[3], N = 528, C = 1, H = 3, W = 3, "Const_2");
	variable(&T[4], N = 0, C = 0, H = 1, W = 528, "Const_6");
	variable(&T[5], N = 0, C = 0, H = 1, W = 816, "Const_7");
	variable(&T[6], N = 0, C = 0, H = 1, W = 1344, "Const_10");
	variable(&T[7], N = 0, C = 0, H = 1, W = 816, "Const_16");
	variable(&T[8], N = 0, C = 0, H = 1, W = 528, "Const_17");
	variable(&T[9], N = 0, C = 0, H = 1, W = 1344, "Const_19");
	variable(&T[10], N = 816, C = 1, H = 3, W = 3, "Const_22");
	variable(&T[11], N = 0, C = 0, H = 1, W = 192, "Const_23");
	variable(&T[12], N = 0, C = 0, H = 1, W = 192, "Const_26");
	variable(&T[13], N = 0, C = 0, H = 1, W = 528, "Const_28");
	variable(&T[14], N = 0, C = 0, H = 1, W = 528, "Const_29");
	variable(&T[15], N = 0, C = 0, H = 1, W = 288, "Const_31");
	variable(&T[16], N = 0, C = 0, H = 1, W = 528, "Const_35");
	variable(&T[17], N = 0, C = 0, H = 1, W = 48, "Const_37");
	variable(&T[18], N = 0, C = 0, H = 1, W = 1344, "Const_38");
	variable(&T[19], N = 0, C = 0, H = 1, W = 528, "Const_43");
	variable(&T[20], N = 0, C = 0, H = 1, W = 288, "Const_45");
	variable(&T[21], N = 0, C = 0, H = 1, W = 528, "Const_47");
	variable(&T[22], N = 0, C = 0, H = 1, W = 1344, "Const_49");
	variable(&T[23], N = 0, C = 0, H = 1, W = 816, "Const_53");
	variable(&T[24], N = 0, C = 0, H = 1, W = 816, "Const_56");
	variable(&T[25], N = 0, C = 0, H = 1, W = 528, "Const_58");
	variable(&T[26], N = 816, C = 1, H = 3, W = 3, "Const_65");
	variable(&T[27], N = 0, C = 0, H = 1, W = 192, "Const_66");
	variable(&T[28], N = 0, C = 0, H = 1, W = 1344, "Const_67");
	variable(&T[29], N = 0, C = 0, H = 1, W = 816, "Const_68");
	variable(&T[30], N = 0, C = 0, H = 1, W = 1001, "Const_69");
	variable(&T[31], N = 0, C = 0, H = 1, W = 288, "Const_76");
	variable(&T[32], N = 0, C = 0, H = 1, W = 144, "Const_82");
	variable(&T[33], N = 0, C = 0, H = 1, W = 816, "Const_84");
	variable(&T[34], N = 816, C = 1, H = 3, W = 3, "Const_87");
	variable(&T[35], N = 0, C = 0, H = 1, W = 192, "Const_91");
	variable(&T[36], N = 0, C = 0, H = 1, W = 288, "Const_94");
	variable(&T[37], N = 0, C = 0, H = 1, W = 528, "Const_97");
	variable(&T[38], N = 144, C = 1, H = 3, W = 3, "Const_99");
	variable(&T[39], N = 288, C = 1, H = 3, W = 3, "Const_102");
	variable(&T[40], N = 0, C = 0, H = 1, W = 144, "Const_103");
	variable(&T[41], N = 0, C = 0, H = 1, W = 816, "Const_107");
	variable(&T[42], N = 0, C = 0, H = 1, W = 192, "Const_111");
	variable(&T[43], N = 0, C = 0, H = 1, W = 288, "Const_116");
	variable(&T[44], N = 0, C = 0, H = 1, W = 144, "Const_118");
	variable(&T[45], N = 192, C = 1, H = 3, W = 3, "Const_119");
	variable(&T[46], N = 0, C = 0, H = 1, W = 48, "Const_120");
	variable(&T[47], N = 528, C = 1, H = 3, W = 3, "Const_129");
	variable(&T[48], N = 1344, C = 1, H = 3, W = 3, "Const_130");
	variable(&T[49], N = 0, C = 0, H = 1, W = 528, "Const_131");
	variable(&T[50], N = 0, C = 0, H = 1, W = 48, "Const_138");
	variable(&T[51], N = 528, C = 1, H = 3, W = 3, "Const_142");
	variable(&T[52], N = 0, C = 0, H = 1, W = 816, "Const_144");
	variable(&T[53], N = 0, C = 0, H = 1, W = 288, "Const_145");
	variable(&T[54], N = 0, C = 0, H = 1, W = 288, "Const_147");
	variable(&T[55], N = 0, C = 0, H = 1, W = 1344, "Const_151");
	variable(&T[56], N = 0, C = 0, H = 1, W = 1344, "Const_152");
	variable(&T[57], N = 0, C = 0, H = 1, W = 816, "Const_154");
	variable(&T[58], N = 288, C = 1, H = 3, W = 3, "Const_163");
	variable(&T[59], N = 0, C = 0, H = 1, W = 288, "Const_166");
	variable(&T[60], N = 0, C = 0, H = 1, W = 528, "Const_171");
	variable(&T[61], N = 0, C = 0, H = 1, W = 528, "Const_173");
	variable(&T[62], N = 0, C = 0, H = 1, W = 144, "Const_178");
	variable(&T[63], N = 0, C = 0, H = 1, W = 528, "Const_182");
	variable(&T[64], N = 0, C = 0, H = 1, W = 528, "Const_183");
	variable(&T[65], N = 0, C = 0, H = 1, W = 528, "Const_194");
	variable(&T[66], N = 0, C = 0, H = 1, W = 1344, "Const_195");
	variable(&T[67], N = 0, C = 0, H = 1, W = 48, "Const_198");
	variable(&T[68], N = 0, C = 0, H = 1, W = 1344, "Const_204");
	variable(&T[69], N = 0, C = 0, H = 1, W = 192, "Const_207");
	variable(&T[70], N = 0, C = 0, H = 1, W = 528, "Const_209");
	variable(&T[71], N = 0, C = 0, H = 1, W = 816, "Const_211");
	variable(&T[72], N = 192, C = 1, H = 3, W = 3, "Const_216");
	variable(&T[73], N = 0, C = 0, H = 1, W = 1344, "Const_217");
	variable(&T[74], N = 0, C = 0, H = 1, W = 288, "Const_219");
	variable(&T[75], N = 288, C = 1, H = 3, W = 3, "Const_221");
	variable(&T[76], N = 1001, C = 1792, H = 1, W = 1, "Const_222");
	variable(&T[77], N = 0, C = 0, H = 1, W = 288, "Const_223");
	variable(&T[78], N = 0, C = 0, H = 1, W = 1344, "Const_228");
	variable(&T[79], N = 0, C = 0, H = 1, W = 816, "Const_237");
	variable(&T[80], N = 48, C = 1, H = 3, W = 3, "Const_238");
	variable(&T[81], N = 0, C = 0, H = 1, W = 192, "Const_245");
	variable(&T[82], N = 1344, C = 1, H = 3, W = 3, "Const_248");
	variable(&T[83], N = 0, C = 0, H = 1, W = 816, "Const_249");
	variable(&T[84], N = 1344, C = 1, H = 3, W = 3, "Const_250");
	variable(&T[85], N = 0, C = 0, H = 1, W = 288, "Const_252");
	variable(&T[86], N = 0, C = 0, H = 1, W = 192, "Const_253");
	variable(&T[87], N = 528, C = 1, H = 3, W = 3, "Const_254");
	variable(&T[88], N = 0, C = 0, H = 1, W = 1344, "Const_259");
	external(&T[0], N = 1, C = 3, H = 224, W = 224);
	variable(&T[89], N = 48, C = 3, H = 3, W = 3, "Const_90");
	variable(&T[91], N = 0, C = 0, H = 1, W = 48, "MobilenetV2/Conv/Conv2D_bn_offset");
	variable(&T[99], N = 24, C = 48, H = 1, W = 1, "Const_167");
	variable(&T[101], N = 0, C = 0, H = 1, W = 24, "MobilenetV2/expanded_conv/project/Conv2D_bn_offset");
	variable(&T[103], N = 144, C = 24, H = 1, W = 1, "Const_165");
	variable(&T[105], N = 0, C = 0, H = 1, W = 144, "MobilenetV2/expanded_conv_1/expand/Conv2D_bn_offset");
	variable(&T[113], N = 32, C = 144, H = 1, W = 1, "Const_73");
	variable(&T[115], N = 0, C = 0, H = 1, W = 32, "MobilenetV2/expanded_conv_1/project/Conv2D_bn_offset");
	variable(&T[117], N = 192, C = 32, H = 1, W = 1, "Const_3");
	variable(&T[119], N = 0, C = 0, H = 1, W = 192, "MobilenetV2/expanded_conv_2/expand/Conv2D_bn_offset");
	variable(&T[127], N = 32, C = 192, H = 1, W = 1, "Const_115");
	variable(&T[129], N = 0, C = 0, H = 1, W = 32, "MobilenetV2/expanded_conv_2/project/Conv2D_bn_offset");
	variable(&T[132], N = 192, C = 32, H = 1, W = 1, "Const_255");
	variable(&T[134], N = 0, C = 0, H = 1, W = 192, "MobilenetV2/expanded_conv_3/expand/Conv2D_bn_offset");
	variable(&T[142], N = 48, C = 192, H = 1, W = 1, "Const_157");
	variable(&T[144], N = 0, C = 0, H = 1, W = 48, "MobilenetV2/expanded_conv_3/project/Conv2D_bn_offset");
	variable(&T[146], N = 288, C = 48, H = 1, W = 1, "Const_227");
	variable(&T[148], N = 0, C = 0, H = 1, W = 288, "MobilenetV2/expanded_conv_4/expand/Conv2D_bn_offset");
	variable(&T[156], N = 48, C = 288, H = 1, W = 1, "Const_193");
	variable(&T[158], N = 0, C = 0, H = 1, W = 48, "MobilenetV2/expanded_conv_4/project/Conv2D_bn_offset");
	variable(&T[161], N = 288, C = 48, H = 1, W = 1, "Const_243");
	variable(&T[163], N = 0, C = 0, H = 1, W = 288, "MobilenetV2/expanded_conv_5/expand/Conv2D_bn_offset");
	variable(&T[171], N = 48, C = 288, H = 1, W = 1, "Const_215");
	variable(&T[173], N = 0, C = 0, H = 1, W = 48, "MobilenetV2/expanded_conv_5/project/Conv2D_bn_offset");
	variable(&T[176], N = 288, C = 48, H = 1, W = 1, "Const_226");
	variable(&T[178], N = 0, C = 0, H = 1, W = 288, "MobilenetV2/expanded_conv_6/expand/Conv2D_bn_offset");
	variable(&T[186], N = 88, C = 288, H = 1, W = 1, "Const_229");
	variable(&T[188], N = 0, C = 0, H = 1, W = 88, "MobilenetV2/expanded_conv_6/project/Conv2D_bn_offset");
	variable(&T[190], N = 528, C = 88, H = 1, W = 1, "Const_104");
	variable(&T[192], N = 0, C = 0, H = 1, W = 528, "MobilenetV2/expanded_conv_7/expand/Conv2D_bn_offset");
	variable(&T[200], N = 88, C = 528, H = 1, W = 1, "Const_143");
	variable(&T[202], N = 0, C = 0, H = 1, W = 88, "MobilenetV2/expanded_conv_7/project/Conv2D_bn_offset");
	variable(&T[205], N = 528, C = 88, H = 1, W = 1, "Const_25");
	variable(&T[207], N = 0, C = 0, H = 1, W = 528, "MobilenetV2/expanded_conv_8/expand/Conv2D_bn_offset");
	variable(&T[215], N = 88, C = 528, H = 1, W = 1, "Const_202");
	variable(&T[217], N = 0, C = 0, H = 1, W = 88, "MobilenetV2/expanded_conv_8/project/Conv2D_bn_offset");
	variable(&T[220], N = 528, C = 88, H = 1, W = 1, "Const_225");
	variable(&T[222], N = 0, C = 0, H = 1, W = 528, "MobilenetV2/expanded_conv_9/expand/Conv2D_bn_offset");
	variable(&T[230], N = 88, C = 528, H = 1, W = 1, "Const_98");
	variable(&T[232], N = 0, C = 0, H = 1, W = 88, "MobilenetV2/expanded_conv_9/project/Conv2D_bn_offset");
	variable(&T[235], N = 528, C = 88, H = 1, W = 1, "Const_169");
	variable(&T[237], N = 0, C = 0, H = 1, W = 528, "MobilenetV2/expanded_conv_10/expand/Conv2D_bn_offset");
	variable(&T[245], N = 136, C = 528, H = 1, W = 1, "Const_246");
	variable(&T[247], N = 0, C = 0, H = 1, W = 136, "MobilenetV2/expanded_conv_10/project/Conv2D_bn_offset");
	variable(&T[249], N = 816, C = 136, H = 1, W = 1, "Const_162");
	variable(&T[251], N = 0, C = 0, H = 1, W = 816, "MobilenetV2/expanded_conv_11/expand/Conv2D_bn_offset");
	variable(&T[259], N = 136, C = 816, H = 1, W = 1, "Const_106");
	variable(&T[261], N = 0, C = 0, H = 1, W = 136, "MobilenetV2/expanded_conv_11/project/Conv2D_bn_offset");
	variable(&T[264], N = 816, C = 136, H = 1, W = 1, "Const_52");
	variable(&T[266], N = 0, C = 0, H = 1, W = 816, "MobilenetV2/expanded_conv_12/expand/Conv2D_bn_offset");
	variable(&T[274], N = 136, C = 816, H = 1, W = 1, "Const_40");
	variable(&T[276], N = 0, C = 0, H = 1, W = 136, "MobilenetV2/expanded_conv_12/project/Conv2D_bn_offset");
	variable(&T[279], N = 816, C = 136, H = 1, W = 1, "Const_114");
	variable(&T[281], N = 0, C = 0, H = 1, W = 816, "MobilenetV2/expanded_conv_13/expand/Conv2D_bn_offset");
	variable(&T[289], N = 224, C = 816, H = 1, W = 1, "Const_242");
	variable(&T[291], N = 0, C = 0, H = 1, W = 224, "MobilenetV2/expanded_conv_13/project/Conv2D_bn_offset");
	variable(&T[293], N = 1344, C = 224, H = 1, W = 1, "Const_203");
	variable(&T[295], N = 0, C = 0, H = 1, W = 1344, "MobilenetV2/expanded_conv_14/expand/Conv2D_bn_offset");
	variable(&T[303], N = 224, C = 1344, H = 1, W = 1, "Const_92");
	variable(&T[305], N = 0, C = 0, H = 1, W = 224, "MobilenetV2/expanded_conv_14/project/Conv2D_bn_offset");
	variable(&T[308], N = 1344, C = 224, H = 1, W = 1, "Const_133");
	variable(&T[310], N = 0, C = 0, H = 1, W = 1344, "MobilenetV2/expanded_conv_15/expand/Conv2D_bn_offset");
	variable(&T[318], N = 224, C = 1344, H = 1, W = 1, "Const_258");
	variable(&T[320], N = 0, C = 0, H = 1, W = 224, "MobilenetV2/expanded_conv_15/project/Conv2D_bn_offset");
	variable(&T[323], N = 1344, C = 224, H = 1, W = 1, "Const_60");
	variable(&T[325], N = 0, C = 0, H = 1, W = 1344, "MobilenetV2/expanded_conv_16/expand/Conv2D_bn_offset");
	variable(&T[333], N = 448, C = 1344, H = 1, W = 1, "Const_100");
	variable(&T[335], N = 0, C = 0, H = 1, W = 448, "MobilenetV2/expanded_conv_16/project/Conv2D_bn_offset");
	variable(&T[337], N = 1792, C = 448, H = 1, W = 1, "Const_71");
	variable(&T[339], N = 0, C = 0, H = 1, W = 1792, "MobilenetV2/Conv_1/Conv2D_bn_offset");
}

void kernel(tensor * T)
{
	convxbias(&T[90], &T[0], &T[89], bias = 0.0, padding = 1, stride = 2, groups = 1);
	add(&T[92], &T[90], &T[91]);
	relu(&T[93], &T[92]);
	min(&T[94], &T[93], y = 6.0);
	convxbias(&T[95], &T[94], &T[80], bias = 0.0, padding = 1, stride = 1, groups = 48);
	batch_normalization(&T[96], &T[95], &T[50], &T[46], &T[17], &T[67], epsilon = 0.001);
	relu(&T[97], &T[96]);
	min(&T[98], &T[97], y = 6.0);
	convxbias(&T[100], &T[98], &T[99], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[102], &T[100], &T[101]);
	convxbias(&T[104], &T[102], &T[103], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[106], &T[104], &T[105]);
	relu(&T[107], &T[106]);
	min(&T[108], &T[107], y = 6.0);
	convxbias(&T[109], &T[108], &T[38], bias = 0.0, padding = 1, stride = 2, groups = 144);
	batch_normalization(&T[110], &T[109], &T[62], &T[44], &T[32], &T[40], epsilon = 0.001);
	relu(&T[111], &T[110]);
	min(&T[112], &T[111], y = 6.0);
	convxbias(&T[114], &T[112], &T[113], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[116], &T[114], &T[115]);
	convxbias(&T[118], &T[116], &T[117], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[120], &T[118], &T[119]);
	relu(&T[121], &T[120]);
	min(&T[122], &T[121], y = 6.0);
	convxbias(&T[123], &T[122], &T[45], bias = 0.0, padding = 1, stride = 1, groups = 192);
	batch_normalization(&T[124], &T[123], &T[35], &T[27], &T[81], &T[12], epsilon = 0.001);
	relu(&T[125], &T[124]);
	min(&T[126], &T[125], y = 6.0);
	convxbias(&T[128], &T[126], &T[127], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[130], &T[128], &T[129]);
	add(&T[131], &T[130], &T[116]);
	convxbias(&T[133], &T[131], &T[132], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[135], &T[133], &T[134]);
	relu(&T[136], &T[135]);
	min(&T[137], &T[136], y = 6.0);
	convxbias(&T[138], &T[137], &T[72], bias = 0.0, padding = 1, stride = 2, groups = 192);
	batch_normalization(&T[139], &T[138], &T[11], &T[69], &T[42], &T[86], epsilon = 0.001);
	relu(&T[140], &T[139]);
	min(&T[141], &T[140], y = 6.0);
	convxbias(&T[143], &T[141], &T[142], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[145], &T[143], &T[144]);
	convxbias(&T[147], &T[145], &T[146], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[149], &T[147], &T[148]);
	relu(&T[150], &T[149]);
	min(&T[151], &T[150], y = 6.0);
	convxbias(&T[152], &T[151], &T[75], bias = 0.0, padding = 1, stride = 1, groups = 288);
	batch_normalization(&T[153], &T[152], &T[53], &T[54], &T[74], &T[2], epsilon = 0.001);
	relu(&T[154], &T[153]);
	min(&T[155], &T[154], y = 6.0);
	convxbias(&T[157], &T[155], &T[156], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[159], &T[157], &T[158]);
	add(&T[160], &T[159], &T[145]);
	convxbias(&T[162], &T[160], &T[161], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[164], &T[162], &T[163]);
	relu(&T[165], &T[164]);
	min(&T[166], &T[165], y = 6.0);
	convxbias(&T[167], &T[166], &T[39], bias = 0.0, padding = 1, stride = 1, groups = 288);
	batch_normalization(&T[168], &T[167], &T[31], &T[43], &T[85], &T[36], epsilon = 0.001);
	relu(&T[169], &T[168]);
	min(&T[170], &T[169], y = 6.0);
	convxbias(&T[172], &T[170], &T[171], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[174], &T[172], &T[173]);
	add(&T[175], &T[174], &T[160]);
	convxbias(&T[177], &T[175], &T[176], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[179], &T[177], &T[178]);
	relu(&T[180], &T[179]);
	min(&T[181], &T[180], y = 6.0);
	convxbias(&T[182], &T[181], &T[58], bias = 0.0, padding = 1, stride = 2, groups = 288);
	batch_normalization(&T[183], &T[182], &T[59], &T[20], &T[15], &T[77], epsilon = 0.001);
	relu(&T[184], &T[183]);
	min(&T[185], &T[184], y = 6.0);
	convxbias(&T[187], &T[185], &T[186], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[189], &T[187], &T[188]);
	convxbias(&T[191], &T[189], &T[190], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[193], &T[191], &T[192]);
	relu(&T[194], &T[193]);
	min(&T[195], &T[194], y = 6.0);
	convxbias(&T[196], &T[195], &T[87], bias = 0.0, padding = 1, stride = 1, groups = 528);
	batch_normalization(&T[197], &T[196], &T[49], &T[14], &T[60], &T[19], epsilon = 0.001);
	relu(&T[198], &T[197]);
	min(&T[199], &T[198], y = 6.0);
	convxbias(&T[201], &T[199], &T[200], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[203], &T[201], &T[202]);
	add(&T[204], &T[203], &T[189]);
	convxbias(&T[206], &T[204], &T[205], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[208], &T[206], &T[207]);
	relu(&T[209], &T[208]);
	min(&T[210], &T[209], y = 6.0);
	convxbias(&T[211], &T[210], &T[51], bias = 0.0, padding = 1, stride = 1, groups = 528);
	batch_normalization(&T[212], &T[211], &T[70], &T[13], &T[63], &T[25], epsilon = 0.001);
	relu(&T[213], &T[212]);
	min(&T[214], &T[213], y = 6.0);
	convxbias(&T[216], &T[214], &T[215], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[218], &T[216], &T[217]);
	add(&T[219], &T[218], &T[204]);
	convxbias(&T[221], &T[219], &T[220], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[223], &T[221], &T[222]);
	relu(&T[224], &T[223]);
	min(&T[225], &T[224], y = 6.0);
	convxbias(&T[226], &T[225], &T[47], bias = 0.0, padding = 1, stride = 1, groups = 528);
	batch_normalization(&T[227], &T[226], &T[65], &T[64], &T[16], &T[21], epsilon = 0.001);
	relu(&T[228], &T[227]);
	min(&T[229], &T[228], y = 6.0);
	convxbias(&T[231], &T[229], &T[230], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[233], &T[231], &T[232]);
	add(&T[234], &T[233], &T[219]);
	convxbias(&T[236], &T[234], &T[235], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[238], &T[236], &T[237]);
	relu(&T[239], &T[238]);
	min(&T[240], &T[239], y = 6.0);
	convxbias(&T[241], &T[240], &T[3], bias = 0.0, padding = 1, stride = 1, groups = 528);
	batch_normalization(&T[242], &T[241], &T[8], &T[4], &T[61], &T[37], epsilon = 0.001);
	relu(&T[243], &T[242]);
	min(&T[244], &T[243], y = 6.0);
	convxbias(&T[246], &T[244], &T[245], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[248], &T[246], &T[247]);
	convxbias(&T[250], &T[248], &T[249], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[252], &T[250], &T[251]);
	relu(&T[253], &T[252]);
	min(&T[254], &T[253], y = 6.0);
	convxbias(&T[255], &T[254], &T[34], bias = 0.0, padding = 1, stride = 1, groups = 816);
	batch_normalization(&T[256], &T[255], &T[29], &T[23], &T[41], &T[52], epsilon = 0.001);
	relu(&T[257], &T[256]);
	min(&T[258], &T[257], y = 6.0);
	convxbias(&T[260], &T[258], &T[259], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[262], &T[260], &T[261]);
	add(&T[263], &T[262], &T[248]);
	convxbias(&T[265], &T[263], &T[264], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[267], &T[265], &T[266]);
	relu(&T[268], &T[267]);
	min(&T[269], &T[268], y = 6.0);
	convxbias(&T[270], &T[269], &T[10], bias = 0.0, padding = 1, stride = 1, groups = 816);
	batch_normalization(&T[271], &T[270], &T[33], &T[7], &T[24], &T[71], epsilon = 0.001);
	relu(&T[272], &T[271]);
	min(&T[273], &T[272], y = 6.0);
	convxbias(&T[275], &T[273], &T[274], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[277], &T[275], &T[276]);
	add(&T[278], &T[277], &T[263]);
	convxbias(&T[280], &T[278], &T[279], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[282], &T[280], &T[281]);
	relu(&T[283], &T[282]);
	min(&T[284], &T[283], y = 6.0);
	convxbias(&T[285], &T[284], &T[26], bias = 0.0, padding = 1, stride = 2, groups = 816);
	batch_normalization(&T[286], &T[285], &T[83], &T[79], &T[5], &T[57], epsilon = 0.001);
	relu(&T[287], &T[286]);
	min(&T[288], &T[287], y = 6.0);
	convxbias(&T[290], &T[288], &T[289], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[292], &T[290], &T[291]);
	convxbias(&T[294], &T[292], &T[293], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[296], &T[294], &T[295]);
	relu(&T[297], &T[296]);
	min(&T[298], &T[297], y = 6.0);
	convxbias(&T[299], &T[298], &T[84], bias = 0.0, padding = 1, stride = 1, groups = 1344);
	batch_normalization(&T[300], &T[299], &T[88], &T[78], &T[28], &T[6], epsilon = 0.001);
	relu(&T[301], &T[300]);
	min(&T[302], &T[301], y = 6.0);
	convxbias(&T[304], &T[302], &T[303], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[306], &T[304], &T[305]);
	add(&T[307], &T[306], &T[292]);
	convxbias(&T[309], &T[307], &T[308], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[311], &T[309], &T[310]);
	relu(&T[312], &T[311]);
	min(&T[313], &T[312], y = 6.0);
	convxbias(&T[314], &T[313], &T[48], bias = 0.0, padding = 1, stride = 1, groups = 1344);
	batch_normalization(&T[315], &T[314], &T[55], &T[9], &T[66], &T[56], epsilon = 0.001);
	relu(&T[316], &T[315]);
	min(&T[317], &T[316], y = 6.0);
	convxbias(&T[319], &T[317], &T[318], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[321], &T[319], &T[320]);
	add(&T[322], &T[321], &T[307]);
	convxbias(&T[324], &T[322], &T[323], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[326], &T[324], &T[325]);
	relu(&T[327], &T[326]);
	min(&T[328], &T[327], y = 6.0);
	convxbias(&T[329], &T[328], &T[82], bias = 0.0, padding = 1, stride = 1, groups = 1344);
	batch_normalization(&T[330], &T[329], &T[18], &T[73], &T[22], &T[68], epsilon = 0.001);
	relu(&T[331], &T[330]);
	min(&T[332], &T[331], y = 6.0);
	convxbias(&T[334], &T[332], &T[333], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[336], &T[334], &T[335]);
	convxbias(&T[338], &T[336], &T[337], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[340], &T[338], &T[339]);
	relu(&T[341], &T[340]);
	min(&T[342], &T[341], y = 6.0);
	avg_pool(&T[343], &T[342], size = 7, padding = 0, stride = 1);
	convxbias(&T[344], &T[343], &T[76], bias = 0.0, padding = 1, stride = 1, groups = 1);
	add(&T[345], &T[344], &T[30]);
	reshape(&T[346], &T[345], N = 0, C = 0, H = 1, W = 1001);
	reshape(&T[347], &T[346], N = 0, C = 0, H = 1, W = 1001);
	softmax(&T[348], &T[347]);
	reshape(&T[1], &T[348], N = 0, C = 0, H = 1, W = 1001);
}
*/

// #########################################################
// # (relu, min6) -> (relu6) // merge
// # (batch_normalization) -> (sqrt, sub, div, mul, add) // separate
// # -------------------------------------------------------

void init(tensor * T)
{
	external(&T[0], 1, 3, 224, 224);  /* [1, 3, 224, 224], layer : 0,  */ 
	variable(&T[2], 32, 3, 3, 3, "MobilenetV1/Conv2d_0/weights");  /* [32, 3, 3, 3], layer : 0,  */ 
	variable(&T[11], 32, 1, 3, 3, "MobilenetV1/Conv2d_1_depthwise/depthwise_weights");  /* [32, 1, 3, 3], layer : 0,  */ 
	variable(&T[101], 256, 1, 3, 3, "MobilenetV1/Conv2d_6_depthwise/depthwise_weights");  /* [256, 1, 3, 3], layer : 0,  */ 
	variable(&T[103], 0, 0, 1, 256, "MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma");  /* [1, 256], layer : 0,  */ 
	variable(&T[104], 0, 0, 1, 256, "MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta");  /* [1, 256], layer : 0,  */ 
	variable(&T[105], 0, 0, 1, 256, "MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean");  /* [1, 256], layer : 0,  */ 
	variable(&T[106], 0, 0, 1, 256, "MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance");  /* [1, 256], layer : 0,  */ 
	variable(&T[110], 512, 256, 1, 1, "MobilenetV1/Conv2d_6_pointwise/weights");  /* [512, 256, 1, 1], layer : 0,  */ 
	variable(&T[112], 0, 0, 1, 512, "MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[113], 0, 0, 1, 512, "MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[114], 0, 0, 1, 512, "MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[115], 0, 0, 1, 512, "MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[119], 512, 1, 3, 3, "MobilenetV1/Conv2d_7_depthwise/depthwise_weights");  /* [512, 1, 3, 3], layer : 0,  */ 
	variable(&T[13], 0, 0, 1, 32, "MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma");  /* [1, 32], layer : 0,  */ 
	variable(&T[121], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[122], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[123], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[124], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[128], 512, 512, 1, 1, "MobilenetV1/Conv2d_7_pointwise/weights");  /* [512, 512, 1, 1], layer : 0,  */ 
	variable(&T[130], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[14], 0, 0, 1, 32, "MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta");  /* [1, 32], layer : 0,  */ 
	variable(&T[131], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[132], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[133], 0, 0, 1, 512, "MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[137], 512, 1, 3, 3, "MobilenetV1/Conv2d_8_depthwise/depthwise_weights");  /* [512, 1, 3, 3], layer : 0,  */ 
	variable(&T[139], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[140], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[15], 0, 0, 1, 32, "MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean");  /* [1, 32], layer : 0,  */ 
	variable(&T[141], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[142], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[146], 512, 512, 1, 1, "MobilenetV1/Conv2d_8_pointwise/weights");  /* [512, 512, 1, 1], layer : 0,  */ 
	variable(&T[148], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[149], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[150], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[16], 0, 0, 1, 32, "MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance");  /* [1, 32], layer : 0,  */ 
	variable(&T[151], 0, 0, 1, 512, "MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[155], 512, 1, 3, 3, "MobilenetV1/Conv2d_9_depthwise/depthwise_weights");  /* [512, 1, 3, 3], layer : 0,  */ 
	variable(&T[157], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[158], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[159], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[160], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[164], 512, 512, 1, 1, "MobilenetV1/Conv2d_9_pointwise/weights");  /* [512, 512, 1, 1], layer : 0,  */ 
	variable(&T[166], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[167], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[168], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[169], 0, 0, 1, 512, "MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[173], 512, 1, 3, 3, "MobilenetV1/Conv2d_10_depthwise/depthwise_weights");  /* [512, 1, 3, 3], layer : 0,  */ 
	variable(&T[175], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[176], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[177], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[178], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[182], 512, 512, 1, 1, "MobilenetV1/Conv2d_10_pointwise/weights");  /* [512, 512, 1, 1], layer : 0,  */ 
	variable(&T[184], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[185], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[186], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[187], 0, 0, 1, 512, "MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[20], 64, 32, 1, 1, "MobilenetV1/Conv2d_1_pointwise/weights");  /* [64, 32, 1, 1], layer : 0,  */ 
	variable(&T[191], 512, 1, 3, 3, "MobilenetV1/Conv2d_11_depthwise/depthwise_weights");  /* [512, 1, 3, 3], layer : 0,  */ 
	variable(&T[193], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[194], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[195], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[196], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[200], 512, 512, 1, 1, "MobilenetV1/Conv2d_11_pointwise/weights");  /* [512, 512, 1, 1], layer : 0,  */ 
	variable(&T[202], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[203], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[204], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[205], 0, 0, 1, 512, "MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[209], 512, 1, 3, 3, "MobilenetV1/Conv2d_12_depthwise/depthwise_weights");  /* [512, 1, 3, 3], layer : 0,  */ 
	variable(&T[22], 0, 0, 1, 64, "MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma");  /* [1, 64], layer : 0,  */ 
	variable(&T[211], 0, 0, 1, 512, "MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma");  /* [1, 512], layer : 0,  */ 
	variable(&T[212], 0, 0, 1, 512, "MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta");  /* [1, 512], layer : 0,  */ 
	variable(&T[213], 0, 0, 1, 512, "MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean");  /* [1, 512], layer : 0,  */ 
	variable(&T[214], 0, 0, 1, 512, "MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance");  /* [1, 512], layer : 0,  */ 
	variable(&T[218], 1024, 512, 1, 1, "MobilenetV1/Conv2d_12_pointwise/weights");  /* [1024, 512, 1, 1], layer : 0,  */ 
	variable(&T[220], 0, 0, 1, 1024, "MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma");  /* [1, 1024], layer : 0,  */ 
	variable(&T[23], 0, 0, 1, 64, "MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta");  /* [1, 64], layer : 0,  */ 
	variable(&T[221], 0, 0, 1, 1024, "MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta");  /* [1, 1024], layer : 0,  */ 
	variable(&T[222], 0, 0, 1, 1024, "MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean");  /* [1, 1024], layer : 0,  */ 
	variable(&T[223], 0, 0, 1, 1024, "MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance");  /* [1, 1024], layer : 0,  */ 
	variable(&T[227], 1024, 1, 3, 3, "MobilenetV1/Conv2d_13_depthwise/depthwise_weights");  /* [1024, 1, 3, 3], layer : 0,  */ 
	variable(&T[229], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma");  /* [1, 1024], layer : 0,  */ 
	variable(&T[230], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta");  /* [1, 1024], layer : 0,  */ 
	variable(&T[24], 0, 0, 1, 64, "MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean");  /* [1, 64], layer : 0,  */ 
	variable(&T[231], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean");  /* [1, 1024], layer : 0,  */ 
	variable(&T[232], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance");  /* [1, 1024], layer : 0,  */ 
	variable(&T[236], 1024, 1024, 1, 1, "MobilenetV1/Conv2d_13_pointwise/weights");  /* [1024, 1024, 1, 1], layer : 0,  */ 
	variable(&T[238], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma");  /* [1, 1024], layer : 0,  */ 
	variable(&T[239], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta");  /* [1, 1024], layer : 0,  */ 
	variable(&T[240], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean");  /* [1, 1024], layer : 0,  */ 
	variable(&T[25], 0, 0, 1, 64, "MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance");  /* [1, 64], layer : 0,  */ 
	variable(&T[241], 0, 0, 1, 1024, "MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance");  /* [1, 1024], layer : 0,  */ 
	variable(&T[246], 1001, 1024, 1, 1, "MobilenetV1/Logits/Conv2d_1c_1x1/weights");  /* [1001, 1024, 1, 1], layer : 0,  */ 
	variable(&T[247], 0, 0, 1, 1001, "MobilenetV1/Logits/Conv2d_1c_1x1/biases");  /* [1, 1001], layer : 0,  */ 
	variable(&T[29], 64, 1, 3, 3, "MobilenetV1/Conv2d_2_depthwise/depthwise_weights");  /* [64, 1, 3, 3], layer : 0,  */ 
	variable(&T[4], 0, 0, 1, 32, "MobilenetV1/Conv2d_0/BatchNorm/gamma");  /* [1, 32], layer : 0,  */ 
	variable(&T[31], 0, 0, 1, 64, "MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma");  /* [1, 64], layer : 0,  */ 
	variable(&T[32], 0, 0, 1, 64, "MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta");  /* [1, 64], layer : 0,  */ 
	variable(&T[33], 0, 0, 1, 64, "MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean");  /* [1, 64], layer : 0,  */ 
	variable(&T[34], 0, 0, 1, 64, "MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance");  /* [1, 64], layer : 0,  */ 
	variable(&T[38], 128, 64, 1, 1, "MobilenetV1/Conv2d_2_pointwise/weights");  /* [128, 64, 1, 1], layer : 0,  */ 
	variable(&T[40], 0, 0, 1, 128, "MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma");  /* [1, 128], layer : 0,  */ 
	variable(&T[5], 0, 0, 1, 32, "MobilenetV1/Conv2d_0/BatchNorm/beta");  /* [1, 32], layer : 0,  */ 
	variable(&T[41], 0, 0, 1, 128, "MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta");  /* [1, 128], layer : 0,  */ 
	variable(&T[42], 0, 0, 1, 128, "MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean");  /* [1, 128], layer : 0,  */ 
	variable(&T[43], 0, 0, 1, 128, "MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance");  /* [1, 128], layer : 0,  */ 
	variable(&T[47], 128, 1, 3, 3, "MobilenetV1/Conv2d_3_depthwise/depthwise_weights");  /* [128, 1, 3, 3], layer : 0,  */ 
	variable(&T[49], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma");  /* [1, 128], layer : 0,  */ 
	variable(&T[50], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta");  /* [1, 128], layer : 0,  */ 
	variable(&T[6], 0, 0, 1, 32, "MobilenetV1/Conv2d_0/BatchNorm/moving_mean");  /* [1, 32], layer : 0,  */ 
	variable(&T[51], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean");  /* [1, 128], layer : 0,  */ 
	variable(&T[52], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance");  /* [1, 128], layer : 0,  */ 
	variable(&T[56], 128, 128, 1, 1, "MobilenetV1/Conv2d_3_pointwise/weights");  /* [128, 128, 1, 1], layer : 0,  */ 
	variable(&T[58], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma");  /* [1, 128], layer : 0,  */ 
	variable(&T[59], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta");  /* [1, 128], layer : 0,  */ 
	variable(&T[60], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean");  /* [1, 128], layer : 0,  */ 
	variable(&T[7], 0, 0, 1, 32, "MobilenetV1/Conv2d_0/BatchNorm/moving_variance");  /* [1, 32], layer : 0,  */ 
	variable(&T[61], 0, 0, 1, 128, "MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance");  /* [1, 128], layer : 0,  */ 
	variable(&T[65], 128, 1, 3, 3, "MobilenetV1/Conv2d_4_depthwise/depthwise_weights");  /* [128, 1, 3, 3], layer : 0,  */ 
	variable(&T[67], 0, 0, 1, 128, "MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma");  /* [1, 128], layer : 0,  */ 
	variable(&T[68], 0, 0, 1, 128, "MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta");  /* [1, 128], layer : 0,  */ 
	variable(&T[69], 0, 0, 1, 128, "MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean");  /* [1, 128], layer : 0,  */ 
	variable(&T[70], 0, 0, 1, 128, "MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance");  /* [1, 128], layer : 0,  */ 
	variable(&T[74], 256, 128, 1, 1, "MobilenetV1/Conv2d_4_pointwise/weights");  /* [256, 128, 1, 1], layer : 0,  */ 
	variable(&T[76], 0, 0, 1, 256, "MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma");  /* [1, 256], layer : 0,  */ 
	variable(&T[77], 0, 0, 1, 256, "MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta");  /* [1, 256], layer : 0,  */ 
	variable(&T[78], 0, 0, 1, 256, "MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean");  /* [1, 256], layer : 0,  */ 
	variable(&T[79], 0, 0, 1, 256, "MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance");  /* [1, 256], layer : 0,  */ 
	variable(&T[83], 256, 1, 3, 3, "MobilenetV1/Conv2d_5_depthwise/depthwise_weights");  /* [256, 1, 3, 3], layer : 0,  */ 
	variable(&T[85], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma");  /* [1, 256], layer : 0,  */ 
	variable(&T[86], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta");  /* [1, 256], layer : 0,  */ 
	variable(&T[87], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean");  /* [1, 256], layer : 0,  */ 
	variable(&T[88], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance");  /* [1, 256], layer : 0,  */ 
	variable(&T[92], 256, 256, 1, 1, "MobilenetV1/Conv2d_5_pointwise/weights");  /* [256, 256, 1, 1], layer : 0,  */ 
	variable(&T[94], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma");  /* [1, 256], layer : 0,  */ 
	variable(&T[95], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta");  /* [1, 256], layer : 0,  */ 
	variable(&T[96], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean");  /* [1, 256], layer : 0,  */ 
	variable(&T[97], 0, 0, 1, 256, "MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance");  /* [1, 256], layer : 0,  */ 
	bn_sqrt(&T[253], &T[7], 0.001000);
	bn_sqrt(&T[257], &T[16], 0.001000);
	bn_sqrt(&T[261], &T[25], 0.001000);
	bn_sqrt(&T[265], &T[34], 0.001000);
	bn_sqrt(&T[269], &T[43], 0.001000);
	bn_sqrt(&T[273], &T[52], 0.001000);
	bn_sqrt(&T[277], &T[61], 0.001000);
	bn_sqrt(&T[281], &T[70], 0.001000);
	bn_sqrt(&T[285], &T[79], 0.001000);
	bn_sqrt(&T[289], &T[88], 0.001000);
	bn_sqrt(&T[293], &T[97], 0.001000);
	bn_sqrt(&T[297], &T[106], 0.001000);
	bn_sqrt(&T[301], &T[115], 0.001000);
	bn_sqrt(&T[305], &T[124], 0.001000);
	bn_sqrt(&T[309], &T[133], 0.001000);
	bn_sqrt(&T[313], &T[142], 0.001000);
	bn_sqrt(&T[317], &T[151], 0.001000);
	bn_sqrt(&T[321], &T[160], 0.001000);
	bn_sqrt(&T[325], &T[169], 0.001000);
	bn_sqrt(&T[329], &T[178], 0.001000);
	bn_sqrt(&T[333], &T[187], 0.001000);
	bn_sqrt(&T[337], &T[196], 0.001000);
	bn_sqrt(&T[341], &T[205], 0.001000);
	bn_sqrt(&T[345], &T[214], 0.001000);
	bn_sqrt(&T[349], &T[223], 0.001000);
	bn_sqrt(&T[353], &T[232], 0.001000);
	bn_sqrt(&T[357], &T[241], 0.001000);
}

void kernel(tensor * T)
{
	convxbias(&T[3], &T[0], &T[2], tensor_n = 0.000000, padding = 1, stride = 2, groups = 1);  /* [1, 32, 112, 112], layer : 1,  */ 
	bn_sub(&T[254], &T[3], &T[6]);
	bn_div(&T[255], &T[254], &T[253]);
	bn_mul(&T[256], &T[255], &T[4]);
	bn_add(&T[8], &T[256], &T[5]); /* [1, 32, 112, 112], layer : 2,  */ 
	relu6(&T[10], &T[8]);  /* [1, 32, 112, 112], layer : 3,  */ 
	convxbias(&T[12], &T[10], &T[11], tensor_n = 0.000000, padding = 1, stride = 1, groups = 32);  /* [1, 32, 112, 112], layer : 5,  */ 
	bn_sub(&T[258], &T[12], &T[15]);
	bn_div(&T[259], &T[258], &T[257]);
	bn_mul(&T[260], &T[259], &T[13]);
	bn_add(&T[17], &T[260], &T[14]); /* [1, 32, 112, 112], layer : 6,  */ 
	relu6(&T[19], &T[17]);  /* [1, 32, 112, 112], layer : 7,  */ 
	convxbias(&T[21], &T[19], &T[20], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 64, 112, 112], layer : 9,  */ 
	bn_sub(&T[262], &T[21], &T[24]);
	bn_div(&T[263], &T[262], &T[261]);
	bn_mul(&T[264], &T[263], &T[22]);
	bn_add(&T[26], &T[264], &T[23]); /* [1, 64, 112, 112], layer : 10,  */ 
	relu6(&T[28], &T[26]);  /* [1, 64, 112, 112], layer : 11,  */ 
	convxbias(&T[30], &T[28], &T[29], tensor_n = 0.000000, padding = 1, stride = 2, groups = 64);  /* [1, 64, 56, 56], layer : 13,  */ 
	bn_sub(&T[266], &T[30], &T[33]);
	bn_div(&T[267], &T[266], &T[265]);
	bn_mul(&T[268], &T[267], &T[31]);
	bn_add(&T[35], &T[268], &T[32]); /* [1, 64, 56, 56], layer : 14,  */ 
	relu6(&T[37], &T[35]);  /* [1, 64, 56, 56], layer : 15,  */ 
	convxbias(&T[39], &T[37], &T[38], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 128, 56, 56], layer : 17,  */ 
	bn_sub(&T[270], &T[39], &T[42]);
	bn_div(&T[271], &T[270], &T[269]);
	bn_mul(&T[272], &T[271], &T[40]);
	bn_add(&T[44], &T[272], &T[41]); /* [1, 128, 56, 56], layer : 18,  */ 
	relu6(&T[46], &T[44]);  /* [1, 128, 56, 56], layer : 19,  */ 
	convxbias(&T[48], &T[46], &T[47], tensor_n = 0.000000, padding = 1, stride = 1, groups = 128);  /* [1, 128, 56, 56], layer : 21,  */ 
	bn_sub(&T[274], &T[48], &T[51]);
	bn_div(&T[275], &T[274], &T[273]);
	bn_mul(&T[276], &T[275], &T[49]);
	bn_add(&T[53], &T[276], &T[50]); /* [1, 128, 56, 56], layer : 22,  */ 
	relu6(&T[55], &T[53]);  /* [1, 128, 56, 56], layer : 23,  */ 
	convxbias(&T[57], &T[55], &T[56], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 128, 56, 56], layer : 25,  */ 
	bn_sub(&T[278], &T[57], &T[60]);
	bn_div(&T[279], &T[278], &T[277]);
	bn_mul(&T[280], &T[279], &T[58]);
	bn_add(&T[62], &T[280], &T[59]); /* [1, 128, 56, 56], layer : 26,  */ 
	relu6(&T[64], &T[62]);  /* [1, 128, 56, 56], layer : 27,  */ 
	convxbias(&T[66], &T[64], &T[65], tensor_n = 0.000000, padding = 1, stride = 2, groups = 128);  /* [1, 128, 28, 28], layer : 29,  */ 
	bn_sub(&T[282], &T[66], &T[69]);
	bn_div(&T[283], &T[282], &T[281]);
	bn_mul(&T[284], &T[283], &T[67]);
	bn_add(&T[71], &T[284], &T[68]); /* [1, 128, 28, 28], layer : 30,  */ 
	relu6(&T[73], &T[71]);  /* [1, 128, 28, 28], layer : 31,  */ 
	convxbias(&T[75], &T[73], &T[74], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 256, 28, 28], layer : 33,  */ 
	bn_sub(&T[286], &T[75], &T[78]);
	bn_div(&T[287], &T[286], &T[285]);
	bn_mul(&T[288], &T[287], &T[76]);
	bn_add(&T[80], &T[288], &T[77]); /* [1, 256, 28, 28], layer : 34,  */ 
	relu6(&T[82], &T[80]);  /* [1, 256, 28, 28], layer : 35,  */ 
	convxbias(&T[84], &T[82], &T[83], tensor_n = 0.000000, padding = 1, stride = 1, groups = 256);  /* [1, 256, 28, 28], layer : 37,  */ 
	bn_sub(&T[290], &T[84], &T[87]);
	bn_div(&T[291], &T[290], &T[289]);
	bn_mul(&T[292], &T[291], &T[85]);
	bn_add(&T[89], &T[292], &T[86]); /* [1, 256, 28, 28], layer : 38,  */ 
	relu6(&T[91], &T[89]);  /* [1, 256, 28, 28], layer : 39,  */ 
	convxbias(&T[93], &T[91], &T[92], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 256, 28, 28], layer : 41,  */ 
	bn_sub(&T[294], &T[93], &T[96]);
	bn_div(&T[295], &T[294], &T[293]);
	bn_mul(&T[296], &T[295], &T[94]);
	bn_add(&T[98], &T[296], &T[95]); /* [1, 256, 28, 28], layer : 42,  */ 
	relu6(&T[100], &T[98]);  /* [1, 256, 28, 28], layer : 43,  */ 
	convxbias(&T[102], &T[100], &T[101], tensor_n = 0.000000, padding = 1, stride = 2, groups = 256);  /* [1, 256, 14, 14], layer : 45,  */ 
	bn_sub(&T[298], &T[102], &T[105]);
	bn_div(&T[299], &T[298], &T[297]);
	bn_mul(&T[300], &T[299], &T[103]);
	bn_add(&T[107], &T[300], &T[104]); /* [1, 256, 14, 14], layer : 46,  */ 
	relu6(&T[109], &T[107]);  /* [1, 256, 14, 14], layer : 47,  */ 
	convxbias(&T[111], &T[109], &T[110], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 49,  */ 
	bn_sub(&T[302], &T[111], &T[114]);
	bn_div(&T[303], &T[302], &T[301]);
	bn_mul(&T[304], &T[303], &T[112]);
	bn_add(&T[116], &T[304], &T[113]); /* [1, 512, 14, 14], layer : 50,  */ 
	relu6(&T[118], &T[116]);  /* [1, 512, 14, 14], layer : 51,  */ 
	convxbias(&T[120], &T[118], &T[119], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 53,  */ 
	bn_sub(&T[306], &T[120], &T[123]);
	bn_div(&T[307], &T[306], &T[305]);
	bn_mul(&T[308], &T[307], &T[121]);
	bn_add(&T[125], &T[308], &T[122]); /* [1, 512, 14, 14], layer : 54,  */ 
	relu6(&T[127], &T[125]);  /* [1, 512, 14, 14], layer : 55,  */ 
	convxbias(&T[129], &T[127], &T[128], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 57,  */ 
	bn_sub(&T[310], &T[129], &T[132]);
	bn_div(&T[311], &T[310], &T[309]);
	bn_mul(&T[312], &T[311], &T[130]);
	bn_add(&T[134], &T[312], &T[131]); /* [1, 512, 14, 14], layer : 58,  */ 
	relu6(&T[136], &T[134]);  /* [1, 512, 14, 14], layer : 59,  */ 
	convxbias(&T[138], &T[136], &T[137], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 61,  */ 
	bn_sub(&T[314], &T[138], &T[141]);
	bn_div(&T[315], &T[314], &T[313]);
	bn_mul(&T[316], &T[315], &T[139]);
	bn_add(&T[143], &T[316], &T[140]); /* [1, 512, 14, 14], layer : 62,  */ 
	relu6(&T[145], &T[143]);  /* [1, 512, 14, 14], layer : 63,  */ 
	convxbias(&T[147], &T[145], &T[146], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 65,  */ 
	bn_sub(&T[318], &T[147], &T[150]);
	bn_div(&T[319], &T[318], &T[317]);
	bn_mul(&T[320], &T[319], &T[148]);
	bn_add(&T[152], &T[320], &T[149]); /* [1, 512, 14, 14], layer : 66,  */ 
	relu6(&T[154], &T[152]);  /* [1, 512, 14, 14], layer : 67,  */ 
	convxbias(&T[156], &T[154], &T[155], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 69,  */ 
	bn_sub(&T[322], &T[156], &T[159]);
	bn_div(&T[323], &T[322], &T[321]);
	bn_mul(&T[324], &T[323], &T[157]);
	bn_add(&T[161], &T[324], &T[158]); /* [1, 512, 14, 14], layer : 70,  */ 
	relu6(&T[163], &T[161]);  /* [1, 512, 14, 14], layer : 71,  */ 
	convxbias(&T[165], &T[163], &T[164], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 73,  */ 
	bn_sub(&T[326], &T[165], &T[168]);
	bn_div(&T[327], &T[326], &T[325]);
	bn_mul(&T[328], &T[327], &T[166]);
	bn_add(&T[170], &T[328], &T[167]); /* [1, 512, 14, 14], layer : 74,  */ 
	relu6(&T[172], &T[170]);  /* [1, 512, 14, 14], layer : 75,  */ 
	convxbias(&T[174], &T[172], &T[173], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 77,  */ 
	bn_sub(&T[330], &T[174], &T[177]);
	bn_div(&T[331], &T[330], &T[329]);
	bn_mul(&T[332], &T[331], &T[175]);
	bn_add(&T[179], &T[332], &T[176]); /* [1, 512, 14, 14], layer : 78,  */ 
	relu6(&T[181], &T[179]);  /* [1, 512, 14, 14], layer : 79,  */ 
	convxbias(&T[183], &T[181], &T[182], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 81,  */ 
	bn_sub(&T[334], &T[183], &T[186]);
	bn_div(&T[335], &T[334], &T[333]);
	bn_mul(&T[336], &T[335], &T[184]);
	bn_add(&T[188], &T[336], &T[185]); /* [1, 512, 14, 14], layer : 82,  */ 
	relu6(&T[190], &T[188]);  /* [1, 512, 14, 14], layer : 83,  */ 
	convxbias(&T[192], &T[190], &T[191], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 85,  */ 
	bn_sub(&T[338], &T[192], &T[195]);
	bn_div(&T[339], &T[338], &T[337]);
	bn_mul(&T[340], &T[339], &T[193]);
	bn_add(&T[197], &T[340], &T[194]); /* [1, 512, 14, 14], layer : 86,  */ 
	relu6(&T[199], &T[197]);  /* [1, 512, 14, 14], layer : 87,  */ 
	convxbias(&T[201], &T[199], &T[200], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 89,  */ 
	bn_sub(&T[342], &T[201], &T[204]);
	bn_div(&T[343], &T[342], &T[341]);
	bn_mul(&T[344], &T[343], &T[202]);
	bn_add(&T[206], &T[344], &T[203]); /* [1, 512, 14, 14], layer : 90,  */ 
	relu6(&T[208], &T[206]);  /* [1, 512, 14, 14], layer : 91,  */ 
	convxbias(&T[210], &T[208], &T[209], tensor_n = 0.000000, padding = 1, stride = 2, groups = 512);  /* [1, 512, 7, 7], layer : 93,  */ 
	bn_sub(&T[346], &T[210], &T[213]);
	bn_div(&T[347], &T[346], &T[345]);
	bn_mul(&T[348], &T[347], &T[211]);
	bn_add(&T[215], &T[348], &T[212]); /* [1, 512, 7, 7], layer : 94,  */ 
	relu6(&T[217], &T[215]);  /* [1, 512, 7, 7], layer : 95,  */ 
	convxbias(&T[219], &T[217], &T[218], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 1024, 7, 7], layer : 97,  */ 
	bn_sub(&T[350], &T[219], &T[222]);
	bn_div(&T[351], &T[350], &T[349]);
	bn_mul(&T[352], &T[351], &T[220]);
	bn_add(&T[224], &T[352], &T[221]); /* [1, 1024, 7, 7], layer : 98,  */ 
	relu6(&T[226], &T[224]);  /* [1, 1024, 7, 7], layer : 99,  */ 
	convxbias(&T[228], &T[226], &T[227], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1024);  /* [1, 1024, 7, 7], layer : 101,  */ 
	bn_sub(&T[354], &T[228], &T[231]);
	bn_div(&T[355], &T[354], &T[353]);
	bn_mul(&T[356], &T[355], &T[229]);
	bn_add(&T[233], &T[356], &T[230]); /* [1, 1024, 7, 7], layer : 102,  */ 
	relu6(&T[235], &T[233]);  /* [1, 1024, 7, 7], layer : 103,  */ 
	convxbias(&T[237], &T[235], &T[236], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 1024, 7, 7], layer : 105,  */ 
	bn_sub(&T[358], &T[237], &T[240]);
	bn_div(&T[359], &T[358], &T[357]);
	bn_mul(&T[360], &T[359], &T[238]);
	bn_add(&T[242], &T[360], &T[239]); /* [1, 1024, 7, 7], layer : 106,  */ 
	relu6(&T[244], &T[242]);  /* [1, 1024, 7, 7], layer : 107,  */ 
	avg_pool(&T[245], &T[244], size = 7, padding = 0, stride = 1);  /* [1, 1024, 1, 1], layer : 109,  */ 
	convxbias(&T[248], &T[245], &T[246], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 1001, 1, 1], layer : 110,  */ 
	add(&T[249], &T[248], &T[247]);  /* [1, 1001, 1, 1], layer : 111,  */ 
	reshape(&T[250], &T[249], 0, 0, 1, 1001);  /* [1, 1001], layer : 112,  */ 
	reshape(&T[251], &T[250], 0, 0, 1, 1001);  /* [1, 1001], layer : 113,  */ 
	softmax(&T[252], &T[251]);  /* [1, 1001], layer : 114,  */ 
	reshape(&T[1], &T[252], 0, 0, 1, 1001);  /* [1, 1001], layer : 115,  */ 
}
#endif

int main(int argc, char* argv[])
{
	if(argc<2)
	{
		cout << "Usage: mobilenetTest <ppm image file>" << endl;
		return 0;
	}
	//nnKernel kernel = nnCompileProgram("./graph.nnef");
  
  //--------------------------------------------------
#if 1

	extern tensor T[512];
    init(T);
	const std::string filename = argv[1];
	std::ifstream file(filename.c_str(),std::ios::binary);
	std::string str;
	char ptr[inputsize];
	
	//read header
	int iteration = 5;
	for(int i=0;i<iteration;i++)
	{
		std::getline(file,str);
		std::cout << str << std::endl;
		if (str == "255")
			break;
	}
    
	file.read(ptr,(inputsize));
	assert(file);
	uint8_t tmpValue = 0;
	//cout << "input shape: " << P[0].shape << endl;
	T[0].data = (float * ) calloc(inputsize * sizeof(float), sizeof(float));
	T[0].n = 1;
	T[0].c = 3;
	T[0].h = 224;
	T[0].w = 224;
	int idx =0;
	for(int h=0;h<224;h++)//height
	{
		for(int w=0;w<224;w++)//width
		{
			for(int ch=0;ch<3;ch++)//channel
			{
				tmpValue = (uint8_t)ptr[idx];
				T[0].data[0 * T[0].c * T[0].h * T[0].w + ch * T[0].h * T[0].w + h * T[0].w + w] = tmpValue/255.0;
				idx++;
			}
		}
	}

	//initialize the input by 28 x 28 (0,1)matrix of the images
	//******** NNEF *********
	//nnRun(kernel, 1);
    double start = clock();
    kernel(T);
    double end = clock();
    printf("[kernel time = %1.3f seconds]\n",(end-start)/CLOCKS_PER_SEC);
	//***********************
  
	float fvalue = 0;
	float maxValue = std::numeric_limits<float>::min();
	idx = 0;
    float TOP[5] = {maxValue, maxValue, maxValue, maxValue, maxValue};
    float TOP_POS[5] = {0,0,0,0,0};
	for (int i = 0; i < T[1].size; i++)
	{
		fvalue = T[1].data[i];
        int pos = 0;
        bool find = false;
        for (int j = 0; j < 5; j++)
        {
		    if(fvalue > TOP[j])
		    {
			    //maxValue = fvalue;
                TOP[j] = fvalue;
			    idx = i-1;
                pos = j;
                find = true;
                //std::cout << "got max value: " << maxValue << "with idx: " << i-1 << std::endl;
                TOP_POS[pos] = idx;
                break;
		    }
        }
	}

  	string line;
	ifstream label_file ("imagenet1000_clsid_to_human.txt");

	vector<string> labels;
	if(label_file.is_open())
	{
		while(getline(label_file,line))
		{
			labels.push_back(line);
		}
		label_file.close();
	}

	cout << "TOP-1 : " << TOP[0] << ", " << labels[TOP_POS[0]] << endl;
    cout << "TOP-2 : " << TOP[1] << ", " << labels[TOP_POS[1]] << endl;
    cout << "TOP-3 : " << TOP[2] << ", " << labels[TOP_POS[2]] << endl;
    cout << "TOP-4 : " << TOP[3] << ", " << labels[TOP_POS[3]] << endl;
    cout << "TOP-5 : " << TOP[4] << ", " << labels[TOP_POS[4]] << endl;

#endif
  return 0;
}

