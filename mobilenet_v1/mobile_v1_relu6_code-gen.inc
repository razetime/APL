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
}

// #########################################################
// # (relu, min6) -> (relu6) // merge
// # -------------------------------------------------------

void kernel(tensor * T)
{
        convxbias(&T[3], &T[0], &T[2], tensor_n = 0.000000, padding = 1, stride = 2, groups = 1);  /* [1, 32, 112, 112], layer : 1,  */
        batch_normalization(&T[8], &T[3], &T[6], &T[7], &T[5], &T[4], epsilon = 0.001000);  /* [1, 32, 112, 112], layer : 2,  */
        relu6(&T[10], &T[8]);  /* [1, 32, 112, 112], layer : 3,  */
        convxbias(&T[12], &T[10], &T[11], tensor_n = 0.000000, padding = 1, stride = 1, groups = 32);  /* [1, 32, 112, 112], layer : 5,  */
        batch_normalization(&T[17], &T[12], &T[15], &T[16], &T[14], &T[13], epsilon = 0.001000);  /* [1, 32, 112, 112], layer : 6,  */
        relu6(&T[19], &T[17]);  /* [1, 32, 112, 112], layer : 7,  */
        convxbias(&T[21], &T[19], &T[20], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 64, 112, 112], layer : 9,  */
        batch_normalization(&T[26], &T[21], &T[24], &T[25], &T[23], &T[22], epsilon = 0.001000);  /* [1, 64, 112, 112], layer : 10,  */
        relu6(&T[28], &T[26]);  /* [1, 64, 112, 112], layer : 11,  */
        convxbias(&T[30], &T[28], &T[29], tensor_n = 0.000000, padding = 1, stride = 2, groups = 64);  /* [1, 64, 56, 56], layer : 13,  */
        batch_normalization(&T[35], &T[30], &T[33], &T[34], &T[32], &T[31], epsilon = 0.001000);  /* [1, 64, 56, 56], layer : 14,  */
        relu6(&T[37], &T[35]);  /* [1, 64, 56, 56], layer : 15,  */
        convxbias(&T[39], &T[37], &T[38], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 128, 56, 56], layer : 17,  */
        batch_normalization(&T[44], &T[39], &T[42], &T[43], &T[41], &T[40], epsilon = 0.001000);  /* [1, 128, 56, 56], layer : 18,  */
        relu6(&T[46], &T[44]);  /* [1, 128, 56, 56], layer : 19,  */
        convxbias(&T[48], &T[46], &T[47], tensor_n = 0.000000, padding = 1, stride = 1, groups = 128);  /* [1, 128, 56, 56], layer : 21,  */
        batch_normalization(&T[53], &T[48], &T[51], &T[52], &T[50], &T[49], epsilon = 0.001000);  /* [1, 128, 56, 56], layer : 22,  */
        relu6(&T[55], &T[53]);  /* [1, 128, 56, 56], layer : 23,  */
        convxbias(&T[57], &T[55], &T[56], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 128, 56, 56], layer : 25,  */
        batch_normalization(&T[62], &T[57], &T[60], &T[61], &T[59], &T[58], epsilon = 0.001000);  /* [1, 128, 56, 56], layer : 26,  */
        relu6(&T[64], &T[62]);  /* [1, 128, 56, 56], layer : 27,  */
        convxbias(&T[66], &T[64], &T[65], tensor_n = 0.000000, padding = 1, stride = 2, groups = 128);  /* [1, 128, 28, 28], layer : 29,  */
        batch_normalization(&T[71], &T[66], &T[69], &T[70], &T[68], &T[67], epsilon = 0.001000);  /* [1, 128, 28, 28], layer : 30,  */
        relu6(&T[73], &T[71]);  /* [1, 128, 28, 28], layer : 31,  */
        convxbias(&T[75], &T[73], &T[74], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 256, 28, 28], layer : 33,  */
        batch_normalization(&T[80], &T[75], &T[78], &T[79], &T[77], &T[76], epsilon = 0.001000);  /* [1, 256, 28, 28], layer : 34,  */
        relu6(&T[82], &T[80]);  /* [1, 256, 28, 28], layer : 35,  */
        convxbias(&T[84], &T[82], &T[83], tensor_n = 0.000000, padding = 1, stride = 1, groups = 256);  /* [1, 256, 28, 28], layer : 37,  */
        batch_normalization(&T[89], &T[84], &T[87], &T[88], &T[86], &T[85], epsilon = 0.001000);  /* [1, 256, 28, 28], layer : 38,  */
        relu6(&T[91], &T[89]);  /* [1, 256, 28, 28], layer : 39,  */
        convxbias(&T[93], &T[91], &T[92], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 256, 28, 28], layer : 41,  */
        batch_normalization(&T[98], &T[93], &T[96], &T[97], &T[95], &T[94], epsilon = 0.001000);  /* [1, 256, 28, 28], layer : 42,  */
        relu6(&T[100], &T[98]);  /* [1, 256, 28, 28], layer : 43,  */
        convxbias(&T[102], &T[100], &T[101], tensor_n = 0.000000, padding = 1, stride = 2, groups = 256);  /* [1, 256, 14, 14], layer : 45,  */
        batch_normalization(&T[107], &T[102], &T[105], &T[106], &T[104], &T[103], epsilon = 0.001000);  /* [1, 256, 14, 14], layer : 46,  */
        relu6(&T[109], &T[107]);  /* [1, 256, 14, 14], layer : 47,  */
        convxbias(&T[111], &T[109], &T[110], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 49,  */
        batch_normalization(&T[116], &T[111], &T[114], &T[115], &T[113], &T[112], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 50,  */
        relu6(&T[118], &T[116]);  /* [1, 512, 14, 14], layer : 51,  */
        convxbias(&T[120], &T[118], &T[119], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 53,  */
        batch_normalization(&T[125], &T[120], &T[123], &T[124], &T[122], &T[121], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 54,  */
        relu6(&T[127], &T[125]);  /* [1, 512, 14, 14], layer : 55,  */
        convxbias(&T[129], &T[127], &T[128], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 57,  */
        batch_normalization(&T[134], &T[129], &T[132], &T[133], &T[131], &T[130], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 58,  */
        relu6(&T[136], &T[134]);  /* [1, 512, 14, 14], layer : 59,  */
        convxbias(&T[138], &T[136], &T[137], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 61,  */
        batch_normalization(&T[143], &T[138], &T[141], &T[142], &T[140], &T[139], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 62,  */
        relu6(&T[145], &T[143]);  /* [1, 512, 14, 14], layer : 63,  */
        convxbias(&T[147], &T[145], &T[146], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 65,  */
        batch_normalization(&T[152], &T[147], &T[150], &T[151], &T[149], &T[148], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 66,  */
        relu6(&T[154], &T[152]);  /* [1, 512, 14, 14], layer : 67,  */
        convxbias(&T[156], &T[154], &T[155], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 69,  */
        batch_normalization(&T[161], &T[156], &T[159], &T[160], &T[158], &T[157], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 70,  */
        relu6(&T[163], &T[161]);  /* [1, 512, 14, 14], layer : 71,  */
        convxbias(&T[165], &T[163], &T[164], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 73,  */
        batch_normalization(&T[170], &T[165], &T[168], &T[169], &T[167], &T[166], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 74,  */
        relu6(&T[172], &T[170]);  /* [1, 512, 14, 14], layer : 75,  */
        convxbias(&T[174], &T[172], &T[173], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 77,  */
        batch_normalization(&T[179], &T[174], &T[177], &T[178], &T[176], &T[175], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 78,  */
        relu6(&T[181], &T[179]);  /* [1, 512, 14, 14], layer : 79,  */
        convxbias(&T[183], &T[181], &T[182], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 81,  */
        batch_normalization(&T[188], &T[183], &T[186], &T[187], &T[185], &T[184], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 82,  */
        relu6(&T[190], &T[188]);  /* [1, 512, 14, 14], layer : 83,  */
        convxbias(&T[192], &T[190], &T[191], tensor_n = 0.000000, padding = 1, stride = 1, groups = 512);  /* [1, 512, 14, 14], layer : 85,  */
        batch_normalization(&T[197], &T[192], &T[195], &T[196], &T[194], &T[193], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 86,  */
        relu6(&T[199], &T[197]);  /* [1, 512, 14, 14], layer : 87,  */
        convxbias(&T[201], &T[199], &T[200], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 512, 14, 14], layer : 89,  */
        batch_normalization(&T[206], &T[201], &T[204], &T[205], &T[203], &T[202], epsilon = 0.001000);  /* [1, 512, 14, 14], layer : 90,  */
        relu6(&T[208], &T[206]);  /* [1, 512, 14, 14], layer : 91,  */
        convxbias(&T[210], &T[208], &T[209], tensor_n = 0.000000, padding = 1, stride = 2, groups = 512);  /* [1, 512, 7, 7], layer : 93,  */
        batch_normalization(&T[215], &T[210], &T[213], &T[214], &T[212], &T[211], epsilon = 0.001000);  /* [1, 512, 7, 7], layer : 94,  */
        relu6(&T[217], &T[215]);  /* [1, 512, 7, 7], layer : 95,  */
        convxbias(&T[219], &T[217], &T[218], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 1024, 7, 7], layer : 97,  */
        batch_normalization(&T[224], &T[219], &T[222], &T[223], &T[221], &T[220], epsilon = 0.001000);  /* [1, 1024, 7, 7], layer : 98,  */
        relu6(&T[226], &T[224]);  /* [1, 1024, 7, 7], layer : 99,  */
        convxbias(&T[228], &T[226], &T[227], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1024);  /* [1, 1024, 7, 7], layer : 101,  */
        batch_normalization(&T[233], &T[228], &T[231], &T[232], &T[230], &T[229], epsilon = 0.001000);  /* [1, 1024, 7, 7], layer : 102,  */
        relu6(&T[235], &T[233]);  /* [1, 1024, 7, 7], layer : 103,  */
        convxbias(&T[237], &T[235], &T[236], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 1024, 7, 7], layer : 105,  */
        batch_normalization(&T[242], &T[237], &T[240], &T[241], &T[239], &T[238], epsilon = 0.001000);  /* [1, 1024, 7, 7], layer : 106,  */
        relu6(&T[244], &T[242]);  /* [1, 1024, 7, 7], layer : 107,  */
        avg_pool(&T[245], &T[244], size = 7, padding = 0, stride = 1);  /* [1, 1024, 1, 1], layer : 109,  */
        convxbias(&T[248], &T[245], &T[246], tensor_n = 0.000000, padding = 1, stride = 1, groups = 1);  /* [1, 1001, 1, 1], layer : 110,  */
        add(&T[249], &T[248], &T[247]);  /* [1, 1001, 1, 1], layer : 111,  */
        reshape(&T[250], &T[249], 0, 0, 1, 1001);  /* [1, 1001], layer : 112,  */
        reshape(&T[251], &T[250], 0, 0, 1, 1001);  /* [1, 1001], layer : 113,  */
        softmax(&T[252], &T[251]);  /* [1, 1001], layer : 114,  */
        reshape(&T[1], &T[252], 0, 0, 1, 1001);  /* [1, 1001], layer : 115,  */
}
