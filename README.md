# APL
Advanced Programming Language

# How to run the benchmarks

Steps:

## Mobilenet_v1:

	cd APL/mobilenet_v1
	
	make clean
	
	make
	
	# inference an image
	
	./mobilenetTest images/gold_fish_224.ppm
	

## LeNet_Mnist:

	cd APL/LeNet_Mnist
	
	make clean
	
	make
	
	./mnistTest
	

# Operators:

	APL/nnlib.c
