1. System Requirements:
	Ubuntu	16.04
	Python	2.7.12
	Python Packages:
		numpy 1.14.5
		scipy 1.1.0

	Tensorflow and dependencies:
		Tensorflow  1.4.1
		CUDA	    8.0.61
		cuDNN	    6.0.21

2. Installation Guide (required time, <120 minutes):

- Operation System
	Ubuntu 16.04 download from https://www.ubuntu.com/download/desktop
	
- Python and packages
	Download Python 2.7.12 tarball on https://www.python.org/downloads/release/python-2712/
	Unzip and install:
		tar -zxvf Python-2.7.12.tgz
		cd ./Python-2.7.12
		./configure
		make

	Package Installation:
		pip install numpy==1.14.5
		pip install scipy==1.1.0

	Tensorflow Installation:
		(for GPU use)
		pip install tensorflow-gpu==1.4.1
		(for CPU only)
		pip install tensorflow==1.4.1


(for GPU use)

- CUDA Toolkit 8.0 
	wget -O cuda_8_linux.run https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
	sudo chmod +x cuda_8_linux.run
	./cuda_8_linux.run

- cuDNN 6.0.21
	Download CUDNN tarball on https://developer.nvidia.com/cudnn
	Unzip and install:
		tar -zxvf cudnn-8.0-linux-x64-v6.0.tgz 

For more details, please refer to CUDA, CuDNN, and Tensorflow installation guide on Github: 			
	https://gist.github.com/ksopyla/813a62d6afc4307755e5832a3b62f432


3. Demo Instructions (required time, <1 min):

Input1: ./dataset/        # List of Target Sequence(s)
	File format:
	Target number   30 bp target sequence (4 bp + 20 bp protospacer + PAM + 3 bp)
	  1   TCTGGCGGTCTCAAGCACTACCTACGTCAG
	  2   GGTGGGGGTTAGACGAATATCAGGAGACTA

Input2: ./DeepxCas9_weight/ # Pre-trained Weight Files

Output: RANK_final_DeepxCas9_weight.txt
	Predicted activity score for sequence 1 and 2:
	25.55347824, 22.9860401153564

Run script:
	python ./Test.py

Modification for personalized runs:

	<Test.py>
	## System Paths ##
	path                 = './dataset/'
	parameters           = {'0': 'sample.txt'}

	## Run Parameters ##
	TEST_NUM_SET         = [0] # List can be expanded in case of multiple test parameters
	best_model_path_list = ['./DeepxCas9_weight/']

sample.txt can be replaced or modified to include target sequence of interest

 


