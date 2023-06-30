make install: 
	conda install pycocotools
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install transformers==4.30.2 scipy tensorboard