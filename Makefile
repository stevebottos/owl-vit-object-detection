make install: 
	conda install pycocotools
	conda install -n owl ipykernel --update-deps --force-reinstall
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt