# MaskRCNNDetectron2
Running Mask R-CNN using Detectron2

Installation required:
    • Download Pythonv3.6 or higher [sudo apt install python3.8]
    • Install Pytorch and Torchvision using the commands recommended on the Pytorch website (https://pytorch.org/) 
    	[pip3 install torch torchvision]
    • Install opencv python: “pip3 install opencv-contrib-python”
    • Install pycocotools: “pip3 install cython; 
    • $pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    • Build detectron2 from source using the commands found here: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#build-detectron2-from-source
clone detectron2 in a specific folder 
$git clone https://github.com/facebookresearch/detectron2.git
To run detectron2
$python3 -m pip install -e detectron2]

    • Install required modules: pip install cv2
					pip install random
					pip install os

For further help with setting up detectron2 visit: https://github.com/facebookresearch/detectron2/blob/master/README.md
