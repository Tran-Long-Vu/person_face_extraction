# Person detection inference module
- Input: folder of images in rgb format
- Output:batch of List(Bounding boxes) of inferred persons inside image batch. 


## How to install & setup
- pip install requirements.txt.
- python setup directory.py.
- put images in './image_data'.
- put onnx model in './models'.
- './outputs' images is for reference.

## Run
- python person_extractor.py. 
- python face_extractor.py
- check output in person_results.log and face_results.log

## Results
- logged in .log format
- saved in 'results' dict

## Comments




