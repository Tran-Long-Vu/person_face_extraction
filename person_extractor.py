# import modules.
import glob
import os
import onnxruntime # cpu 
import tqdm as tqdm
import insightface
import numpy as np
from scrfd import SCRFD, Threshold
from PIL import Image, ImageDraw
import logging
import cv2
#globals:
BATCH_SIZE = 32 # batch of images to load each iteration
PATH_TO_IMAGE_FOLDER = './image_data'
DEVICE = 'CPU'


logging.basicConfig(
    filename='person_results.log',  # Specify your log file name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BatchExtractor():
    def __init__(self):
        self.detector = self.load_onnx_model()
        self.threshold = Threshold(probability=0.4)    
        
        # self.y
        # self.z 
        # self.d
        pass
    
    
    def load_onnx_model(self):
        ''' 
        read onnx model and put in ready mode. 
        '''
        detector = insightface.model_zoo.get_model('./models/scrfd_person.onnx', download=False)
        detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))
        return detector
    
    def detect_person(self, img, detector):
        bboxes, kpss = detector.detect(img)
        bboxes = np.round(bboxes[:,:4]).astype(np.int64)
        kpss = np.round(kpss).astype(np.int64)
        kpss[:,:,0] = np.clip(kpss[:,:,0], 0, img.shape[1])
        kpss[:,:,1] = np.clip(kpss[:,:,1], 0, img.shape[0])
        vbboxes = bboxes.copy()
        vbboxes[:,0] = kpss[:, 0, 0]
        vbboxes[:,1] = kpss[:, 0, 1]
        vbboxes[:,2] = kpss[:, 4, 0]
        vbboxes[:,3] = kpss[:, 4, 1]
        return bboxes, vbboxes
    
    def detect_person_folder(self, path_to_folder):
        '''  
        Input: image folder name 
        Output: dict of {img_path: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]}
        '''
        # Use the provided path_to_folder instead of a fixed variable
        img_paths = glob.glob(os.path.join(path_to_folder, '*.jpg'))
        
        results = {}  # Initialize an empty dictionary to hold results
        for img_path in tqdm.tqdm(img_paths, desc="Processing person"):
            img = cv2.imread(img_path)
            # print('            path to image: ' + img_path)
            bboxes, vbboxes = self.detect_person(img, self.detector)

            # Keep bounding boxes as float and nest them in a list
            bboxes_list = [bbox.tolist() for bbox in bboxes]
            
            # Store results in a dictionary with img_path as the key
            results[img_path] = bboxes_list  
        
        # Print results for debugging
        #for img_path, bboxes in results.items():
        #    print(f"Image Path: {img_path}, Bounding Boxes: {bboxes}")
        #    print('\n')
        #print(type(results))
        return results  # Return the dictionary of results


        
        
    
    
if __name__ == "__main__":
    # run code
    print("starting test")
    extractor = BatchExtractor()
    results = extractor.detect_person_folder(PATH_TO_IMAGE_FOLDER)
    for img_path, bboxes in results.items():
        logging.info(f"Image Path: {img_path}, Bounding Boxes: {bboxes}")

    # run - all
    pass
    
    
        
        
        
        
        



