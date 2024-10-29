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


logging.basicConfig(
    filename='face_results.log',  # Specify your log file name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
#globals:
BATCH_SIZE = 32 # batch of images to load each iteration
PATH_TO_IMAGE_FOLDER = './image_data'
DEVICE = 'CPU'


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
        detector = SCRFD.from_path("./models/scrfd_face.onnx")
        print("loaded detector")
        return detector
    def detect_person_folder(self, path_to_folder):
        '''  
        Input: image folder name 
        Output: list of [img_path, [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]
        '''
        # Use the provided path_to_folder instead of a fixed variable
        img_paths = glob.glob(os.path.join(path_to_folder, '*.jpg'))
        
        results = {}  # Initialize an empty list to hold results
        for img_path in tqdm.tqdm(img_paths, desc="Processing faces"):
            img = Image.open(img_path)
            print('            path to image: ' + img_path)
            
            faces = self.detector.detect(img, threshold=self.threshold)
            
            bboxes = []  # Initialize a list to hold bounding boxes for the current image

            for face in faces:
                bbox = face.bbox  
                x1, y1 = bbox.upper_left.x, bbox.upper_left.y  # Top-left corner
                x2, y2 = bbox.lower_right.x, bbox.lower_right.y  # Bottom-right corner
                
                # Append the bounding box to the list
                bboxes.append([x1, y1, x2, y2])
                bboxes_list = [[int(x) for x in bbox] for bbox in bboxes]
            # Nest img_path and bboxes into the results list
            results[img_path] = bboxes_list
        # for img_path, bboxes in results.items():
        #     print(f"Image Path: {img_path}, Bounding Boxes: {bboxes}")
        #     print('\n')
        # print(type(results))
        return results 

    
    def postprocess_bbox():
        '''   
        input:batch of List(conf,bboxes)
        output: dict of returned image data.
        '''
        # load image & bbox.
        # 
        # 
        # crop image according to bbox
        # return cropped_image. 
        pass
    
    
    
if __name__ == "__main__":
    # run code
    print("starting test")
    extractor = BatchExtractor()
    results = extractor.detect_person_folder(PATH_TO_IMAGE_FOLDER)
    for img_path, bboxes in results.items():
        logging.info(f"Image Path: {img_path}, Bounding Boxes: {bboxes}")

    # run - all
    pass
    
    
    
    
    
    
    






















