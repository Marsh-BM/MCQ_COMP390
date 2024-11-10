import cv2
import numpy as np
import os
import torch
from torchvision import transforms
import csv


class IDScanner:
    def __init__(self, image, model, device):
    # def __init__(self, image, model, device):
        self.image = image # ID_
        self.model = model
        self.device = device
        self.idx_to_answer = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

    def preprocess(self, images):
        # Preprocess the images
        processed_images = []
        for image in images:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((184, 30)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ])
            processed_image = transform(image).unsqueeze(0)  
            processed_images.append(processed_image)
        
        return torch.cat(processed_images, dim=0)  
    
    def predict_digits(self, images):
        images = self.preprocess(images).to(self.device)  
        self.model.eval()  
        with torch.no_grad():  
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_digits = [self.idx_to_answer[pred.item()] for pred in predicted]
        
        return predicted_digits

    def split_id(self, page_num):
        image = self.image
        cropped_images = [] 
        predicted_id = '' 
        # Get the image`s height and width
        height, width = image.shape[:2]
        # Calculate the total spacing width (8 gaps * 3 pixels)
        total_spacing_width = 8 * 3
        # Adjust the total width available for the parts
        available_width = width - total_spacing_width
        # Calculate the width of each part
        part_width = available_width // 9

        # Split the image into 9 parts
        for i in range(9):
            start_x = i * (part_width + 4)
            end_x = start_x + part_width
            cropped_image = self.image[:, start_x:end_x]
            cropped_images.append(cropped_image)

            # predicted_digit = self.predict_digit(cropped_image)
            # predicted_id += predicted_digit

            # # Save the cropped image
            # output_path = os.path.join(output_dir, f'{page_num}_part_{i+1}.jpg')
            # cv2.imwrite(output_path, cropped_image)
        predicted_ids = self.predict_digits(cropped_images) 
        
        # for i, cropped_image in enumerate(cropped_images):
        #     output_path = os.path.join(output_dir, f'{page_num}_part_{i+1}.jpg')
        #     cv2.imwrite(output_path, cropped_image)

        predicted_id = ''.join(predicted_ids)
        print(predicted_id)
        return predicted_id


 

    
# if __name__ == '__main__':
#     # Read the image
#     image = cv2.imread('id_train0.png')    
#     # image = cv2.imread('Batman.png') 
#     scale_percent = 50 # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


#     # Deskew the image
#     deskewed = deskew(image)


#     cv2.imshow('Deskewed', deskewed)
#     cv2.waitKey(0)

#     # Save the image
#     cv2.imwrite('test01.png', deskewed)

#     # Directory to save the split images
#     output_dir = 'ID_test'

#     # Split the image and save the parts
#     split_id('test01.png', output_dir)