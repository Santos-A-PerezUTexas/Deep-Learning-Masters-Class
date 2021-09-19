if image_index == 0:
          print (f'image index is zero')
          image_file_name = "..\data\\train\\"+labelsFILE_image_row[0]
          print (f'This is the image name at zero:  {image_file_name}')
          val = input("Press any key")
          print(val)
        elif image_index < 4:
          print (f'image index is less than 4')
          image_file_name = "..\data\\train\\"+labelsFILE_image_row[0]
          print (f'This is the image name {image_index}:  {image_file_name}')
          print (f'This is the image list so far {self.image_list}')
          val = input("Press any key")
          print(val)
          
          
          
          
             print(f'Image index is {image_index}, about to evaluate is bigger than 0')    
        
        if image_index > 0:
                  
          #image_file_name = "../data/train/"+labelsFILE_image_row[0]  for colab Sept 18
          #print(image_file_name)  commented Sept 17 evening
          image_file_name = "..\data\\train\\"+labelsFILE_image_row[0] 
          print(f'Loading Image {image_file_name}')
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          #self.Image_tensor = torch.tensor(self.Image_To_Tensor(self.one_image), requires_grad=True) 
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
          
          if image_index == 4:
            print (f'Image index is 4, this is the current self.Image_tensor, {self.Image_tensor}')
            print (f'This tensor is going to be added to the list, which looks like this {self.image_list}')
            val = input("PRESS ANY KEY to continue")
            print(val)
            
          self.imageDATASET[image_index-1] = self.Image_tensor  #added -1 to image_index Sept 18   
          self.image_list.append(self.Image_tensor)
          
          
            for i in LABEL_NAMES:  #from string to string, iterate through the strongs
            if i==current_label_string:
              self.label_list.append(label_string_to_number)
              
            label_string_to_number += 1
          
                  
          
          
          #print (f'I just assigned self.labels[{image_index-1}] the value {self.labels[image_index-1]} which corresponds to label {labelsFILE_image_row[1]}')
          
        image_index += 1 
        
        