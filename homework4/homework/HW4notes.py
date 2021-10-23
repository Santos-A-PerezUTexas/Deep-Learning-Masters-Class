SLIDES: https://piazza.com/class/ksjhagmd59d6sg?cid=191
SLIDES: https://piazza.com/class/ksjhagmd59d6sg?cid=191
SLIDES: https://piazza.com/class/ksjhagmd59d6sg?cid=191

HW4 SLIDES:  https://docs.google.com/presentation/d/e/2PACX-1vR6bYbuIJeA1og36QOdEMANAXbdbCpjaPVWkZNthAqIJ-iKOPKtYcLnb7n_rlNGvbiukP7W3JLaM1HQ/pub?start=false&loop=false&delayms=3000&slide=id.p1
        
    



Model
How much should our model differ from our answer in HW3? I was able to get full marks on HW3 with the model I've implemented
but now it performs poorly with AP  less than 0.2. I have also implemented Focal loss and have attempted to train with Focal loss and BCEWithLogitsLoss and the
result is the same. Is there any major architecture changes that need to be done in our FCN model?

---->Make sure that your detect function is working correctly. If you look to grader/tests.py, you will see that the tests for the detect function only checks if 
---->the output is in the correct format, not that it actually returns valid detections. See below from line 177.

            assert len(d) == 3, 'Return three lists of detections'
            assert len(d[0]) <= 30 and len(d[1]) <= 30 and len(d[2]) <= 30, 'Returned more than 30 detections per class'
            assert all(len(i) == 5 for c in d for i in c), 'Each detection should be a tuple (score, cx, cy, w/2, h/2)'

------->Your detect function is used by the grader when it evaluates your actual model at line 197. So, if your detect function is working properly, 
-------->you will also see poor performance when the grader evaluates your model.

-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 
---->Q1:

dense_transform.py, Toheatmap class: this line does not work as it says image is PIL so does not have shape:

  peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)

  After I change it to:
    
  peak, size = detections_to_heatmap(dets, image.size, radius=self.radius)

Is this correct?

----->A:  ToHeatmap should be used after ToTensor, that way it is not a PIL image anymore.
  

-------Q2:

When the model received the input from dataloader, the input is a tuple of 3 tensors of size:

torch.Size([32, 3, 96, 128])    #Image
torch.Size([32, 3, 128, 96])    #Heatmap
torch.Size([32, 2, 128, 96])    #Labels for Extracredit

I understand the first is image. What about the rest two? Are those heatmaps of objects? And why is the 2nd dim different? 

-------->A. The second tensor is the heatmap, the third is the labels for the extra credit (full object detection). 
  They should have the same dimensions as the image; maybe the switch was caused by the edit you made?
  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
  
 
  assertion backwards for the Radius_1 test parameters.
  
          assert len(p) == (img > 0).sum(), 'Expected exactly %d detections, got %d' % (len(p), (img > 0).sum())
    
    STRING SHOULD INSTEAD BE:
      
               'Expected exactly %d detections, got %d' % ((img > 0).sum(),len(p))
 
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
    
      
      Could the local minimum also be an object? eg a bright light from an oncoming vehicle against a dark background?
           --->The peaks are extracted from your network’s output so you don’t have to worry about what the image looks like before.
  
  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
  Width and Height
Is there are a resource we can look into when dealing with 
balancing  the two losses between the image and the width and height channels?
or calculating losses for separate channels?   

----->For detection, BCE loss range from 0~1, for height and width prediction, the loss is much larger.
----->So you may want to assign BCE loss a much higher weight.
    -----> Binary Cross Entropy loss (criterion)  between the target and the input probabilities
    ----->CLASStorch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')  #notice weight parameter
            -> weight (Tensor, optional) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.

I see so would this affect the Focal Loss implementation or should we stick to BCE for height and widht?
How would we go about doing the backwards  pass during training when the losses have been calculated?
---->For height and width, as a regression task, you need MSE or MAE loss function. 
---->For backpropagation, just like:
  
              L1 = BCE()
              L2 = MSE()

              L = w1*L1 + w2*L2
              L.backward()
          
 
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------

  Less than 150 Epochs

  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------

Detection Test
Is there a way we can change the detection test in the test.py to in such a way that it will always get full points unless we have our
detect or peak extraction wrong? Finding it impossible to debug what is wrong and thought passing the correct label to the test would help in such a case.

----->I’m not sure if there’s an easy way to do this, but I’m also not sure I understand why you want to. The extract_peak function is already graded separately.
----->For the detect function you could consider creating your own test case (creating an image with red/green/blue dots and using that insted of your model output)
----->and seeing if the function returns the expected output.

  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------




  
