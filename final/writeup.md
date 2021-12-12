# UT Fall 2021 Deep Learning Final Project

#### Teammember: Santos Perez | Yilong Chen | Weizhe Tang           

## Introdudction 
Bulding a good AI model sucessfully playing in a game competition has always been a formidable task  for game companies and machine learning engineers. The famous competition between  Alpha Go and Ke Jie  indeed proves that a machine can do better job than humans in some game competitions. Along those lines, in this project, we are tasked with building an agent which could compete with various AI models in a 2x2 competition in the SuperTuxIcehockey game. Although we've never played this game, when we watched the demo it seemed similar to a soccer competition.  With this knowledge,  we began  by first dividing our group into two sub-groups, to explore more solutions. Mr. Santos Perez  was group1 (along with someone else who dropped course), and Mr. Weizhe Tang and Mr. Yilong Chen  in group2.  We worked together by holding weekly zoom meetings to discuss progress and exchange ideas. In the end, with this approach, we decided to solve the problem with an image based agent.


## Originality
### Data collection
#### 1 We decided to use the VideoRecorder class, since this would allow us to store both render_data instance labels, as well as soccer state.

#### 2  A class method called _to_image300_400(params) was implemented, which is the same as _to_image from HW5,  but it converts -1..1 image coordinates to 300/400 image coordinates as depicted in the project slides (x 400 range, y 300 range), with the 0,0 coordinate now at the top left.

#### 3 The collect() method from HW5 was then invoked, and was called with the soccer state  and the render_data instance as params, as well as the image and proj/view for team1 or team2 images.
###4 To balance the data set, 5000 image/labels pairs were generated for team1, and 5000 for team2.

###5  Labels were then generated with either the soccer/puck location, or the render_data instance (bitshifted by 24), which in the end we decide not to use.

### Planner
#### 1  The planner had to meet the timeout limitations, and a model which closely resembles HW4 was used for that purpose, and was tested and complied with time limits.  
#### 2  The method spatial_argmax() was borrowed from HW5, but the points were normalized/converted to  300/400 range after after the call to spatial_argmax, and that normalized output was then returned by the model.  Training the model was similar to HW5 training. 

### Controller
#### 1
#### 2
#### 3

## Exploration period(before the mid of second week(10 days))
### Group1
The first group decided to do an image based agent. His main idea is similar to the HW5,  to use the planner to predict/detect the image location of the puck, then use controller to steer toward the puck.
First week's main problem for Santos is collecting the data. The  difficulties he encountered is (1) how to convert the world coordinate to image coordinate, (2) Whether to predict render_data bitshift or image coords, (3) whether to use -1..1 coords or 300/400, (4) whether to predict puck_in_image flag/classification. 
Solutions: (*1) image_coords were generated using a modified version of HW5 _to_image()_, which converted -1..1 range to 300/400 range as per the Project slides.   (*2)Bitshifted  Render_data instance labels were succesfully generated (4K image/labels), but during training after 80 epochs the loss increased, and model predicted a 3 object as a 6 object (for instance).  So it was decided to use image coords as labels instead of render_data instance.  (*3) 300/400 range for image coords was used because the image is 400,300 (x,y), and the slides made use of this range, which would also have allowed us to use render_data instances to predict location of '8' object (which we decided not to use in the end).  (*4) render_data instances were the best choice for puck_in_image classification, since a modified version of _to_image_ did not properly classify puck based on range > 400 (or abs(x)>1 in in the -1..1 range).  However, as the controller worked without predicting puck_in_image, we decided not to classify puck.   

### Group2
Initially, Yilong and Weizhe also think about Image agent. However, they think that Image agent is not a optimal solution for this project, so, they turn toward the state agent. They are doing some research on internet and find the q-learning might be a good solution for this game. The main problem they encounter is how to design a reward, and Yilong comes out with a clever idea that he could use the summation of (the distance between the ball and the goal) and (the distance between the kart and the ball) to measure the reward. I.E. reward = previous_distance - current_distance, if the distance become smaller, there is a reward. Another problem they encounter in the beginning is that there are plenty of different states, the puck's location,kart's velocity, kart's location, kart's front. The limitation is that they are using a q-table for this competition, so the size of the table will become extremely large. Weizhe thinks they should use a network to replace the q-table. However, The main problem is the designing of loss function. Where can they get the label if we want to return an action. Yilong thinks that they should firstly try to approximate the state value into nearest place. I.E. 3.66 to 3.6, then their table will become small. Then, They choose Yilong's Idea. After bulding the model and firstly train it, although it does not perform well, but it is learning and the opponent's score is not as high as previous. After they did some analysis on the recording, they found that the kart seems to follow certain pattern after few hundred episode. Weizhe proposes that the model should be given right to try more new things during first few episode, so he add a decay_factor for the action randomness factor and increase the action randomness factor in initial of training. Although they see karts has more option in action,however it does not imporve the final results. Then They try to modify the reward and try to add the final results to the reward system. Yilong thinks if the game is won,it should reward the fianl action to a very large number like 1000. Weizhe thinks that if the game is losed, some of the important actions should be punished. He try to write a memory list that contains previous n-steps actions and punish them all. However, it does not work pretty well. After careful thoughts and thinking, we think q learning is not good enough for the task, since its table is discrete and our task is continuous. Plus, we chunk our state to the nearest number which makes this situation even worse, I.E. we will do same action in the similar location,which will not fit to our complex environment. Then we try to investigate the DQN approach.

## Final decision 
After first 10 days, we found that group 1's planner performs really well. The only thing left is the Controller, Since Weizhe and Yilong are doing the state agent first so they have some experience with the controller, so they try to build the Controller for this planner. Santos try to improve his planner's performance to a furthurn steps. 

## Some Thoughts and Enlightment from this project
Santos:
Yilong:
Weizhe: I think I learned a lot, q_learning, DQN. I think the enlightment i got, especially when our team decide whether we should let the model to predict the world coordinate of the puck based on the image, although we did not take that advice, but wondering is the future can do those things or not, or it is already finished.

## Final words
Thank for professor Philipp Krähenbühl's amazing lecture and his amazing TAs' help (Christopher, Beiming, Nobel)! Have a nice holiday and Merry Chrismas!
