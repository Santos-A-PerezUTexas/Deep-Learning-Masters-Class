import math 
from .planner import Planner, load_model
import torchvision.transforms.functional as TF 
import numpy as np
import torch


class Team:
    agent_type = 'image'

    def __init__(self):

        self.team = None
        self.num_players = None
        self.frame = 1
        self.forward_next = False

        self.planner = False
        self.MSEloss = torch.nn.MSELoss()
        self.total_loss_puck = 0
        self.total_loss_No_puck = 0
        self.total_loss_puck_count = 0
        self.total_loss_No_puck_count = 0
         
        
        if not self.planner:
          print ("\n\n NOT USING PLANNER \n\n")
          
        if self.planner:

          print ("\n\n     Player (TEAM) INIT: USING PLANNER \n\n")
          self.Planner = load_model()
          self.Planner.eval()
          
        self.prior_state = []
        self.prior_soccer_state1 = []
        self.prior_soccer_state2 = []
        self.DEBUG = False
        
        if self.DEBUG:
          print ("\n\n DEBUG MODE IS ON \n\n")

        if self.DEBUG==False:
          print ("\n\n DEBUG MODE IS OFF \n\n")

    def new_match(self, team: int, num_players: int) -> list:

        self.team, self.num_players = team, num_players

        print ("\n\n new-match() was called STARTING NEW MATCH (message from player.py-newmatch() \n\n")
        #print (['tux']* num_players)

        return ['tux', 'tux']

    
    def _to_image(self, x, proj, view, normalization=True):  #DEC 1, 2021: ERASE THIS..................

        out_of_frame = False
        op = np.array(list(x) + [1])
        p = proj @ view @ op
        x = p[0] / p[-1]   #p is [float, float, float, float]
        y = -p[1] / p[-1]
        
        aimpoint = np.array([x, y])

        if abs(x) > 1 or abs(y)>1:
          print ("NOTE-------------------------------------------------------->We got a coordinate > 1, OUT_OF_FRAME TRUE")
          print (x,y)

        if normalization:
          print("NORMALIZING -1...1...........................NORMALIZING")
          aimpoint = np.clip(aimpoint, -1, 1) 

        if normalization == False:
          print ("NO -1...1 NORMALIZATION!!!!!!!!!!!!")
        
        
        return aimpoint
            
    def front_flag(self, puck_loc, threshold=2.0):
        #puck_loc => puck_loc -- model output

        x=puck_loc[0]
        return (x>(200-threshold)) and (x<(200+threshold))





    def act(self, player_state, player_image, soccer_state = None, heatmap1=None, heatmap2=None):  #REMOVE SOCCER STATE!!!!!!!!
        
        use_soccer_world_coords = True  #USES SOCCER COORDS
        use_actual_coords = False    #USES ACTUAL COORDS FOR SOCCER BALL ACTION
        use_image_coords = True

        #Dec 7 2021:
        action_P1 = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        action_P2 = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}




        if use_soccer_world_coords:
          use_image_coords = False

        self.my_team = player_state[0]['kart']['player_id']%2

        goal_post = [[-10.449999809265137, 0.07000000029802322, -64.5], 
                     [10.449999809265137, 0.07000000029802322, -64.5]] 

        goal_width = goal_post[1][0]-goal_post[0][0]  
        y_goal = -64.5


        if self.my_team == 0:  #RED TEAM
          goal_post = [[10.460000038146973, 0.07000000029802322, 64.5], 
                     [-10.510000228881836, 0.07000000029802322, 64.5]] 
          y_goal = 64.5

        current_kart1_dir = -64.5
        current_kart2_dir  = -64.5

        if (player_state[0]['kart']['front'][2] - player_state[0]['kart']['location'][2]) > 0:
          current_kart1_dir = 64.5 
        
        if (player_state[1]['kart']['front'][2] - player_state[1]['kart']['location'][2]) > 0:
          current_kart2_dir = 64.5 
                
        wrong_direction_kart1 = False
        wrong_direction_kart2 = False

        if (current_kart1_dir != y_goal):
          wrong_direction_kart1 = True

        if (current_kart2_dir != y_goal):
          wrong_direction_kart2 = True
 
              
        if self.planner:
          
          image1 = TF.to_tensor(player_image[0])[None]
          image2 = TF.to_tensor(player_image[1])[None]
          
          #aim_point_image_Player1, _ = self.Planner(image1)
          #aim_point_image_Player2, _ = self.Planner(image2)
          aim_point_image_Player1 = self.Planner(image1)
          aim_point_image_Player2 = self.Planner(image2)

          aim_point_image_Player1 = aim_point_image_Player1.squeeze(0)
          aim_point_image_Player2 = aim_point_image_Player2.squeeze(0)
          
          x1 = aim_point_image_Player1[0]     
          y1 = aim_point_image_Player1[1]     
          x2 = aim_point_image_Player2[0]     
          y2 = aim_point_image_Player2[1]
          
          
          #self.prior_soccer_state1.append(aim_point_image_Player1)
          #self.prior_soccer_state2.append(aim_point_image_Player2)
        
        #self.prior_state.append(player_state)
        
        if not self.planner:

          
          x1 = 1     
          y1 = -1     
          x2 = 1     
          y2 = -1     
          
        is_behind_1 = False
        is_behind_2 = False
        
        if y1 >= 1:
          is_behind_1 = True

        if y2 >= 1:
          is_behind_2 = True

        
        #ERASE BEGIN DEC 1, 2021-----------------------------------------

        x = 1
        y = 1
        z = 1

        if self.DEBUG:
          x =soccer_state['ball']['location'][0]
          y =soccer_state['ball']['location'][1] 
          z =soccer_state['ball']['location'][2]
        
        

        xyz = np.random.rand(3)
        xz =  np.random.rand(2)
        
        xz[0] = x
        xz[1] = z
        
        xyz[0] = x
        xyz[1] = y
        xyz[2] = z
        
        proj = np.array(player_state[0]['camera']['projection']).T
        view = np.array(player_state[0]['camera']['view']).T

        if use_image_coords and self.DEBUG:
          print ("USING NORMALIZED IMAGE COORDS FOR PUCK ACTUAL")
          aim_point_image_actual_1 = self._to_image(xyz, proj, view, normalization=True) 
        if use_soccer_world_coords and self.DEBUG:
          print("USING UNNORMALED IMAGE COORDS FOR PUCK ACTUAL")
          aim_point_image_actual_1 = self._to_image(xyz, proj, view, normalization=False)

        if self.DEBUG:
          print("\n\n Player 1~~~~~~~~~~~ aimpoint predicted, aimpoint actual:", aim_point_image_Player1, 
               aim_point_image_actual_1)
        
        if self.DEBUG:
          print("\nThe pure world socccer coords are:  ", xz)

        
        puck_flag = 0


        if heatmap1 and self.DEBUG:
          print ("\n\n\ DOING BITSHIFT ON INSTANCE \n\n")
          heatmap1[0] = heatmap1[0] >> 24
          for i in range (300):
            for j in range (400):
              if heatmap1[0][i][j]  == 8:
                puck_flag = 1

          loss_v_image = abs(aim_point_image_Player1.detach()-xz).mean()
          
          if self.DEBUG:
            print("\nPlayer 1~~~~~~~~~~~ CURRENT LOSS predicted/image coords and frame is", loss_v_image, self.frame)

            loss_v_world = abs(aim_point_image_Player1.detach()-xz).mean()
            print("\nPlayer 1~~~~~~~~~~~ CURRENT LOSS predicted/world, and frame is", loss_v_world, self.frame)

            if puck_flag:
              print("\n    *******THERE IS A PUCK IN THE IMAGE!!!!!!!!!!!!!!!!! <-------------")
         
              self.total_loss_puck += loss_v_world          
              self.total_loss_puck_count += 1
              print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS (WORLD COORDS) FOR PUCK *IN IMAGE*", self.total_loss_puck/self.total_loss_puck_count)
              if self.total_loss_No_puck_count > 0:
                print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS (WORLD COORDS)  NO PUCK", self.total_loss_No_puck/self.total_loss_No_puck_count)


          if puck_flag==0 and self.DEBUG:
            print ("\n   WARNING:   NO PUCK IN IMAGE...................................]]]]]]]]]]")
            self.total_loss_No_puck += loss_v_world 
            self.total_loss_No_puck_count += 1
            print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS (WORLD) NO PUCK", self.total_loss_No_puck/self.total_loss_No_puck_count)  
            if self.total_loss_puck_count >0:
              print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS (WORLD) FOR PUCK *IN IMAGE*", self.total_loss_puck/self.total_loss_puck_count)

        

        if use_actual_coords:
          x1 = aim_point_image_actual_1[0]     
          y1 = aim_point_image_actual_1[1]
        

        #ERASE END DEC 1, 2021-------------------------------------------




        forward_drive =  dict(acceleration=1, steer=0, brake = False)
        backward_drive = dict(acceleration=0, steer=0, brake = True)
        turn_left = dict(acceleration=1, steer=-1, brake = False)
        turn_right =dict(acceleration=1, steer=1, brake = False)
   
        forward_aimpoint_1 = dict(acceleration=1, steer=x1, brake = False, drift=True)
        backward_aimpoint_1 = dict(acceleration=0, steer=x1, brake = True)
        forward_aimpoint_2 = dict(acceleration=1, steer=x2, brake = False, drift=True)
        backward_aimpoint_2 = dict(acceleration=0, steer=x2, brake = True)

        goal_aim_point = dict(acceleration=0, steer=x2, brake = True)
      
        
        output1 = forward_aimpoint_1
        output2 = forward_aimpoint_2


        if player_state[0]['kart']['velocity'][2] > 5:
          output1 = backward_aimpoint_1


        if player_state[0]['kart']['velocity'][2] > 3:
          output2 = backward_aimpoint_2

        self.frame += 1

        if self.frame > 150 and self.DEBUG:
          print ("\n\n STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS ")
          print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS NO PUCK", self.total_loss_No_puck/self.total_loss_No_puck_count)
          print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS FOR PUCK *IN IMAGE*", self.total_loss_puck/self.total_loss_puck_count)
          print ("-----------------------------------------------------------------------------------")
       
        return [output1, output2]
