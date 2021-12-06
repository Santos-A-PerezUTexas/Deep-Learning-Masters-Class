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

        self.planner = True
        self.MSEloss = torch.nn.MSELoss()
        self.total_loss_puck = 0
        self.total_loss_No_puck = 0
        self.total_loss_puck_count = 0
        self.total_loss_No_puck_count = 0
         
        

        if self.planner:

          print ("IN CONDITIONAL")
          self.Planner = load_model()
          print ("LOADED PLANNER")
          self.Planner.eval()
          print ("LOADED PLANNER EVAL")

        self.prior_state = []
        self.prior_soccer_state1 = []
        self.prior_soccer_state2 = []
        self.DEBUG = True


    def new_match(self, team: int, num_players: int) -> list:

        self.team, self.num_players = team, num_players

        #print (['tux']* num_players)

        return ['tux', 'tux']

    
    def _to_image(self, x, proj, view):  #DEC 1, 2021: ERASE THIS..................

        out_of_frame = False
        op = np.array(list(x) + [1])
        p = proj @ view @ op
        x = p[0] / p[-1]   #p is [float, float, float, float]
        y = -p[1] / p[-1]
        
        aimpoint = np.array([x, y])

        clipped_aim_point = np.clip(aimpoint, -1, 1) 
        
        
        return clipped_aim_point
            
    def front_flag(self, puck_loc, threshold=2.0):
        #puck_loc => puck_loc -- model output

        x=puck_loc[0]
        return (x>(200-threshold)) and (x<(200+threshold))





    def act(self, player_state, player_image, soccer_state = None, heatmap1=None, heatmap2=None):  #REMOVE SOCCER STATE!!!!!!!!
        
        use_soccer_world_coords = True  #USES SOCCER COORDS
        use_actual_coords = False    #USES ACTUAL COORDS FOR SOCCER BALL ACTION
        use_image_coords = True

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
 

    
        print ("\n---------------------------ACT() BLOCK BEGIN---------------------")
              
        if self.planner:
          
          print ("\nUSING PLANNER USING PLANNER USING PLANNER USING PLANNER ")

          image1 = TF.to_tensor(player_image[0])[None]
          image2 = TF.to_tensor(player_image[1])[None]
          print ("LINE 175")
          #aim_point_image_Player1, _ = self.Planner(image1)
          #aim_point_image_Player2, _ = self.Planner(image2)
          aim_point_image_Player1 = self.Planner(image1)
          aim_point_image_Player2 = self.Planner(image2)

          print ("LINE 180")
          aim_point_image_Player1 = aim_point_image_Player1.squeeze(0)
          aim_point_image_Player2 = aim_point_image_Player2.squeeze(0)
          
          x1 = aim_point_image_Player1[0]     
          y1 = aim_point_image_Player1[1]     
          x2 = aim_point_image_Player2[0]     
          y2 = aim_point_image_Player2[1]
          
          
          #self.prior_soccer_state1.append(aim_point_image_Player1)
          #self.prior_soccer_state2.append(aim_point_image_Player2)
          print ("LINE 192")
        
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

        if use_image_coords:
          print ("USING IMAGE COORDS FOR PUCK ACTUAL")
          aim_point_image_actual_1 = self._to_image(xyz, proj, view) 
        if use_soccer_world_coords:
          print("USING WORLD COORDS FOR PUCK ACTUAL")
          aim_point_image_actual_1 = xz

        print("\n\n Player 1~~~~~~~~~~~ aimpoint predicted, aimpoint actual:", aim_point_image_Player1, 
               aim_point_image_actual_1)

        #print("\n~~~~~~~~~~~ LOSS:", self.MSEloss())

        #x = aim_point_image_Player1.detach()

        #loss_v = self.MSEloss(x, aim_point_image_actual_1)
     
        puck_flag = 0


        if heatmap1 and self.DEBUG:
          heatmap1[0] = heatmap1[0] >> 24
          for i in range (300):
            for j in range (400):
              if heatmap1[0][i][j]  == 8:
                puck_flag = 1

          loss_v = abs(aim_point_image_Player1.detach()-aim_point_image_actual_1).mean()

          print("\nPlayer 1~~~~~~~~~~~ CURRENT LOSS, and frame", loss_v, self.frame)

          if puck_flag:
            print("\n    *******THERE IS A PUCK IN THE IMAGE!!!!!!!!!!!!!!!!! <-------------")
         
            self.total_loss_puck += loss_v          
            self.total_loss_puck_count += 1
            print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS FOR PUCK *IN IMAGE*", self.total_loss_puck/self.total_loss_puck_count)
            if self.total_loss_No_puck_count > 0:
              print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS NO PUCK", self.total_loss_No_puck/self.total_loss_No_puck_count)


          if puck_flag==0:
            print ("\n   WARNING:   NO PUCK IN IMAGE...................................]]]]]]]]]]")
            self.total_loss_No_puck += loss_v 
            self.total_loss_No_puck_count += 1
            print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS NO PUCK", self.total_loss_No_puck/self.total_loss_No_puck_count)  
            if self.total_loss_puck_count >0:
              print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS FOR PUCK *IN IMAGE*", self.total_loss_puck/self.total_loss_puck_count)

        

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
