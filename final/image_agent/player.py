import math 
from .planner import Planner, load_model
import torchvision.transforms.functional as TF 
import numpy as np
import torch


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
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
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """


        """
           TODO: feel free to edit or delete any of the code below
        """
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
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0....1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional, unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        #he agent also sees an image for each player, from which it can infer where the puck is and
        # where the other teams players are.
        #comment Nov 28, 2021 6:40pm
        # TODO: Change me. I'm just cruising straight  Changed
        #print (player_state[0]['kart']['location'])
        #[dict(acceleration=1, steer=-.2, nitro=True, fire=True)] * self.num_players
        
        use_soccer_world_coords = True

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

        x=soccer_state['ball']['location'][0]
        y = soccer_state['ball']['location'][1] 
        z=soccer_state['ball']['location'][2]
        xyz = np.random.rand(3)
        xz =  np.random.rand(2)
        xz[0] = x
        xz[1] = z
        xyz[0] = x
        xyz[1] = y
        xyz[2] = z
        proj = np.array(player_state[0]['camera']['projection']).T
        view = np.array(player_state[0]['camera']['view']).T
        if use_soccer_world_coords == False:
          aim_point_image_actual_1 = self._to_image(xyz, proj, view) 
        if use_soccer_world_coords:
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

        use_actual_coords = False    #OMIT THIS DEC 1 2021

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



        #print ("\n ^^^^^^^^^^^^^^^ VELOCITIES^^^^^^^^^^^^^^^^ \n")
        #print (player_state[0]['kart']['velocity'])
        #print (player_state[1]['kart']['velocity'])

        #print ("\n ^^^^^^^^^^^^^^^ KART FRONT COORDS^^^^^^^^^^^^^^^^ \n")
        #print (player_state[0]['kart']['front'])
        #print (player_state[1]['kart']['front'])


        #print ("\n ^^^^^^^^^^^^^^^ KART COORDS^^^^^^^^^^^^^^^^ \n")

        #print (player_state[0]['kart']['location'])
        #print (player_state[1]['kart']['location'])


        #print ("\n ^^^^^^^^^^^^^^^^ KART ROTATION^^^^^^^^^^^^^^^^ \n")

        #print (player_state[0]['kart']['rotation'])
        #print (player_state[1]['kart']['rotation'])


        #print ("\n ^^^^^^^^^^^^^^^^ PLAYER ID, TEAM^^^^^^^^^^^^^^^^ \n")

        #print (player_state[0]['kart']['player_id'], self.my_team)
        #print (player_state[1]['kart']['id'], self.my_team)

        #direction kart is facing is player_state[0]['kart']['location'][2] -
        # minus player_state[0]['kart']['front'][2], different for both teams. 


        #print ("\n ^^^^^^^^^^^^^^^ CURRENT KART DIRECTIONS^^^^^^^^^^^^^^^^ \n")

        #print ("\n Kart 1 Current Direction:  ", current_kart1_dir)
        #print ("\n Kart 2 Current Direction:  ", current_kart2_dir)
      
        #if wrong_direction_kart1:
         # print ("\n DANGER ----------- Kart 1 is headed in the wrong direction!")

        #if wrong_direction_kart2:
          #print ("\n DANGER ----------- Kart 2 is headed in the wrong direction!")

      #Case 1, right direction, puck in frame
      #Case 2, right direction, puck not in frame
      #Case 3, wrong direction, puck in frame
      #Case 4, wrong direction, puck not in frame
      #Sub Cases:  Puck behind, to the left, or to the right, how to know?





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
