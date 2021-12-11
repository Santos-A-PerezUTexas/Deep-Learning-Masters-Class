import math 
from .planner import Planner, load_model
import torchvision.transforms.functional as TF 
import numpy as np
import torch
#aim_point_image_Player1  is Predicted from planner
#aim_point_image_Player2  is Predicted from planner
#aim_point_image_actual_1  is the *image* coord for *actual* soccer coords, temporary hack just for player 1
#xyz and xz  contains  actual soccer coords, hack
#x,y,z contain actual soccer coords as well


class Team:
    agent_type = 'image'

    def __init__(self):

        self.team = None
        self.num_players = None
        self.frame = 1
        self.forward_next = False

        self.planner = True     #set to true to use the planner, debugging purposes
        self.DEBUG = False      #SET TO TRUE TO DEBUG


        self.MSEloss = torch.nn.MSELoss()  #for DEBUGGING 
        self.total_loss_puck = 0           #for DEBUGGING
        self.total_loss_No_puck = 0        #for DEBUGGING
        self.total_loss_puck_count = 0     #for DEBUGGING
        self.total_loss_No_puck_count = 0  #for DEBUGGING
      

        #Dec 8, 2021

        self.team = None
        self.num_players = None
        self.current_team = 'not_sure'
        self.rescue_count = [0,0]
        self.rescue_steer = [1,1]
        self.recovery = [False,False]
        self.prev_loc = [[0,0],[0,0]]

        #Dec 8, 2021 (END)

        
        if not self.planner:
          print ("\n\n NOT USING PLANNER \n\n")

        if self.planner:

          print ("\n\n     Player (TEAM) INIT: USING PLANNER \n\n")
          
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
          self.Planner = load_model()
          self.Planner.eval()
          self.Planner = self.Planner.to(self.device)
          print (self.Planner)
          
        #self.prior_state = [] 
        #self.prior_soccer_state1 = []
        #self.prior_soccer_state2 = []
        
        if self.DEBUG:
          print ("\n\n DEBUG MODE IS ON \n\n")

        if self.DEBUG==False:
          print ("\n\n DEBUG MODE IS OFF \n\n")



    def new_match(self, team: int, num_players: int) -> list:

        self.team, self.num_players = team, num_players

        print ("\n\n new-match() was called STARTING NEW MATCH (message from player.py-newmatch() \n\n")
        #print (['tux']* num_players)

        return ['tux', 'tux']

    
    def to_numpy(self, location):
        return np.float32([location[0], location[2]])

    def _to_image300_400(self, coords, proj, view):
        W, H = 400, 300
        p = proj @ view @ np.array(list(coords) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    def _to_image(self, x, proj, view, normalization=True):  #FOR DEBUGGING

        out_of_frame = False
        op = np.array(list(x) + [1])
        p = proj @ view @ op
        x = p[0] / p[-1]   #p is [float, float, float, float]
        y = -p[1] / p[-1]
        aimpoint = np.array([x, y])

        if abs(x) > 1 or abs(y)>1:
          print ("Coordinate > 1, POSSIBLE PUCK NOT IN IMAGE")
          print (x,y)

        if normalization:
          print("NORMALIZING -1...1...........................NORMALIZING")
          aimpoint = np.clip(aimpoint, -1, 1) 

        if normalization == False:
          print ("NO -1...1 NORMALIZATION!!!!!!!!!!!!")
        
        return aimpoint

    def x_intersect(self, kart_loc, kart_front):
        slope = (kart_loc[1] - kart_front[1])/(kart_loc[0] - kart_front[0])
        intersect = kart_loc[1] - (slope*kart_loc[0])
        facing_up_grid = kart_front[1] > kart_loc[1]
        if slope == 0:
            x_intersect = kart_loc[1]
        else:
            if facing_up_grid:
                x_intersect = (65-intersect)/slope
            else:
                x_intersect = (-65-intersect)/slope
        return (x_intersect, facing_up_grid)


    def front_flag(self, puck_loc, threshold=2.0):
        #puck_loc => puck_loc -- model output

        x=puck_loc[0]
        return (x>(200-threshold)) and (x<(200+threshold))


    def model_controller(self, puck_loc, location,front,velocity,index):

        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}

        pos_me = location
        front_me = front
        kart_velocity = velocity
        velocity_mag = np.sqrt(kart_velocity[0] ** 2 + kart_velocity[2] ** 2)

        x = puck_loc[0]  # 0-400
        y = puck_loc[1]  # 0-300

        # clipping x and y values
        if x < 0:
            x = 0
        if x > 400:
            x = 400

        if y < 0:
            y = 0
        if y > 300:
            y = 300

        if self.current_team == 'not_sure':
            if -58 < pos_me[1] < -50:
                self.current_team = 'red'
            else:
                self.current_team = 'blue'
            print('Current Team:', self.current_team)

        x_intersect, facing_up_grid = self.x_intersect(pos_me, front_me)
        
        lean_val = 2
        if -10 < pos_me[0] < 10:
            lean_val = 0
        if facing_up_grid and 9 < x_intersect < 40:
            # if red team
            if self.current_team == 'red':
                x += lean_val
            else:
                x -= lean_val
        if facing_up_grid and -40 < x_intersect < -9:
            # if red team
            if self.current_team == 'red':
                x -= lean_val
            else:
                x += lean_val

        # facing inside goal
        if (not facing_up_grid) and 0 < x_intersect < 10:
            # if red team
            if self.current_team == 'red':
                x += lean_val
            else:
                x -= lean_val
        if (not facing_up_grid) and -10 < x_intersect < 0:
            # if red team
            if self.current_team == 'red':
                x -= lean_val
            else:
                x += lean_val

        if velocity_mag > 20:
            action['acceleration'] = 0.2

        if x < 200:
            action['steer'] = -1
        elif x > 200:
            action['steer'] = 1
        else:
            action['steer'] = 0

        if x < 50 or x > 350:
            action['drift'] = True
            action['acceleration'] = 0.2
        else:
            action['drift'] = False

        if x < 100 or x > 300:
            action['acceleration'] = 0.5

        if self.recovery[index] == True:
            action['steer'] = self.rescue_steer[index]
            action['acceleration'] = 0
            action['brake'] = True
            self.rescue_count[index] -= 2
            # print('rescue_count',self.rescue_count)
            # no rescue if initial condition
            if self.rescue_count[index] < 1 or ((-57 < pos_me[1] < 57 and -7 < pos_me[0] < 1) and velocity_mag < 5):
                self.rescue_count[index] = 0
                self.recovery[index] = False
        else:
            if self.prev_loc[index][0] == np.int32(pos_me)[0] and self.prev_loc[index][1] == np.int32(pos_me)[1]:
                self.rescue_count[index] += 5
            else:
                if self.recovery[index] == False:
                    self.rescue_count[index] = 0

            if self.rescue_count[index] < 2:
                if x < 200:
                    self.rescue_steer[index] = 1
                else:
                    self.rescue_steer[index] = -1
            if self.rescue_count[index] > 30 or (y > 200):
                # case of puck near bottom left/right
                if velocity_mag > 10:
                    self.rescue_count[index] = 30
                    self.rescue_steer[index] = 0
                else:
                    self.rescue_count[index] = 20
                self.recovery[index] = True

        self.prev_loc[index] = np.int32(pos_me)

        return action


    def get_instance_coords (instance, object=8):

      x=1
      y=1
      Flag = False

      for i in range (300):
        for j in range (400):
          if instance[i][j]  == object:
            x = i
            y = j
            Flag = True
            return Flag, x, y

      return Flag, x,y

    def act(self, player_state, player_image, soccer_state = None, heatmap1=None, heatmap2=None):  #REMOVE SOCCER STATE!!!!!!!!
        
        use_soccer_world_coords = False  #USES SOCCER COORDS
        use_actual_coords = False    #USES ACTUAL COORDS FOR SOCCER BALL *ACTION*
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

          if self.frame >= 30:  #call the planner
            
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            

            
            aim_point_image_Player1 = self.Planner(image1)
            aim_point_image_Player2 = self.Planner(image2)
            aim_point_image_Player1 = aim_point_image_Player1.squeeze(0)
            aim_point_image_Player2 = aim_point_image_Player2.squeeze(0)
            
            #aim_point_image_Player1 = aim_point_image_Player1.detach().cpu().numpy()
            #aim_point_image_Player2 = aim_point_image_Player2.detach().cpu().numpy()

          if self.frame < 30:   #do not call planner, soccer cord is likely [0,0]
            
            proj1 = np.array(player_state[0]['camera']['projection']).T
            view1 = np.array(player_state[0]['camera']['view']).T
            proj2 = np.array(player_state[1]['camera']['projection']).T
            view2 = np.array(player_state[1]['camera']['view']).T
            x = np.float32([0,0,0]) 
            aim_point_image_Player1 = self._to_image300_400(x, proj1, view1) 
            aim_point_image_Player2 = self._to_image300_400(x, proj2, view2)
            
                    
          x1 = aim_point_image_Player1[0]     
          y1 = aim_point_image_Player1[1]     
          x2 = aim_point_image_Player2[0]     
          y2 = aim_point_image_Player2[1]
          
          
          #self.prior_soccer_state1.append(aim_point_image_Player1)
          #self.prior_soccer_state2.append(aim_point_image_Player2)
        
        #self.prior_state.append(player_state)
        
        if not self.planner:

          #use random points, for debugging or testing, -1...1 coordinates, not 300/400          
          x1 = x2 =  1     
          y1 = y2 = -1     
               
          
        is_behind_1 = False
        is_behind_2 = False
        
        if y1 >= 1:
          is_behind_1 = True

        if y2 >= 1:
          is_behind_2 = True

        xyz = np.random.rand(3)
        xz =  np.random.rand(2)
        
        xyz[0] = 1
        xyz[1] = 1
        xyz[2] = 1

        if self.DEBUG:   #use soccer state only for debbugging 

          xyz[0] =soccer_state['ball']['location'][0]
          xyz[1] =soccer_state['ball']['location'][1] 
          xyz[2] =soccer_state['ball']['location'][2]
        
        
        xz[0] = xyz[0]
        xz[1] = xyz[2]
                
        proj = np.array(player_state[0]['camera']['projection']).T
        view = np.array(player_state[0]['camera']['view']).T
        

        if use_image_coords and self.DEBUG:
          print ("USING  IMAGE COORDS FOR PUCK ACTUAL COORDS  HACK")
          aim_point_image_actual_1 = self._to_image300_400(xyz, proj, view) 
        if use_soccer_world_coords and self.DEBUG:
          print("USING IMAGE COORDS FOR PUCK ACTUAL COORDS HACK")
          aim_point_image_actual_1 = self._to_image300_400(xyz, proj, view)

        if self.DEBUG:
          if aim_point_image_actual_1[0] < 0:
            aim_point_image_actual_1[0] = 0
          if aim_point_image_actual_1[0] > 400:
            aim_point_image_actual_1[0] = 400

          if aim_point_image_actual_1[1] < 0:
            aim_point_image_actual_1[1] = 0
          if aim_point_image_actual_1[1] > 300:
            aim_point_image_actual_1[1] = 300
          
          print("\n\n Player 1~~~~~~~~~~~ aimpoint predicted, aimpoint actual:", aim_point_image_Player1, 
               aim_point_image_actual_1)
          print("\nThe pure world socccer coords and frame are:  ", xyz, self.frame)
          if (xz[0] == 0) and (xz[0]==1):
            print ("\n\n\n *** ZERO COORDS AT FRAME ***", self.frame)
         
        
        puck_flag = 0


        if heatmap1 and self.DEBUG:
          print ("\n\n\ DOING BITSHIFT ON INSTANCE \n\n")
          heatmap1[0] = heatmap1[0] >> 24
          for i in range (300):
            for j in range (400):
              if heatmap1[0][i][j]  == 8:
                puck_flag = 1

          
          #aim_point_image_Player1  is Predicted from planner

          ##xz has the actual (hacked) coords for the soccer ball. must convert to 300/400


          if self.frame >= 30 and self.DEBUG:
            xz_image = self._to_image300_400(xyz, proj, view) 
            detached_p = aim_point_image_Player1.detach().cpu().numpy()
            #loss_v_image = 0
            loss_v_image = torch.mean(torch.abs(aim_point_image_Player1-xz_image))     #.mean()
          if self.frame < 30 and self.DEBUG:
            loss_v_image = abs(aim_point_image_Player1-xz).mean()

          if self.DEBUG and self.frame >= 30:
            print("\nPlayer 1~~~~~~~~~~~ CURRENT LOSS predicted/image coords and frame is", loss_v_image, self.frame)

            loss_v_world = abs(detached_p-xz_image).mean()
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

        if self.frame > 500 and self.DEBUG:
          print ("\n\n STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS STATS ")
          print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS NO PUCK", self.total_loss_No_puck/self.total_loss_No_puck_count)
          print("\nPlayer 1~~~~~~~~~~~RUNNING AVERAGE LOSS FOR PUCK *IN IMAGE*", self.total_loss_puck/self.total_loss_puck_count)
          #print ("\n\n THESE ARE THE MINX, MAXX, MINY, MAXY:", self.min_x, self.max_x, self.min_y, self.max_y)
          print ("-----------------------------------------------------------------------------------")
       
        Kart_A_front = player_state[0]['kart']['front']
        Kart_A_location = player_state[0]['kart']['location']
        Kart_A_vel = player_state[0]['kart']['velocity']
        pos_A = self.to_numpy(Kart_A_location)
        front_A =self.to_numpy(Kart_A_front)
        
        Kart_B_front = player_state[1]['kart']['front']
        Kart_B_location = player_state[1]['kart']['location']
        Kart_B_vel = player_state[0]['kart']['velocity']
        pos_B = self.to_numpy(Kart_B_location)
        front_B = self.to_numpy(Kart_B_front)

        
        #Dec 10, 2021:
        
        #Call the planner only if self.Frames > 40 or so:
        """
        if self.frame < 40:
            action_A = self.model_controller(np.float32([0,0]),pos_A,front_A,Kart_A_vel,0)
            action_B = self.model_controller(np.float32([0,0]),pos_B,front_B,Kart_B_vel,1)

        if Self.frame>40:
            aim_point_image_Player1 = self.Planner(image1)
            aim_point_image_Player2 = self.Planner(image2)
            action_A = self.model_controller(aim_point_image_Player1,pos_A,front_A,Kart_A_vel,0)
            action_B = self.model_controller(aim_point_image_Player2,pos_B,front_B,Kart_B_vel,1)
        
        """
        
        action_A = self.model_controller(aim_point_image_Player1,pos_A,front_A,Kart_A_vel,0)
        action_B = self.model_controller(aim_point_image_Player2,pos_B,front_B,Kart_B_vel,1)

        #action_A = self.model_controller(np.float32([0,0]),pos_A,front_A,Kart_A_vel,0)
        #action_B = self.model_controller(np.float32([0,0]),pos_B,front_B,Kart_B_vel,1)



        ret = [action_A,action_B]

        return ret
