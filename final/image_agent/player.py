import math 
from .planner import Planner, load_model
import torchvision.transforms.functional as TF 
import numpy as np


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.Planner = load_model()
        self.Planner.eval()

        #Nov 28, 2021:

        

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

        return ['tux', 'kiki']

    def front_flag(self, puck_loc, threshold=2.0):
        #puck_loc => puck_loc -- model output

        x=puck_loc[0]
        return (x>(200-threshold)) and (x<(200+threshold))

    def act(self, player_state, player_image):
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
        print ("                   ENTERING ACT()    NOV 28 2021                  ")
        
       
        
        print ("LOADED MODULE")
        #aim_point_image = self.P(TF.to_tensor(player_image)[None]).squeeze(0).cpu().detach().numpy()
        
        aim_point_image_Player1 = self.Planner(TF.to_tensor(player_image[0])[None]).squeeze(0)
        aim_point_image_Player2 = self.Planner(TF.to_tensor(player_image[1])[None]).squeeze(0)

        #aim_point_image = aim_point_image.detach().cpu().numpy()  
 
        print ("---------------------------ACT() BLOCK BEGIN---------------------")
        #print(planner.forward(player_image))
        #print(planner)
        print ("\n these are the aim point coords for the ball: \n",  aim_point_image_Player1)
        print ("\n Front flag is--->", self.front_flag(aim_point_image_Player1) )

        #print ("\n Predicted coordinates are: \n -------------\n", x)
        
        #print("\n 1........ACT()  player_state[0][kart][location] FLOOR\n")
        #print(player_state[0]['kart']['location'])
        #print(f"\n    2.......ACT() player_state[0][kart][front] \n")
        #print (player_state[0]['kart']['front'])   
        #print(f"\n      3......ACT() player_state[0][camera][projection]\n")
        #print (player_state[0]['camera']['projection'])
        #print(f"\n         4......ACT() player_state[0][camera][view]\n")
        #print (player_state[0]['camera']['view'])
        print ("---------------------------ACT() BLOCK END---------------------")
        

        #these are the coords ONLY for player 1's view
        #these are the coords ONLY for player 1's view
        #these are the coords ONLY for player 1's view
        
        x1 =aim_point_image_Player1[0]     
        y1 = aim_point_image_Player1[1]     

        
        x2 = aim_point_image_Player2[0]     
        y2 = aim_point_image_Player2[1]     

        
        is_behind_1 = False
        is_behind_2 = False
        
        is_behind_1 = np.sign(y1.detach().numpy()) and abs(y1)>=1
        is_behind_2 = np.sign(y2.detach().numpy()) and abs(y2)>=1

        
        forward_drive =  dict(acceleration=1, steer=0, brake = False)
        backward_drive = dict(acceleration=0, steer=0, brake = True)
        turn_left = dict(acceleration=1, steer=-1, brake = False)
        turn_right =dict(acceleration=1, steer=1, brake = False)
   
        forward_aimpoint_1 = dict(acceleration=1, steer=x1, brake = False)
        backward_aimpoint_1 = dict(acceleration=0, steer=x1, brake = True)
        forward_aimpoint_2 = dict(acceleration=1, steer=x2, brake = False)
        backward_aimpoint_2 = dict(acceleration=0, steer=x2, brake = True)
      
        output1 = forward_aimpoint_1
        output2 = forward_aimpoint_2

        if is_behind_1:
          output1 =  backward_aimpoint_1
          print ("                          PUCK IS BEHIND<-------------------------")
          
        if is_behind_2:
          output2 =  backward_aimpoint_2
          print ("                          PUCK IS BEHIND<-------------------------")
          

         
        
        return [output1, output2]
