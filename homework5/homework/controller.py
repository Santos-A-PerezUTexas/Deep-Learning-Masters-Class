import pystk
import numpy as np


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller

    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    
    The point (-1,-1) is the top left of the screen and (1, 1) the bottom right.
    The aim point is a point on the center of the track 15 meters away

    pystk.Action.steer the steering angle of the kart normalized to -1 … 1
    pystk.Action.acceleration the acceleration of the kart normalized to 0 … 1
    pystk.Action.brake boolean indicator for braking
    pystk.Action.drift a special action that makes the kart drift, useful for tight turns
    pystk.Action.nitro burns nitro for fast acceleration


    https://piazza.com/class/ksjhagmd59d6sg?cid=975
    How do you interpret velocity in the game? Is it X meters per frame, which is a step in the game?

    https://piazza.com/class/ksjhagmd59d6sg?cid=805
    Hi, I just have a question on the steering function. I assume that -1 will turn the wheel "all the way"
    to the left (45 degrees to the left from the [0,0] vector) and 1 will steer it to the right by 45 degrees?
    The standard "max speed" is around 22-23 if holding straight and accelerator set to 1.

    """
    action = pystk.Action()

    #Steering and relative aim point use different units. Use the aim point 
    #and a tuned scaling factor to select the amount of normalized steering.
    
    action.steer = aim_point[0]  
    action.brake = False
    action.nitro = True
    action.acceleration = 1
    
    if aim_point[1] < 0:
      t=1
      action.nitro = True
      #print ("Facing Front")

    if aim_point[1] > 0:
      t=1
      action.acceleration = 0
      #print ("LOOK BEHIND YOU")

    if abs(aim_point[0])<.2:
      action.nitro = True         #NITROOOOOOOO
      #print("NITRO")

    #if abs(aim_point[0])<.2:
     # action.acceleration = 100

    #print (current_vel)
    if abs(aim_point[0])<=.5:
      action.drift = False
    
    #comment 
    
    direction_steer = np.sign(aim_point[0])

    if abs(aim_point[0])>.9:
      action.steer = (abs(aim_point[0])*direction_steer)
      action.acceleration = 0
      action.acceleration = False
      #action.nitro = False

    if (abs(aim_point[0])>.45) and (abs(aim_point[0])<=.7):
      action.drift = True
      action.acceleration = 0.1  #.1

    if (abs(aim_point[0])>.7):
      #print (f'SLIGHT tight curve ahead, speed is {current_vel}, steering at {aim_point[0]}, acceleration {action.acceleration}, brake {action.brake}')
      action.drift = True
      action.acceleration = 0.00
      #action.nitro = False 
      
   
    if (abs(aim_point[0])>.7) and current_vel > 15:
      action.brake = True 
      action.nitro = False
      #print (f'tight curve ahead, speed is {current_vel}, steering at {aim_point[0]}, acceleration {action.acceleration}, brake {action.brake}')

    if current_vel > 23:    #24
      action.brake = True
      #action.nitro = False

    """
    Your code here
    https://piazza.com/class/ksjhagmd59d6sg?cid=358

    Hint: Skid if the steering angle is too large.
    Hint: Target a constant velocity.
    
    Hint: Steering and relative aim point use different units. Use the aim point and a tuned scaling factor
    to select the amount of normalized steering.
    
    Hint: Make sure that your controller is able to complete all levels before proceeding to the next part of 
          the homework because you will use your controller to build the training set for your planner.
          
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)

    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)

    """

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
