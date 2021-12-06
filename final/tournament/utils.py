import numpy as np
from enum import IntEnum
import pystk 
#TOURNAMENT UTILS
#Edit this to generate labels

file_no = 1001 

class Team(IntEnum):   #Two methods, video_grid() and map_image() (as well as map_image()-->_to_coord(x) 
    
    RED = 0
    BLUE = 1


def video_grid(team1_images, team2_images, team1_state='', team2_state=''):  #class Team
    from PIL import Image, ImageDraw
    grid = np.hstack((np.vstack(team1_images), np.vstack(team2_images)))
    grid = Image.fromarray(grid)
    grid = grid.resize((grid.width // 2, grid.height // 2))

    draw = ImageDraw.Draw(grid)
    draw.text((20, 20), team1_state, fill=(255, 0, 0))
    draw.text((20, grid.height // 2 + 20), team2_state, fill=(0, 0, 255))
    return grid


def map_image(team1_state, team2_state, soccer_state, resolution=512, extent=65, anti_alias=1):  #class Team
    BG_COLOR = (0xee, 0xee, 0xec)
    RED_COLOR = (0xa4, 0x00, 0x00)
    BLUE_COLOR = (0x20, 0x4a, 0x87)
    BALL_COLOR = (0x2e, 0x34, 0x36)
    from PIL import Image, ImageDraw
    r = Image.new('RGB', (resolution*anti_alias, resolution*anti_alias), BG_COLOR)

    def _to_coord(x):                                                           #class Team-->def map_image()---->def _to_coord(x)
        return resolution * anti_alias * (x + extent) / (2 * extent)

    
    #--------------------------------------CLASS TEAM-->def map_image() BEGIN  -----------------------------------------------#
    
    draw = ImageDraw.Draw(r)
    # Let's draw the goal line
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][0]], width=5*anti_alias, fill=RED_COLOR)
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][1]], width=5*anti_alias, fill=BLUE_COLOR)

    # and the ball
    x, _, y = soccer_state['ball']['location']
    s = soccer_state['ball']['size']
    draw.ellipse((_to_coord(x-s), _to_coord(y-s), _to_coord(x+s), _to_coord(y+s)), width=2*anti_alias, fill=BALL_COLOR)

    # and karts
    for c, s in [(BLUE_COLOR, team1_state), (RED_COLOR, team2_state)]:
        for k in s:
            x, _, y = k['kart']['location']
            fx, _, fy = k['kart']['front']
            sx, _, sy = k['kart']['size']
            s = (sx+sy) / 2
            draw.ellipse((_to_coord(x - s), _to_coord(y - s), _to_coord(x + s), _to_coord(y + s)), width=5*anti_alias, fill=c)
            draw.line((_to_coord(x), _to_coord(y), _to_coord(x+(fx-x)*2), _to_coord(y+(fy-y)*2)), width=4*anti_alias, fill=0)

    if anti_alias == 1:
        return r
    return r.resize((resolution, resolution), resample=Image.ANTIALIAS)


# Recording functionality
class BaseRecorder:
    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        raise NotImplementedError

    def __and__(self, other):
        return MultiRecorder(self, other)

    def __rand__(self, other):
        return MultiRecorder(self, other)


class MultiRecorder(BaseRecorder):
    def __init__(self, *recorders):
        self._r = [r for r in recorders if r]

    def __call__(self, *args, **kwargs):
        for r in self._r:
            r(*args, **kwargs)


class VideoRecorder(BaseRecorder):
    """
        Produces pretty output videos
    """
    def __init__(self, video_file):
        import imageio
        self._writer = imageio.get_writer(video_file, fps=20)
        self.divide_data = False

    def __call__(self, team1_state, team2_state, soccer_state, actions,
                 team1_images=None, team2_images=None, heatmap1=None, heatmap2=None):

        if team1_images and team2_images:
            
            out_of_frame = False
            is_behind = False
            use_world = True
            

            #print ("\n VideoRecorder() in utils.py ----- Putting Images in Grid, ball location: \n")
            #print(tuple(soccer_state['ball']['location']))
           
            #convert/normalize coordinates!!!!!!!!!!!!!!!!!!!!!! 
            x=soccer_state['ball']['location'][0]
            y = soccer_state['ball']['location'][1] 
            z=soccer_state['ball']['location'][2]
            xyz = np.random.rand(3)
            xz=np.random.rand(2)
            xz[0] = x
            xz[1] = z
            xyz[0] = x
            xyz[1] = y
            xyz[2] = z
            #self.collect(team1_images[0], soccer_state['ball']['location'])
            
            
            proj = np.array(team1_state[0]['camera']['projection']).T
            view = np.array(team1_state[0]['camera']['view']).T
            #print (f'the view is {view.shape}, the proj is {proj.shape}')
                        
            aim_point_image, out_of_frame = self._to_image(xyz, proj, view)  #normalize xz in range -1...1
            #print (f'the aim_point_image is {aim_point_image}')
            
            #NOTE:  TEST FOR THE CASE WHERE PUCK IS OFF FRAME!  WHAT LABEL???
            #NOTE:  TEST FOR THE CASE WHERE PUCK IS OFF FRAME!  WHAT LABEL???
            #NOTE:  TEST FOR THE CASE WHERE PUCK IS OFF FRAME!  WHAT LABEL???
            #NOTE:  TEST FOR THE CASE WHERE PUCK IS OFF FRAME!  WHAT LABEL???
            
            #heatmap = pystk.RenderData(team1_images[0]) 
              
            if heatmap1:
              
              
              heatmap1[0] = heatmap1[0] >> 24

              puck_flag = 0
              for i in range (300):
                for j in range (400):
                  if heatmap1[0][i][j]  == 8:
                    puck_flag = 1
              
            if puck_flag:
              print ("\n Found Puck")
              

            if not puck_flag:
               print ("\n You're out of Puck")

            if puck_flag != out_of_frame:
              print ("\n WARNING -   puck_flag bit shift does not coincide with  out of frame coord flag")
              print ("\n WARNING -   puck_flag bit shift does not coincide with  out of frame coord flag")


              #heatmap1[1] = heatmap1[1] >> 24
              #heatmap2 = heatmap2 >> 24
              #print ("\n==============heatmap shape is  ", heatmap1[0].shape)
              #print ("\n==============image shape is  ", team1_images[0].shape)
              

            if use_world == False:
              self.collect(team1_images[0], puck_flag, aim_point_image)
            
            if use_world:
              self.collect(team1_images[0], puck_flag, xz)

            #self.collect(team1_images[0], xz)  #updated to above on 11/27/2021 to normalize xz in range -1...1
            
        
            self._writer.append_data(np.array(video_grid(team1_images, team2_images,
                                                        'X1 Ball  Location: %f' % aim_point_image[0],
                                                        'Y1 Ball  Location: %f' % aim_point_image[1])))
            #self._writer.append_data(np.array(video_grid(team1_images, team2_images,
             #                                            'Utexas Blue: %d' % soccer_state['score'][1],
              #                                           'Utexas Red: %d' % soccer_state['score'][0])))
        else:            
            self._writer.append_data(np.array(map_image(team1_state, team2_state, soccer_state)))

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close() #
    


    def _to_image(self, x, proj, view):

        out_of_frame = False
        op = np.array(list(x) + [1])
        #print (f' the shapes proj, view, op:  {proj.shape}, {view.shape}, {op.shape}')
        p = proj @ view @ op
        x = p[0] / p[-1]   #p is [float, float, float, float]
        y = -p[1] / p[-1]

        if abs(x) > 1:
          print ("NOTE-------------------------------------------------------->We got a coordinate > 1, OUT_OF_FRAME TRUE")
          out_of_frame = True 
          print (x)
         
        aimpoint = np.array([x, y])

        clipped_aim_point = np.clip(aimpoint, -1, 1) 
       
        return aimpoint, out_of_frame
        #return clipped_aim_point, out_of_frame
    



    def collect(_, im, puck_flag, pt):
        from PIL import Image
        from os import path
        x = np.random.rand(3)
        global file_no 
        id = file_no 
        divide_data = False
        
        
        if puck_flag:
          if divide_data:
            fn = path.join('/content/cs342/final/data_YesPuck/', 'ice_hockey' + '_%05d' % id)
          if divide_data == False:
            fn = path.join('/content/cs342/final/data/', 'ice_hockey' + '_%05d' % id)
        if puck_flag == 0:
          if divide_data:
            fn = path.join('/content/cs342/final/data_NoPuck/', 'ice_hockey' + '_%05d' % id)
          if divide_data == False:
            fn = path.join('/content/cs342/final/data/', 'ice_hockey' + '_%05d' % id)
        Image.fromarray(im).save(fn + '.png')

        #Image.fromarray(heatmap).save(fn + '_heatmap.png')
        x[0] = pt[0]
        x[1] = pt[1]
        x[2] = puck_flag
                
        with open(fn + '.csv', 'w') as f: 
          #f.write('%0.1f,%0.1f,%0.1f' % tuple(x))  #with puck flag
          f.write('%0.1f,%0.1f' % tuple(pt))
        #with open(fn + 'puck_flag.csv', 'w') as f: 
          #f.write('%0.1f' % puck_flag)
        file_no += 1


#test
class StateRecorder(BaseRecorder):
    def __init__(self, state_action_file, record_images=True):
        self._record_images = record_images
        self._f = open(state_action_file, 'wb')

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        from pickle import dump
        data = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            #print ("\n...............Adding team images to the pickle..., below the soccerball location:\n")
            #print(soccer_state['ball']['location'])
            data['team1_images'] = team1_images
            data['team2_images'] = team2_images
        dump(dict(data), self._f)
        self._f.flush()

    def __del__(self):
        if hasattr(self, '_f'):
            self._f.close()


def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

