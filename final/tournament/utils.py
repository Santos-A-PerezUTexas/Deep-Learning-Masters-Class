import numpy as np
from enum import IntEnum

file_no = 1

class Team(IntEnum):
    RED = 0
    BLUE = 1


def video_grid(team1_images, team2_images, team1_state='', team2_state=''):
    from PIL import Image, ImageDraw
    grid = np.hstack((np.vstack(team1_images), np.vstack(team2_images)))
    grid = Image.fromarray(grid)
    grid = grid.resize((grid.width // 2, grid.height // 2))

    draw = ImageDraw.Draw(grid)
    draw.text((20, 20), team1_state, fill=(255, 0, 0))
    draw.text((20, grid.height // 2 + 20), team2_state, fill=(0, 0, 255))
    return grid


def map_image(team1_state, team2_state, soccer_state, resolution=512, extent=65, anti_alias=1):
    BG_COLOR = (0xee, 0xee, 0xec)
    RED_COLOR = (0xa4, 0x00, 0x00)
    BLUE_COLOR = (0x20, 0x4a, 0x87)
    BALL_COLOR = (0x2e, 0x34, 0x36)
    from PIL import Image, ImageDraw
    r = Image.new('RGB', (resolution*anti_alias, resolution*anti_alias), BG_COLOR)

    def _to_coord(x):
        return resolution * anti_alias * (x + extent) / (2 * extent)

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

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        if team1_images and team2_images:
            print ("/n VideoRecorder() in utils.py ----- Putting Images in Grid, ball location: /n")
            print(tuple(soccer_state['ball']['location']))
           
            #convert/normalize coordinates!!!!!!!!!!!!!!!!!!!!!! 
            x=soccer_state['ball']['location'][0]
            y=soccer_state['ball']['location'][2]
            xy = np.random.rand(2)
            xy[0] = x
            xy[1] = y
            #self.collect(team1_images[0], soccer_state['ball']['location'])
            self.collect(team1_images[0], xy)
            #print (len(team1_images[0])) #300
            #print (len(team1_images[0][0])) #400
            #print (len(team1_images[0][0][0])) #3
            
            self._writer.append_data(np.array(video_grid(team1_images, team2_images,
                                                        'X Puck Location: %d' % x,
                                                        'Y Puck Location: %d' % y)))
        
            #self._writer.append_data(np.array(video_grid(team1_images, team2_images,
             #                                            'Utexas Cici Blue: %d' % soccer_state['score'][1],
              #                                           'Utexas Santos Red: %d' % soccer_state['score'][0])))
        else:
            print ("            No  Images, calling map_image, ball location:")
            print(soccer_state['ball']['location'])
            self._writer.append_data(np.array(map_image(team1_state, team2_state, soccer_state)))

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()
    
    def collect(_, im, pt):
        from PIL import Image
        from os import path
        #global n  #global n
        global file_no 
        print ("/nCollect() has been called to generate images, hurray!/n")
        id = file_no #if n < images_per_track else np.random.randint(0, n + 1)
        fn = path.join('/content/cs342/final/data/', 'ice_hockey' + '_%05d' % id)
        Image.fromarray(im).save(fn + '.png')
        #print(f'image size is {Image.fromarray(im).size} ')
        with open(fn + '.csv', 'w') as f: 
          f.write('%0.1f,%0.1f' % tuple(pt))
        file_no += 1



class StateRecorder(BaseRecorder):
    def __init__(self, state_action_file, record_images=True):
        self._record_images = record_images
        self._f = open(state_action_file, 'wb')

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        from pickle import dump
        data = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            print ("/n...............Adding team images to the pickle..., below the soccerball location:/n")
            print(soccer_state['ball']['location'])
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

