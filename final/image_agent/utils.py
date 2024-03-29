import numpy as np
import pystk
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

DATASET_PATH = '/content/cs342/final/data'               
#DATASET_PATH = '/content/cs342/final/data_instance'     #render_data instance path

#Dec 9, 2021
#data2 = torch.from_numpy(data.astype(int))

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        
        self.data = []
        
        
        for f in glob(path.join(dataset_path, '*.csv')):   #change to npy to load render_data instance
            
            data_image = Image.open(f.replace('.csv', '.png'))   #change to npy to load render_data instance
            data_image.load()
            self.data.append(( data_image,    np.loadtxt(f, dtype=np.float32, delimiter=',')  ))
            
            #uncomment below to load render_data instance
            #data_instance = torch.from_numpy(np.load(f).astype(int)) # render_data instance
            #self.data.append((     data_image,    data_instance  ))
        
        self.transform = transform
        #self.totensor = dense_transforms.ToTensor()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        data = self.data[idx]
        data = self.transform(*data)

        #uncomment below to load render_data instance
        #im = data[0]
        #label = data[1]
        #im = self.transform(im)
        #im = self.totensor(im)
        #return im[0], label
      
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def rollout(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        print ("INSIDE OF ROLLOUT")
        print (f'self.k is {self.k}')
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
            print ("Line 85")
        else:
            print ("Line 87")
            print (f'self.k is {self.k}')
            if self.k is not None:
                print ("Stopping")
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            config.mode = config.RaceMode.SOCCER

            print ("Line 96")
            self.k = pystk.Race(config)
            print (f'self.k is {self.k}')
            print ("Line 98")
            self.k.start()
            print ("Line 99")
            self.k.step()
            print ("END OF WHILE LOOP IN ROLLOUT")

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

        print ("UPDATING STATE LINE 116 utils.py")
        for t in range(max_frames):
            print ("Inside FOR loop line 118")
            state.update()  
            print ("Updated state")
            #track.update()  #rollout calls this method and exits (Nov 23, 2021)
            print ("Did updates")
            kart = state.players[0].kart
            print (track.length)
            #if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
             #   print ("inside loop at 125")
              #  if verbose:
               #     print("Finished at t=%d" % t)
                #break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                print ("Calling data_callback, or collect(), to generate data, L135")
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)
            print ("CALLING THE CONTROLLER LINE 142")
            action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t, kart.overall_distance / track.length

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()


if __name__ == '__main__':
    from .controller import control
    from argparse import ArgumentParser
    from os import makedirs


    def noisy_control(aim_pt, vel):
        return control(aim_pt + np.random.randn(*aim_pt.shape) * aim_noise,
                       vel + np.random.randn() * vel_noise)


    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_images', default=10000, type=int)
    parser.add_argument('-m', '--steps_per_track', default=20000, type=int)
    parser.add_argument('--aim_noise', default=0.1, type=float)
    parser.add_argument('--vel_noise', default=5, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass
    print ("In main line 198, creating pytux object")
    print (args.track)
    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = 0, 0


        def collect(_, im, pt):
            from PIL import Image
            from os import path
            global n
            print ("Collect() has been called to generate images")
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(args.output, track + '_%05d' % id)
                Image.fromarray(im).save(fn + '.png')
                with open(fn + '.csv', 'w') as f:
                    f.write('%0.1f,%0.1f' % tuple(pt))
            n += 1


        while n < args.steps_per_track:
            print ("In while loop calling rollout()")
            steps, how_far = pytux.rollout(track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect)
            print(steps, how_far)
            # Add noise after the first round
            aim_noise, vel_noise = args.aim_noise, args.vel_noise
    pytux.close()
