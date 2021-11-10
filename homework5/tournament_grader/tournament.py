import logging
from pathlib import Path
from .grader import load_assignment
import ray

track = 'cocoa_temple'
laps = 1
max_frames = 1000
top = 16
num_proc = 8

font_list = ['Helvetica', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']

# The tournament scorring function
def score_fn(progress, relative_finish_time):
    # Let's use the progress as score if not finished (higher better), or 2-finish_time if finished
    return 2 - relative_finish_time if progress >= 0.99 else progress


class DrivingLog:
    score = None
    name = None
    finish_time = None
    images = None

    def __init__(self, name):
        self.name = name
        self.images = []

    def __lt__(self, other):
        return self.score < other.score


@ray.remote
class TopQueue:
    def __init__(self, k=16):
        self.max_size = k
        self.q = []

    def add(self, driving_log):
        from heapq import heappush, heappop
        heappush(self.q, driving_log)
        while len(self.q) > self.max_size:
            heappop(self.q)

    def results(self):
        return [(i.name, i.score) for i in self.q]

    def make_video(self, out_file):
        from PIL import Image, ImageDraw, ImageFont
        import imageio
        import numpy as np

        # Fetch the top submissions
        top_submissions = sorted(self.q, reverse=True)
        n = len(top_submissions)

        # Find the best video size (tiling and width/height)
        from math import sqrt
        ny = int(sqrt(n))
        nx = (n-1) // ny + 1
        h, w = top_submissions[0].images[0].shape[:2]
        max_frames = max(*[len(s.images) for s in top_submissions])

        # Load a font
        font = ImageFont.load_default()
        for font_name in font_list:
            try:
                font = ImageFont.truetype(font_name, 16)
            except OSError:
                pass
            else:
                break

        # Create the video
        writer = imageio.get_writer(out_file, fps=20)
        for t in range(max_frames+10):
            r_im = Image.new('RGB', (w * nx, h * ny))
            for i, s in enumerate(top_submissions):
                im = Image.fromarray(s.images[t if t < len(s.images) else -1])
                if s.name is not None and t >= len(s.images):
                    # Imprint the name onto the image
                    draw = ImageDraw.Draw(im)
                    text = s.name
                    if s.finish_time is not None:
                        text += '\n{:0.2f}s'.format(s.finish_time)
                    draw.rectangle(draw.multiline_textbbox((0, 0), text, font=font), fill='#ffffff')
                    draw.multiline_text((0, 0), text, fill='#000000', font=font)
                r_im.paste(im, ((i % nx) * w, i // nx * h))

            writer.append_data(np.array(r_im))
        writer.close()


@ray.remote
class Grader:
    def __init__(self, q: TopQueue, track_name, laps=1, max_frames=1000):
        import pystk
        logging.info('Starting pystk')
        pystk_config = pystk.GraphicsConfig.hd()
        pystk_config.screen_width = 128
        pystk_config.screen_height = 96

        pystk.init(pystk_config)
        self.q, self.track_name, self.laps, self.max_frames = q, track_name, laps, max_frames

    def grade(self, submission, name):
        import torchvision.transforms.functional as TF
        import pystk
        import numpy as np
        RESCUE_TIMEOUT = 30

        # Load the module
        logging.info('Loading assignment {!r}'.format(submission))
        module = load_assignment(submission)
        if module is None:
            return None

        # Load the policy
        C = module.control
        P = module.load_model().eval()

        # Firing up pystk
        logging.info('Loading track {!r}'.format(self.track_name))
        config = pystk.RaceConfig(num_kart=1, laps=self.laps)
        config.track = self.track_name
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

        recording = DrivingLog(name)
        # logging.info('Starting race')
        k = pystk.Race(config)
        try:
            state = pystk.WorldState()
            track = pystk.Track()

            k.start()
            k.step()

            last_rescue = 0
            for t in range(self.max_frames):
                state.update()
                track.update()

                kart = state.players[0].kart

                image = np.array(k.render_data[0].image)

                if kart.race_result:
                    recording.score = score_fn(1, t / self.max_frames)
                    recording.finish_time = t * config.step_size
                    break

                recording.images.append(image)
                recording.score = score_fn(kart.overall_distance / (self.laps*track.length), t / self.max_frames)

                aim_point_image = P(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

                current_vel = np.linalg.norm(kart.velocity)
                action = C(aim_point_image, current_vel)

                if current_vel <= 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                    action.rescue = True
                    last_rescue = t

                k.step(action)
        except Exception as e:
            return {'score': -2, 'error': e, 'name': name}
        finally:
            k.stop()
            del k
        self.q.add.remote(recording)
        return {'score': recording.score, 'name': name}


def grade(submissions, workdir, track=track, laps=laps, max_frames=max_frames, top=top, num_proc=num_proc):
    logging.info('Loading Ray')
    ray.init(include_dashboard=False, num_gpus=1)

    logging.info('Setting up grading processes')
    q = TopQueue.remote(top)
    grade_procs = [Grader.remote(q, track, laps, max_frames) for _ in range(num_proc)]

    logging.info('  Starting grading')
    grades = [grade_procs[i % len(grade_procs)].grade.remote(p, n) for i, (n, p) in enumerate(submissions.items())]

    logging.info('  Waiting for grading to finish')
    grades = ray.get(grades)

    logging.info('  Cleanup')
    del grade_procs

    logging.info('  Computing extra credit')
    final_score = {g['name']: {'score': max(10-i, 0), 'info': 'Rank {} ({})'.format(i+1, g.get('error', 'No Error'))}
                   for i, g in enumerate(sorted(grades, key=lambda g: -g['score']))}

    logging.info('Making a video')
    ray.get(q.make_video.remote(Path(workdir) / 'video.mp4'))

    return final_score

