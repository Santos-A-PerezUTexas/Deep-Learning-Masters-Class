#https://medium.com/@rktkek456/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a
#Oct 4 2021

import math
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='scalar/tutorial')

for step in range(-360, 360):
    angle_rad = step * math.pi / 180
    writer.add_scalar('sin', math.sin(angle_rad), step)
    writer.add_scalar('cos', math.cos(angle_rad), step)
    writer.add_scalars('sin and cos', {'sin': math.sin(angle_rad), 'cos': math.cos(angle_rad)}, step)
writer.close()

