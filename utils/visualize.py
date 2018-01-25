# coding:utf8
import visdom
import time
import numpy as np


class Visualizer():
    def __init__(self, part_id, env='default', **kwargs):
        self.viz = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ""
        self.part_id = part_id

    def plot(self, name, y):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.viz.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H:%M:%S'), \
            info=info))
        self.viz.text(self.log_text, win=win)

    def heatmap_many(self, heatmap_out):
        for i, hm in enumerate(heatmap_out, 1):
            self.viz.heatmap(
                X=np.array(hm),
                win=unicode(self.part_id[i]),
                opts=dict(
                    columnnames=range(256),
                    rownames=range(256),
                    colormap='Viridis',
                    title=self.part_id[i],
                )
            )

    def img_vis(self, img):
        self.viz.image(
            img,
            opts=dict(title='Image', caption='Img'),
        )
