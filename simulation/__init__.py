from . import colour
from . import convolution
from . import genome
from . import reduction
from . import features
from . import run


class CLContext:
    def __init__(self, ctx, queue, ifmt, gs, wgs):
        self.ctx = ctx
        self.queue = queue
        self.ifmt = ifmt
        self.gs = gs
        self.wgs = wgs
