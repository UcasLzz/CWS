class CWSMetric(object):
    def __init__(self, *names):
        super(CWSMetric, self).__init__()
        self.names = names
        self.sum_metric = None
        self.num_inst = None
        self.reset()
        

    def update(self, pred, label):
        assert isinstance(pred, list) or isinstance(pred, tuple), 'pred should be list or tuple'
        assert isinstance(label, list) or isinstance(label, tuple), 'label should be list or tuple'
        pred = set(pred)
        label = set(label)
        n = len(pred & label)
        N = len(label)
        M = len(pred)
        for i, name in enumerate(self.names):
            if name == 'P':
                self.sum_metric[i] += float(n)/float(N)
            elif name == 'R':
                self.sum_metric[i] += float(n)/float(M)
            elif name == 'F1':
                P = float(n)/(float(N) + 1e-7)
                R = float(n)/(float(M) + 1e-7)
                self.sum_metric[i] += 2 * P * R / (P + R+ 1e-7)
        self.num_inst += 1

    def reset(self):
        if self.sum_metric is None:
            self.sum_metric = [0] * len(self.names)
        else:
            assert len(self.metric) == len(self.names)
            for metric in self.sum_metric:
                metric = 0
        self.num_inst = 0

    def get(self, name=None):
        if name is not None:
            assert name in self.names
            return (name, self.sum_metric[self.names.find(name)] / self.num_inst)
        else:
            return self.getall()
        

    def getall(self):
        if self.num_inst == 0:
            return zip(self.names, [float('nan')] * len(self.names))
        else:
            return zip(self.names, [metric / self.num_inst for metric in self.sum_metric])