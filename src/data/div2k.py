import os
from data import srdata

class DIV2K(srdata.SRData):#부모클래스 srdata.SRData로부터 상속
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]#data_range는 default값 1-800/801-810
        #data_range = [['1','800'],['801','810']]
        if train:#true이므로 실행
            data_range = data_range[0] #data_range = [1,800] 
        else:
            if args.test_only and len(data_range) == 1: #test_only는 action = store_true인데 default = none이므로, false를 의미 (실행x)
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range)) #self.begin = 1, self.end = 800 (type = int)
        super(DIV2K, self).__init__( #부모 class(srdata.SRData)의 정의를 가져온다. <srdata.SRData.py>
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR') #dir_data/DIV2K_train_HR
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic') #dir_data/DIV2K_train_LR_bicubic
        if self.input_large: self.dir_lr += 'L' #False

