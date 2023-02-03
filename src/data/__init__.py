from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data: #model loader = data.Data(args)
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only: #test_only = 'store_true'
            #store_true : 인자를 적으면 해당인자에 true를 부여 (안적으면 false)
            #default에 아무 값 할당 x > false 따라서 if not false:이므로 실행
            datasets = [] #list 정의
            for d in args.data_train: #data_train = 'DIV2K', args.data_train = args.data_train.split('+')
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG' #modul_name은 d이다. find('DIV2K-Q')를 찾으면 해당 위치index or -1을 반환 
                #DIV2K-Q 포함 안하므로 module_name = d = DIV2K
                m = import_module('data.' + module_name.lower()) # m = 'data.div2k', import_module(data.div2k) module data.div2k를 import함 <data.div2k.py>
                datasets.append(getattr(m, module_name)(args, name=d)) 

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size, #16
                shuffle=True,
                pin_memory=not args.cpu, #args.cpu = False, pin_memory = True
                num_workers=args.n_threads, #num_workers = 6
            )

        self.loader_test = []
        for d in args.data_test: # d in DIV2K (True)
            if d in ['Set5', 'Set14', 'B100', 'Urban100']: #False
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else: #실행
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader( #이거 어디있지????
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads, #args.n_threads=6, 일반적으로 gpu개수의 2~3배
                )
            )
