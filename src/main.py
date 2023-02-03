import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main(): #training을 위한 main.py실행
    global model #전역변수 정의
    if args.data_test == ['video']: #data_set이 video라면 실행, 지금은 DIV2K로 지정되어있음.
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else: #얘가 실행.
        if checkpoint.ok: #checkpoint는 utility에서 정의/ self.ok = True 이므로
            loader = data.Data(args) #data의 __init__의 Data에서 load  <data/__init__.py>
            _model = model.Model(args, checkpoint) #model의 __init__의 Model에서 load <model/__init__.py>
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None #loss의 __init__의 Loss에서 load <loss/__init.py>
            #args.test_only = False, not False = True이므로 loss.Loss실행
            t = Trainer(args, loader, _model, _loss, checkpoint) #<trainer.py>의 Trainer
            while not t.terminate(): #model.train() : train을 실행, False라면 model.test()와 같이 test진행
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
