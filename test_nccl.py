import torch
import sys
import torch.cuda.nccl2 as nccl2


def main():
    rank = int(sys.argv[1])
    path = sys.argv[2]

    uid = nccl2.get_unique_id()
    if rank == 0:
        with open(path, 'wb') as f:
            f.write(uid)
        with open(path, 'rb') as f:
            uid = f.read()
    else:
        with open(path, 'rb') as f:
            uid = f.read()

    with torch.cuda.device(rank):
        nccl2.initialize(2, uid, rank)
        #t = torch.cuda.FloatTensor(10).fill_(rank + 2)
        #nccl2.all_reduce(t)
        #print(t)



if __name__ == '__main__':
  main()
