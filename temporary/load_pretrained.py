import torch


def main():
    coarse_dict = {}
    ckpt = torch.load('pretrained/coarse.pt', map_location=torch.device('cpu'))
    print(type(ckpt))
    for key, val in ckpt['model'].items():
        key = key[8:]
        print(key)
        coarse_dict[key] = val
        #print(val)

if __name__ == '__main__':
    main()