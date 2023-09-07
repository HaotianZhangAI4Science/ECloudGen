from dataset.dataset import create_h5
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='./data/ecloud.h5')
    parser.add_argument('--ecloud_path', type=str, default='./data/ecloud')
    parser.add_argument('--mol_csv', type=str, default='./data/moses2.csv')
    args = parser.parse_args()

    create_h5(args.h5_path, args.ecloud_path, args.mol_csv)