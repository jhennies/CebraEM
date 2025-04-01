import os
import time


def train_unet(
        train_data_dirpath,
        unet_name=None,
        unet_src_dirpath=None,
        out_dirpath=None,
        verbose=False
):

    if out_dirpath is not None:
        if not os.path.exists(out_dirpath):
            import random
            time.sleep(random.uniform(0, 5))
            os.mkdir(out_dirpath)
        os.chdir(out_dirpath)

    if verbose:
        print(f'train_data_dirpath = {train_data_dirpath}')
        print(f'unet_name = {unet_name}')
        print(f'unet_src_dirpath = {unet_src_dirpath}')
        print(f'verbose = {verbose}')

    if unet_src_dirpath is not None:
        import sys
        sys.path.append(unet_src_dirpath)

    import numpy as np
    from tqdm import tqdm
    from tifffile import imread
    from datetime import datetime
    from csbdeep.utils import Path
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(fraction=0.8, total_memory=12000)
    from csbdeep.data.generate import sample_patches_from_multiple_stacks
    from augmend import Augmend, Elastic, Identity, FlipRot90, AdditiveNoise, IntensityScaleShift
    from model import UNetConfig, UNet
    np.random.seed(42)

    root = Path(train_data_dirpath)

    def get_data(subset="train", nfiles=None, inds=None, shuffle=True):
        src = root / subset
        fx = sorted((src / "images").glob("*.tif"))
        fy = sorted((src / "masks").glob("*.tif"))
        assert len(fx) == len(fy)

        for f1, f2 in zip(fx, fy):
            print(f"{Path(f1).name}")
            print(f"{Path(f2).name}")

        if shuffle:
            np.random.seed(42)
            inds0 = np.arange(len(fx))
            np.random.shuffle(inds0)
            fx = np.array(fx)[inds0]
            fy = np.array(fy)[inds0]

        if inds is not None:
            fx = np.array(fx)[inds]
            fy = np.array(fy)[inds]
        else:
            fx = fx[:nfiles]
            fy = fy[:nfiles]

        def crop(x):
            return x[tuple(slice(0, (s // 8) * 8) for s in x.shape)]

        X = [crop(imread(str(f))).astype(np.float32) / 255. for f in tqdm(fx)]

        Y = [crop(imread(str(f)).astype(np.uint8)) for f in tqdm(fy)]

        return X, Y

    def batch_generator(X, Y, patch_size=(32, 112, 112), batch_size=4, shuffle=True):
        if len(X) != len(Y):
            raise ValueError("len(X) != len(Y)")

        if len(X) < batch_size:
            raise ValueError("len(X) < batch_size")

        inds = np.arange(len(X))

        if shuffle:
            np.random.shuffle(inds)

        count = 0
        while True:
            b = tuple(sample_patches_from_multiple_stacks([X[i], Y[i]],
                                                          patch_size=patch_size,
                                                          n_samples=1) for i in inds[:batch_size])
            X_batch, Y_batch = zip(*b)
            X_batch = np.stack(X_batch)[:, 0]
            Y_batch = np.stack(Y_batch)[:, 0]

            yield X_batch, Y_batch

            count += batch_size
            if count + batch_size >= len(X) and shuffle:
                np.random.shuffle(inds)
            inds = np.roll(inds, -batch_size)
            count = count % len(X)

    X, Y = get_data("train")
    Xv, Yv = get_data("val")

    aug = Augmend()
    aug.add([FlipRot90(axis = (1,2)),FlipRot90(axis = (1,2))])
    aug.add([Elastic(grid=5, amount=5, order=0, use_gpu=True, axis = (0,1,2)),
                 Elastic(grid=5, amount=5, order=0, use_gpu=True, axis = (0,1,2))],
                probability=.8)
    aug.add([AdditiveNoise(sigma=(0,0.05)),Identity()], probability=.5)
    aug.add([IntensityScaleShift(scale=(.7,1.2), shift=(-0.1,0.1), axis = (0,1,2)),Identity()])

    def proc_image(x,y, augment = 0):
        """create border mask etc"""
        if augment>0:
            x,y = aug([x,y])
        y = (y>0).astype(np.float32)[...,np.newaxis]
        x = x[...,np.newaxis]
        return x,y

    def class_generator(gen, augment = 0):
        for x,y in gen:
            a,b =  tuple(zip(*tuple(proc_image(_x,_y, augment) for _x,_y in zip(x,y))))
            yield np.stack(a), np.stack(b)

    gen = class_generator(batch_generator(X,Y,
                                          patch_size=(48,128,128),
                                          batch_size=min(1,len(X))),augment = 1)
    gen_val = class_generator(batch_generator(Xv,Yv,batch_size=min(3,len(Xv)),
                                              patch_size=(48,128,128),
                                              shuffle = False),augment = 0)

    conf = UNetConfig(
        axes="ZYX",
        unet_n_depth=3,
        unet_pool_size=(2, 4, 4),
        train_reduce_lr={'factor': 0.5, 'patience': 50, 'min_delta': 0},
        train_class_weight=(1, 5)
    )

    if unet_name is None:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        unet_name = f"{timestamp}_unet"

    model = UNet(conf, name=unet_name, basedir="models")

    Xvv, Yvv = next(gen_val)

    model.train(
        X=None, Y=None, data_gen=gen, validation_data=[Xvv, Yvv],
        epochs=300, steps_per_epoch=512
    )


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description=('Trains a Mueller et al. U-Net\n'
                     'Derived from https://github.com/betaseg/protocol-notebooks/blob/main/unet/run_unet.ipynb'),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('train_data_dirpath', type=str,
                        help='Folder containing the ground truth data')
    parser.add_argument('-name', '--unet_name', type=str, default=None,
                        help='Name for the unet to save in a respective directory')
    parser.add_argument('-src', '--unet_src_dirpath', type=str, default=None,
                        help='Local location of https://github.com/betaseg/protocol-notebooks/tree/main/unet')
    parser.add_argument('--out_dirpath', '-out', type=str, default=None,
                        help='')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    train_data_dirpath = args.train_data_dirpath
    unet_name = args.unet_name
    unet_src_dirpath = args.unet_src_dirpath
    out_dirpath = args.out_dirpath
    verbose = args.verbose

    train_unet(
        train_data_dirpath,
        unet_name=unet_name,
        unet_src_dirpath=unet_src_dirpath,
        out_dirpath=out_dirpath,
        verbose=verbose,
    )

