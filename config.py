import argparse
import os

from model_joint import Model
import datasets.hdf5_loader as dataset
import numpy as np
import pandas as pd
from six.moves import xrange
from util import log

def get_params():

    def str2bool(v):
        return v.lower() == 'true'
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='GANs')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="Orgimage_128", choices=["Orgimage_128"])
    parser.add_argument('--dump_result', type=str2bool, default=False)
    
    # Model
    parser.add_argument('--batch_size_G', type=int, default=16)
    parser.add_argument('--batch_size_L', type=int, default=16)
    parser.add_argument('--batch_size_U', type=int, default=16)
    parser.add_argument('--n_z', type=int, default=128)
        
    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_image_step', type=int, default=10000)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--output_save_step', type=int, default=10000)
    # learning
    parser.add_argument('--max_training_steps', type=int, default=250001)
    parser.add_argument('--learning_rate_g', type=float, default=2e-4)
    parser.add_argument('--learning_rate_d', type=float, default=5e-5)
    parser.add_argument('--update_rate', type=int, default=2)
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument('--data_id', nargs='*', default=None)
    # }}}
    
    config, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    return config, args

def argparser(config, is_train=True):

    dataset_path = os.path.join('./datasets/TMI/', config["dataset"].lower())
    dataset_train, dataset_val, dataset_test = dataset.create_default_splits(dataset_path)
    dataset_train_unlabel, _ = dataset.create_default_splits_unlabel(dataset_path)
    
    config["img"] = []
    labels = []
    with open('./datasets/metadata.tsv','w') as f:
        f.write("Index\tLabel\n")
        for index, labeldex in enumerate(dataset_test.ids):
            config["img"].append(dataset_test.get_data(labeldex)[0])
            label = np.argmax(dataset_test.get_data(labeldex)[1])
            labels.append(label)
            f.write("%d\t%d\n" % (index, label))
    config["img"] = np.array(config["img"])
    log.info(config["img"].shape)
    config["len"] = config["img"].shape[0]
    config["label"] = labels
    
    config["Size"] = config["len"]
    picname = []
    for i in xrange(config["Size"]):
        picname.append("V{step:04d}.jpg".format(step=i+1))
    Csv=pd.DataFrame(columns=['Label'], index=picname, data=labels)
    Csv.to_csv('./datasets/Classification_Results_label.csv',encoding='gbk')
    
    img, label = dataset_train.get_data(dataset_train.ids[0])
    config["h"] = img.shape[0]
    config["w"] = img.shape[1]
    config["c"] = img.shape[2]
    config["num_class"] = label.shape[0]
    
    # --- create model ---
    model = Model(config, debug_information=config["debug"], is_train=is_train)

    return config, model, dataset_train, dataset_train_unlabel, dataset_val, dataset_test
