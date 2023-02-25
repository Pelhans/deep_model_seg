from nntplib import NNTP_PORT
import os, re
import codecs
from tensorflow.python import pywrap_tensorflow

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

model_dir = os.getcwd() # get current work path
print(model_dir)
checkpoint_path = model_dir + "/models/cnn_crf/model.ckpt-238268" # model path

all_weight = {}

print(checkpoint_path)

def get_line(file):
    with codecs.open(file, "r", "utf-8") as inf:
        for line in inf:
            yield line.rstrip("\n")

def get_all_weight():
    # get param from ckpt
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # cout var name and param
    for key in var_to_shape_map:
        if "adam" in key or "global_step" in key:
            continue
        name = "_".join(key.split("/"))
        print("tensor_name: ", name)
        all_weight[name] = np.asarray(reader.get_tensor(key), dtype=np.float32)

def get_word2id():
    outf = codecs.open("./param/word2id.dat", "w", "utf-8")
    wordidmap = {}
    with open("./pb_model/vocab.txt", "r") as vf:
        wordidmap = {k.strip(): idx for (idx, k) in enumerate(vf.readlines())}
    for line in wordidmap.keys():
        outf.write(line + " " + str(wordidmap[line]) + "\n")

def get_embed():
    for name in all_weight.keys():
        if "cnn_crf_embedding" != name:
            continue
        inf = codecs.open("./param/embed.embedding.weight", "w", "utf-8")
        np_tensor = all_weight[name]
        shape = np.shape(np_tensor)
        for i in range(shape[0]):
            inf.write(" ".join([str(j) for j in np_tensor[i]]) + "\n")

def get_transitions():
    for name in all_weight.keys():
        if "bio_transitions" not in name:
            continue
        print("??")
        inf = codecs.open("./param/bio_transitions", "w", "utf-8")
        np_tensor = all_weight[name]
        shape = np.shape(np_tensor)
        for i in range(shape[0]):
            inf.write(" ".join([str(j) for j in np_tensor[i]]) + "\n")

def get_conv_kernel():
    for name in all_weight.keys():
        if "conv" not in name or "kernel" not in name:
            continue
        tmp_name = name.replace("cnn_crf_", "").replace("_conv1d_kernel", "")
        tmp_name = re.sub(r"^cnn_", "layer_", tmp_name)
        inf = codecs.open("./param/" + tmp_name + ".weight", "w", "utf-8")
        np_tensor = all_weight[name]
        shape = np.shape(np_tensor)
        np_tensor = np.transpose(np_tensor, [2, 0, 1])
        np_tensor = np.reshape(np_tensor, [shape[-1], -1])
        for i in range(shape[-1]):
            inf.write(" ".join([str(j) for j in np_tensor[i]]) + "\n")

def get_conv_bias():
    for name in all_weight.keys():
        if "conv" not in name or "bias" not in name:
            continue
        tmp_name = name.replace("cnn_crf_", "").replace("_conv1d_bias", "")
        tmp_name = re.sub(r"^cnn_", "layer_", tmp_name)
        inf = codecs.open("./param/" + tmp_name + ".bias", "w", "utf-8"  )
        np_tensor = all_weight[name]
        np_tensor = np.reshape(np_tensor, [-1])
        inf.write(" ".join([str(j) for j in np_tensor]) + "\n")

def get_linear_kernel():
    for name in all_weight.keys():
        if (not ("linear" in name or "fnn" in name)) or "kernel" not in name:
            continue
        inf = codecs.open("./param/" + name.replace("cnn_crf_", "").replace("_dense_kernel", "") + ".weight", "w", "utf-8")
        np_tensor = all_weight[name]
        shape = np.shape(np_tensor)
        np_tensor = np.transpose(np_tensor, [1, 0])
        np_tensor = np.reshape(np_tensor, [shape[-1], -1])
        for i in range(shape[-1]):
            inf.write(" ".join([str(j) for j in np_tensor[i]]) + "\n")

def get_linear_bias():
    for name in all_weight.keys():
        if (not("linear" in name or "fnn" in name)) or "bias" not in name:
            continue
        inf = codecs.open("./param/" + name.replace("cnn_crf_", "").replace("_dense_bias", "") + ".bias", "w", "utf-8")
        np_tensor = all_weight[name]
        np_tensor = np.reshape(np_tensor, [-1])
        inf.write(" ".join([str(j) for j in np_tensor]) + "\n")

def get_crf():
    for name in all_weight.keys():
        if "bio" not in name:
            continue
        tmp_name = name.replace("cnn_crf_", "").replace("_conv1d_bias", "")
        tmp_name = re.sub(r"^cnn_", "layer_", tmp_name)
        inf = codecs.open("./param/" + tmp_name + ".bias", "w", "utf-8"  )
        np_tensor = all_weight[name]
        np_tensor = np.reshape(np_tensor, [-1])
        inf.write(" ".join([str(j) for j in np_tensor]) + "\n")

if __name__ == "__main__":
    get_word2id()
    get_all_weight()
    get_transitions()
    get_embed()
    get_conv_kernel()
    get_conv_bias()
    get_linear_kernel()
    get_linear_bias()
