# -*- coding:utf-8 -*-
from Age_model_12 import *
from absl import flags
from random import shuffle

import numpy as np
import sys
import os

flags.DEFINE_integer("img_size", 64, "Height and Width")

flags.DEFINE_integer("img_ch", 3, "Image channels")

flags.DEFINE_integer("batch_size", 128, "Batch size")

flags.DEFINE_integer("num_classes", 60, "Numner of classes")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_float("lr", 0.01, "Training learning rate")

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Training text path")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("te_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Testing text path")

flags.DEFINE_string("te_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Testing image path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("graphs", "", "Saving loss and mae graphs")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.SGD(FLAGS.lr)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 255.

    if lab_list == 74:
        lab_list = 72
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    elif lab_list == 75:
        lab_list = 73
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    elif lab_list == 76:
        lab_list = 74
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    elif lab_list == 77:
        lab_list = 75
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    else:
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)

    return img, lab

def te_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 255.

    if lab == 74:
        lab = 72
        lab = lab - 16
    elif lab == 75:
        lab = 73
        lab = lab - 16
    elif lab == 76:
        lab = 74
        lab = lab - 16
    elif lab == 77:
        lab = 75
        lab = lab - 16
    else:
        lab = lab - 16

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):
    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def cal_mae(model, images, labels):

    logits = run_model(model, images, False)
    logits = tf.nn.softmax(logits, 1)
    predict_age = tf.cast(tf.argmax(logits, 1), tf.int32)

    ae = tf.reduce_sum(tf.abs(predict_age - labels))

    return ae

def main():
    model = ensemble_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 1))
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("=======================")
            print("Restored!!!!")
            print("=======================")

    if FLAGS.train:
        count = 0
        tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_path, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(FLAGS.batch_size)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        val_log_dir = FLAGS.graphs + current_time + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(model, batch_images, batch_labels)

                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=count)

                if count % 10 == 0:
                    print("Epochs = {} [{}/{}] Loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 150 == 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img) // FLAGS.batch_size
                    ae = 0
                    for i in range(te_idx):
                        imgs, labs = next(te_iter)

                        ae += cal_mae(model, imgs, labs)
                    MAE = ae / len(te_img)
                    print("================================")
                    print("step = {}, MAE = {}".format(count, MAE))
                    print("================================")

                    with val_summary_writer.as_default():
                        tf.summary.scalar('MAE', MAE, step=count)

                    num_ = int(count // 150)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    if not os.path.isdir(model_dir):
                        os.makedirs(model_dir)
                        print("Make {} files to save checkpoint".format(num_))

                    ckpt = tf.train.Checkpoint(model=model, optim=optim)
                    ckpt_dir = model_dir + "/" + "New_age_estimation_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()