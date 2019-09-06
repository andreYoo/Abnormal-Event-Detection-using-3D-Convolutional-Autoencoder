
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# noinspection PyPep8Naming
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow_utils as tf_utils
import utils as utils
import cv2
import pdb
import os
_temp_dim = 5
_lambda = 0.1

class AEDAE(object):
    def __init__(self, sess, flags, dataset):
        self.sess = sess
        self.flags = flags
        self.dataset = dataset
        self.image_size = dataset.image_size

        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c = [1024, 512, 256, 128]  # 4, 8, 16, 32
        self.dis_c = [64, 128, 256, 256]  # 32, 16, 8, 4

        self._build_net()
        self._tensorboard()
        print("Initialized WGAN SUCCESS!")

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None,5, *self.image_size], name='output')
        self.z = tf.placeholder(tf.float32, shape=[None,5, *self.image_size], name='input')

        self.g_samples = self.generator(self.z)

        # discriminator loss

        self.g_recon_loss = tf.losses.mean_squared_error(self.g_samples,self.Y,weights=1.0)
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        # Optimizers for generator and discriminator
        gen_op = tf.train.RMSPropOptimizer(learning_rate=self.flags.learning_rate).minimize(self.g_recon_loss, var_list=g_vars)
        gen_ops = [gen_op] + self._gen_train_ops
        self.gen_optim = tf.group(*gen_ops)

    def _tensorboard(self):
        tf.summary.scalar('loss_AE/g_recon_loss',self.g_recon_loss)
        tf.summary.image('img_AE/current',self.z[0],max_outputs=5)
        tf.summary.image('img_AE/predict',self.g_samples[0],max_outputs=5)
        tf.summary.image('img_AE/gt',self.Y[0],max_outputs=5)
        self.summary_op = tf.summary.merge_all()

    def encoder(self,data,name='enc_'):
        with tf.variable_scope(name):
            # 64 -> 32 or 32 -> 16
            h0_conv = tf_utils.conv3d(data, self.dis_c[0], name='h0_conv3d')
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')

            # 32 -> 16 or 16 -> 8
            h1_conv = tf_utils.conv3d(h0_lrelu, self.dis_c[1], name='h1_conv3d')
            h1_batchnorm = tf_utils.batch_norm(h1_conv, name='h1_batchnorm', _ops=self._dis_train_ops)
            h1_lrelu = tf_utils.lrelu(h1_batchnorm, name='h1_lrelu')

            # 16 -> 8 or 8 -> 4
            h2_conv = tf_utils.conv3d(h1_lrelu, self.dis_c[2], name='h2_conv3d')
            h2_batchnorm = tf_utils.batch_norm(h2_conv, name='h2_batchnorm', _ops=self._dis_train_ops)
            h2_lrelu = tf_utils.lrelu(h2_batchnorm, name='h2_lrelu')

            h3_conv = tf_utils.conv3d(h2_lrelu, self.dis_c[3], name='h3_conv3d')
            h3_batchnorm = tf_utils.batch_norm(h3_conv, name='h3_batchnorm', _ops=self._dis_train_ops)
            h3_lrelu = tf_utils.lrelu(h3_batchnorm, name='h3_lrelu')

            h4_conv = tf_utils.conv3d(h3_lrelu, self.dis_c[3],k_d=1,k_h=2,k_w=2, name='h4_conv3d')
            h4_batchnorm = tf_utils.batch_norm(h4_conv, name='h4_batchnorm', _ops=self._dis_train_ops)
            h4_lrelu = tf_utils.lrelu(h4_batchnorm, name='h4_lrelu')

            h4_flatten = flatten(h4_lrelu)
            h5_linear = tf_utils.linear(h4_flatten, 1024, name='h4_linear')
            return tf.nn.sigmoid(h5_linear), h5_linear


    def decoder(self,data,name='dec_'):
        with tf.variable_scope(name):
            h0_linear = tf_utils.linear(data, 5*8*self.dis_c[3],input_size=1024, name='h0_linear')
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0],1, 5, 8, self.dis_c[3]])
            h0_batchnorm = tf_utils.batch_norm(h0_reshape, name='h0_batchnorm', _ops=self._gen_train_ops)
            h0_relu = tf.nn.relu(h0_batchnorm, name='h0_relu')

            # 8 x 8
            h1_deconv = tf_utils.deconv3d(h0_relu, self.dis_c[3],output_size=[10,15,256],k_t=4,d_t=5, d_h=2, d_w=2,padding_='SAME', name='h1_deconv3d')
            h1_batchnorm = tf_utils.batch_norm(h1_deconv, name='h1_batchnorm', _ops=self._gen_train_ops)
            h1_relu = tf.nn.relu(h1_batchnorm, name='h1_relu')

            # 16 x 16
            h2_deconv = tf_utils.deconv3d(h1_relu, self.dis_c[2],output_size=[20,30,256],stepup_out=1, name='h2_deconv3d')
            h2_batchnorm = tf_utils.batch_norm(h2_deconv, name='h2_batchnorm', _ops=self._gen_train_ops)
            h2_relu = tf.nn.relu(h2_batchnorm, name='h2_relu')

            h3_deconv = tf_utils.deconv3d(h2_relu, self.dis_c[1],output_size=[40,60,128],stepup_out=2,name='h3_deconv3d')
            h3_batchnorm = tf_utils.batch_norm(h3_deconv, name='h3_batchnorm', _ops=self._gen_train_ops)
            h3_relu = tf.nn.relu(h3_batchnorm, name='h3_relu')


            h4_deconv = tf_utils.deconv3d(h3_relu, self.dis_c[0],output_size=[79,119,64],stepup_out=2,d_t=1, d_h=2, d_w=2,padding_='SAME' ,name='h4_deconv3d')
            h4_batchnorm = tf_utils.batch_norm(h4_deconv, name='h4_batchnorm', _ops=self._gen_train_ops)
            h4_relu = tf.nn.relu(h4_batchnorm, name='h4_relu')

                # 64 x 64
            output = tf_utils.deconv3d(h4_relu, self.image_size[2],output_size=self.image_size, d_t=1, d_h=2, d_w=2,padding_='SAME',name='h5_deconv3d')
            return tf.nn.tanh(output)

    def generator(self, data, name='g_'):
        with tf.variable_scope(name):
            enc_output, _ = self.encoder(data)
            dec_output = self.decoder(enc_output)
            return dec_output



    def train_step(self):
        summary_g = None
        # train discriminator
        _gen_train_data, batch_imgs = self.sample_z_ucsd_ped(num=self.flags.batch_size)
        #batch_imgs = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
        #batch_imgs = self.dataset.corresponding_batch(batch_size=self.flags.batch_size, _path_list)
        gen_feed = {self.z: _gen_train_data, self.Y: batch_imgs}
        _, g_recon_loss, summary = self.sess.run([self.gen_optim, self.g_recon_loss, self.summary_op], feed_dict=gen_feed)

        return [g_recon_loss], summary

    def sample_imgs(self):
        _sample,_prediction = self.sample_z_ucsd_ped(num=self.flags.sample_batch)
        g_feed = {self.z:_sample}
        y_fakes = self.sess.run(self.g_samples, feed_dict=g_feed)
        return [y_fakes], [_sample], [_prediction]

    def sample_z(self, num=64):
        return np.random.uniform(-1., 1., size=[num, self.flags.z_dim])


    def file_parsing_test(self,path):
        _current_list= []
        _future_list= []
        _path = path[0:30]
        _spashed = path.split('/')
        _dotted = path.split('.')
        _img_cnt = int(_spashed[len(_spashed)-1].split('.')[0])
        if os.path.isfile(_path+'%s.tif'%(str(_img_cnt+10)).zfill(3)):
            _current_list.append(path)
            for i in range(_temp_dim-1):
                _img_cnt += 1
                _current_list.append(_path+'%s.tif'%(str(_img_cnt)).zfill(3))
                _future_list.append(_path+'%s.tif'%(str(_img_cnt+5)).zfill(3))
            _img_cnt += 1
            _future_list.append(_path+'%s.tif'%(str(_img_cnt+5)).zfill(3))
        else:
            _img_cnt -=15
            path = _path+'%s.tif'%(str(_img_cnt).zfill(3))
            _current_list.append(path)
            for i in range(_temp_dim - 1):
                _current_list.append(_path + '%s.tif'%(str(_img_cnt).zfill(3)))
                _future_list.append(_path + '%s.tif'%(str(_img_cnt + 5)).zfill(3))
                _img_cnt += 1
            _future_list.append(_path + '%s.tif'%(str(_img_cnt + 5)).zfill(3))
        return _current_list, _future_list

    def file_parsing(self,path):
        _current_list= []
        _future_list= []
        _path = path[0:41]
        _spashed = path.split('/')
        _dotted = path.split('.')
        _img_cnt = int(_spashed[len(_spashed)-1].split('.')[0])
        if os.path.isfile(_path+'%s.tif'%(str(_img_cnt+10)).zfill(3)):
            _current_list.append(path)
            for i in range(_temp_dim-1):
                _img_cnt += 1
                _current_list.append(_path+'%s.tif'%(str(_img_cnt)).zfill(3))
                _future_list.append(_path+'%s.tif'%(str(_img_cnt+5)).zfill(3))
            _img_cnt += 1
            _future_list.append(_path+'%s.tif'%(str(_img_cnt+5)).zfill(3))
        else:
            _img_cnt -=15
            path = _path+'%s.tif'%(str(_img_cnt).zfill(3))
            _current_list.append(path)
            for i in range(_temp_dim - 1):
                _current_list.append(_path + '%s.tif'%(str(_img_cnt).zfill(3)))
                _future_list.append(_path + '%s.tif'%(str(_img_cnt + 5)).zfill(3))
                _img_cnt += 1
            _future_list.append(_path + '%s.tif'%(str(_img_cnt + 5)).zfill(3))
        return _current_list, _future_list

    def sample_z_ucsd_ped(self, num=64):
        _len = len(self.dataset.train_data)
        inidex = np.random.randint(10,_len-10, size=64)
        c_input = np.zeros([num,_temp_dim,self.image_size[0],self.image_size[1],self.image_size[2]],dtype=np.float32)
        f_input = np.zeros([num, _temp_dim, self.image_size[0], self.image_size[1], self.image_size[2]],dtype=np.float32)
        for i in range(len(inidex)):
            _img_path = self.dataset.train_data[inidex[i]]
            _current_list,_future_list = self.file_parsing_test(_img_path)
            for j in range(_temp_dim):
                cimg = cv2.imread(_current_list[j])
                fimg = cv2.imread(_future_list[j])
                c_input[i,j] = (cimg/255.0)-0.5
                f_input[i,j] = (fimg/255.0)-0.5
        return c_input,f_input

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('g_recon_loss', loss[0]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def plots(self, imgs_, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake = np.reshape(imgs_[0], (self.flags.sample_batch, *self.image_size))

        imgs = []
        for img in imgs_fake:
            imgs.append(img)

        # parameters for plot size
        scale, margin = 0.04, 0.01
        n_cols, n_rows = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                if self.image_size[2] == 3:
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                elif self.image_size[2] == 1:
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1]), cmap='Greys_r')
                else:
                    raise NotImplementedError

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        plt.close(fig)

    def epn_plots(self, imgs_,input,gt, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake_ft = np.reshape(imgs_[0], (self.flags.sample_batch, 5,self.image_size[0],self.image_size[1],self.image_size[2]))
        imgs_current = np.reshape(input[0], (self.flags.sample_batch, 5,self.image_size[0],self.image_size[1],self.image_size[2]))
        imgs_gt_ft = np.reshape(gt[0], (self.flags.sample_batch, 5,self.image_size[0],self.image_size[1],self.image_size[2]))


        predict = np.array(np.add(imgs_fake_ft[0],0.5)*255).astype(int)
        input = np.array(np.add(imgs_current[0],0.5)*255).astype(int)
        gt = np.array(np.add(imgs_gt_ft[0],0.5)*255).astype(int)
        # parameters for plot size
        # scale, margin = 0.04, 0.01
        # n_cols, n_rows = 3*len(imgs), 3*len(imgs)
        # cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale
        #
        # fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        # gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        # gs.update(wspace=margin, hspace=margin)
        #
        # imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]
        #
        # # save more bigger image
        # for col_index in range(n_cols):
        #     for row_index in range(n_rows):
        #         ax = plt.subplot(gs[row_index * n_cols + col_index])
        #         plt.axis('off')
        #         ax.set_xticklabels([])
        #         ax.set_yticklabels([])
        #         ax.set_aspect('equal')
        #         if self.image_size[2] == 3:
        #             plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
        #                 self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
        #         elif self.image_size[2] == 1:
        #             plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
        #                 self.image_size[0], self.image_size[1]), cmap='Greys_r')
        #         else:
        #             raise NotImplementedError
        template  = np.zeros([self.image_size[0]*3,self.image_size[1]*5+30,3])

        width = self.image_size[1]
        height = self.image_size[0]
        for i in range(5):
            template[0:height,i*width:(i+1)*width] = predict[i]
            template[height:2*height, i * width: (i + 1) * width] = input[i]
            template[2*height:3*height, i * width: (i + 1) * width] = gt[i]
        cv2.imwrite(save_file + '/sample_{}.png'.format(str(iter_time)),template)



    def epn_test_plots(self, imgs_,input,gt, iter_time, save_file):
            # reshape image from vector to (N, H, W, C)
            imgs_fake_ft = np.reshape(imgs_[0], (self.flags.sample_batch, 5,self.image_size[0],self.image_size[1],self.image_size[2]))
            imgs_current = np.reshape(input[0], (self.flags.sample_batch, 5,self.image_size[0],self.image_size[1],self.image_size[2]))
            imgs_gt_ft = np.reshape(gt[0], (self.flags.sample_batch, 5,self.image_size[0],self.image_size[1],self.image_size[2]))

            for b in range(self.flags.sample_batch):
                predict = np.array(np.add(imgs_fake_ft[b],0.5)*255).astype(int)
                input = np.array(np.add(imgs_current[b],0.5)*255).astype(int)
                gt = np.array(np.add(imgs_gt_ft[b],0.5)*255).astype(int)
                # parameters for plot size
                # scale, margin = 0.04, 0.01
                # n_cols, n_rows = 3*len(imgs), 3*len(imgs)
                # cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale
                #
                # fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
                # gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
                # gs.update(wspace=margin, hspace=margin)
                #
                # imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]
                #
                # # save more bigger image
                # for col_index in range(n_cols):
                #     for row_index in range(n_rows):
                #         ax = plt.subplot(gs[row_index * n_cols + col_index])
                #         plt.axis('off')
                #         ax.set_xticklabels([])
                #         ax.set_yticklabels([])
                #         ax.set_aspect('equal')
                #         if self.image_size[2] == 3:
                #             plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                #                 self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                #         elif self.image_size[2] == 1:
                #             plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                #                 self.image_size[0], self.image_size[1]), cmap='Greys_r')
                #         else:
                #             raise NotImplementedError
                template  = np.zeros([self.image_size[0]*3,self.image_size[1]*5+30,3])
                l2_attention_map = utils.aed_localization(imgs_fake_ft[b],imgs_gt_ft[b])
                je_attention_map = utils.aed_localization(imgs_fake_ft[b],imgs_gt_ft[b],distance_metric='je')

                # fig, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.set_title('l2-distance')
                # im1 = ax1.imshow(l2_attention_map, cmap='rainbow')
                # divider = make_axes_locatable(ax1)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig.colorbar(im1, cax=cax, orientation='vertical')
                #
                # ax2.set_title('J-entropy')
                # im2 = ax2.imshow(je_attention_map, cmap='rainbow')
                # divider = make_axes_locatable(ax2)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig.colorbar(im2, cax=cax, orientation='vertical')
                # plt.savefig(save_file + '/test_colorat_sample_{}.png'.format(str(iter_time) + '-%d' % (b)))
                for i in range(5):
                    plt.imshow(gt[i,:,:,1].astype(np.float64))
                    plt.imshow(l2_attention_map, cmap='rainbow',alpha=0.1)
                    plt.axis("off")
                    plt.savefig(save_file + '/l2_tmp_{}.png'.format(str(iter_time) + '-%d' % (b)))

                    plt.imshow(gt[i,:,:,1].astype(np.float64))
                    plt.imshow(je_attention_map, cmap='rainbow',alpha=0.1)
                    plt.axis("off")
                    plt.savefig(save_file + '/je_tmp_{}.png'.format(str(iter_time) + '-%d' % (b)))

                # for i in range(5):
                #     template[0:height,i*width:(i+1)*width] = predict[i]
                #     template[height:2*height, i * width: (i + 1) * width] = input[i]
                #     template[2*height:3*height, i * width: (i + 1) * width] = gt[i]
                #     cv2.imwrite(save_file + '/test_l2_sample_{}.png'.format(str(iter_time)+'-%d-%d'%(i,b)),template)
                #     cv2.imwrite(save_file + '/test_ge_sample_{}.png'.format(str(iter_time)+'-%d-%d'%(i,b)), map_template)