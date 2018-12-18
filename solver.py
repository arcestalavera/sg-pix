import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torchvision.utils import save_image
from utils import *
from data_loader import *
from tqdm import tqdm
from imageio import imwrite

# from model import Sg2ImModel
# from network.discriminators import PatchDiscriminator, AcCropDiscriminator
# from network.bilinear import crop_bbox_batch
from network.model import Sg2ImModel
from losses import get_gan_losses, VGGLoss
import network.pix2pix.networks as ref_network


class Solver(object):
    DEFAULTS = {}

    def __init__(self, vocab, train_loader, test_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Load Vocab for model and data loader
        self.vocab = vocab

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        # Generator = Scene Graph Model
        kwargs = {
            'vocab': self.vocab,
            'image_size': self.image_size,
            'embedding_dim': self.embedding_dim,
            'gconv_dim': self.gconv_dim,
            'gconv_hidden_dim': self.gconv_hidden_dim,
            'gconv_num_layers': self.gconv_num_layers,
            'mlp_normalization': self.mlp_normalization,
            'normalization': self.normalization,
            'activation': self.activation,
            'mask_size': self.mask_size,
            'layout_noise_dim': self.layout_noise_dim,
            'netG': self.netG,
            'ngf': self.ngf,
            'n_downsample_global': self.n_downsample_global,
            'n_blocks_global': self.n_blocks_global,
            'n_blocks_local': self.n_blocks_local,
            'n_local_enhancers': self.n_local_enhancers,
        }

        self.generator = Sg2ImModel(**kwargs)
        # Discriminators
        # OBJ Discriminator
        # self.obj_discriminator = AcCropDiscriminator(vocab=self.vocab,
        #                                              arch=self.d_obj_arch,
        #                                              normalization=self.d_normalization,
        #                                              activation=self.d_activation,
        #                                              padding=self.d_padding,
        #                                              object_size=self.crop_size)

        self.obj_discriminator = ref_network.define_obj_D(vocab=self.vocab,
                                                       input_nc=3,
                                                       crop_size=self.crop_size,
                                                       ndf=self.ndf,
                                                       n_layers_D=self.n_layers_D_obj,
                                                       norm=self.normalization,
                                                       num_D=self.num_D_obj,
                                                       getIntermFeat=not self.no_objFeat_loss)

        # IMG Discriminator
        self.img_discriminator = ref_network.define_img_D(input_nc=3,
                                                          ndf=self.ndf,
                                                          n_layers_D=self.n_layers_D_img,
                                                          norm=self.normalization,
                                                          num_D=self.num_D_img,
                                                          getIntermFeat=not self.no_imgFeat_loss)

        # Loss
        self.gan_g_loss, self.gan_d_loss = get_gan_losses(self.gan_loss_type)
        self.vgg_loss = VGGLoss()

        # Weights
        # feature matching loss weights
        self.img_feat_weights = 4.0 / (self.n_layers_D_img + 1)
        self.img_D_weights = 1.0 / self.num_D_img
        self.obj_feat_weights = 4.0 / (self.n_layers_D_obj + 1)
        self.obj_D_weights = 1.0 / self.num_D_obj

        # Mode of Model
        if self.init_epochs >= self.eval_mode_after:
            print("Setting Generator to Eval Mode")
            self.generator.eval()
        else:
            print("Setting Generator to Train Mode")
            self.generator.train()

        self.obj_discriminator.train()
        self.img_discriminator.train()

        # Optimizers
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate)
        self.dis_obj_optimizer = torch.optim.Adam(
            self.obj_discriminator.parameters(), lr=self.learning_rate)
        self.dis_img_optimizer = torch.optim.Adam(
            self.img_discriminator.parameters(), lr=self.learning_rate)

        # Others
        self.ac_ind = 1
        self.obj_ind = 0
        if not self.no_objFeat_loss:
            self.ac_ind = 2
            self.obj_ind = 1

        # Print networks
        self.print_network(self.generator, 'Generator')
        self.print_network(self.obj_discriminator, 'Object Discriminator')
        self.print_network(self.img_discriminator, 'Image Discriminator')

        if torch.cuda.is_available():
            self.generator.cuda()
            self.obj_discriminator.cuda()
            self.img_discriminator.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.generator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.img_discriminator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D_IMG.pth'.format(self.pretrained_model))))
        self.obj_discriminator.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D_OBJ.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.generator.zero_grad()
        self.img_discriminator.zero_grad()
        self.obj_discriminator.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def lr_scheduler(self, lr_sched):
        new_lr = lr_sched.compute_lr()
        self.update_lr(new_lr)

    def add_loss(self, total_loss, curr_loss, loss_dict, loss_name, weight=1):
        curr_loss = curr_loss * weight
        loss_dict[loss_name] = curr_loss.item()
        if total_loss is not None:
            total_loss += curr_loss
        else:
            total_loss = curr_loss
        return total_loss

    def log_losses(self, loss, module, losses):
        for loss_name, loss_val in losses.items():
            # print("LOSS {}: {}".format(loss_name,loss_val))
            loss["{}/{}".format(module, loss_name)] = loss_val

        return loss

    def train(self): 
        iters_per_epoch = len(self.train_loader)
        print("Iterations per epoch: " + str(iters_per_epoch))

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0
        fixed_batch = self.build_sample_images(self.test_loader)

        # Start training
        e_init = self.init_epochs
        iter_ctr = e_init * iters_per_epoch
        print("current iteration: {}".format(iter_ctr))
        # e = iter_ctr // iters_per_epoch
        start_time = time.time()
        # while True:
        # Stop training if iter_ctr reached num_iterations
        # if iter_ctr >= self.num_iterations:
        #     break
        # e += 1
        for e in range(e_init, self.num_epochs):
            if (e + 1) == self.eval_mode_after:
                self.generator.eval()
                self.gen_optimizer = torch.optim.Adam(
                    self.generator.parameters(), lr=self.learning_rate)

            for i, batch in enumerate(tqdm(self.train_loader)):
                masks = None
                imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch

                start = time.time()

                # Prepare Data
                imgs = to_var(imgs)
                objs = to_var(objs)
                boxes = to_var(boxes)
                triples = to_var(triples)
                obj_to_img = to_var(obj_to_img)
                triple_to_img = to_var(triple_to_img)
                masks = to_var(masks)
                predicates = triples[:, 1]  # get p from triples(s, p ,o)

                # variables needed for generator and discriminator steps
                step_vars = (imgs, objs, boxes, obj_to_img, predicates, masks)

                # Forward to Model
                model_boxes = boxes
                model_masks = masks
                model_out = self.generator(
                    objs, triples, obj_to_img, boxes_gt=model_boxes, masks_gt=model_masks)

                # Generator Step
                total_loss, losses = self.generator_step(step_vars, model_out)
                # Logging
                loss = {}

                loss['G/total_loss'] = total_loss.data.item()
                loss = self.log_losses(loss, 'G', losses)

                # Discriminator Step
                total_loss, dis_obj_losses, dis_img_losses = self.discriminator_step(
                    step_vars, model_out)

                loss['D/total_loss'] = total_loss.data.item()
                loss = self.log_losses(loss, 'D', dis_obj_losses)
                loss = self.log_losses(loss, 'D', dis_img_losses)

                # Print out log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs * iters_per_epoch) - (e *
                                                                         iters_per_epoch + i)) * elapsed / (e * iters_per_epoch + i + 1)
                    epoch_time = (iters_per_epoch - i) * \
                        elapsed / (e * iters_per_epoch + i + 1)

                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, epoch_time, total_time, e + 1, self.num_epochs, i + 1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(
                                tag, value, e * iters_per_epoch + i + 1)

                # Save model checkpoints
                if (iter_ctr + 1) % self.model_save_step == 0:
                    torch.save(self.generator.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e, iter_ctr + 1)))
                    torch.save(self.obj_discriminator.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_D_OBJ.pth'.format(e, iter_ctr + 1)))
                    torch.save(self.img_discriminator.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_D_IMG.pth'.format(e, iter_ctr + 1)))

                if (iter_ctr + 1) % self.sample_step == 0:
                    self.sample_images(fixed_batch, iter_ctr)

                iter_ctr += 1

    def discriminator_step(self, step_vars, model_out):
        ac_loss_real = None
        ac_loss_fake = None
        d_obj_losses = None
        d_img_losses = None

        imgs, objs, boxes, obj_to_img, _, _ = step_vars
        imgs_pred = model_out[0].detach()

        # Step for Obj Discriminator
        if self.obj_discriminator is not None:
            d_obj_losses = LossManager()
            scores_fake = self.obj_discriminator(
                imgs_pred, boxes, obj_to_img)
            scores_real = self.obj_discriminator(
                imgs, boxes, obj_to_img)

            ac_loss_fake = torch.zeros(1).to(imgs)
            ac_loss_real = torch.zeros(1).to(imgs)
            d_obj_gan_loss = torch.zeros(1).to(imgs)

            for i in range(self.num_D_obj):
                # AC Losses
                ac_score_fake = scores_fake[i][self.ac_ind]
                ac_score_real = scores_real[i][self.ac_ind]
                ac_loss_fake += F.cross_entropy(ac_score_fake, objs)
                ac_loss_real += F.cross_entropy(ac_score_real, objs)

                # OBJ GAN Loss
                obj_score_fake = scores_fake[i][self.obj_ind]
                obj_score_real = scores_real[i][self.obj_ind]
                d_obj_gan_loss += self.gan_d_loss(obj_score_real, obj_score_fake)

            d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
            d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
            d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

            self.reset_grad()
            d_obj_losses.total_loss.backward()
            self.dis_obj_optimizer.step()

        # Step for Img Discriminator
        if self.img_discriminator is not None:
            d_img_losses = LossManager()
            scores_fake = self.img_discriminator(imgs_pred)

            scores_real = self.img_discriminator(imgs)

            # GAN Loss
            d_img_loss = torch.zeros(1).to(imgs)
            for i in range(self.num_D_img):
                img_fake_score = scores_fake[i]
                img_real_score = scores_real[i]
                d_img_loss += self.gan_d_loss(img_real_score[-1], img_fake_score[-1])

            d_img_losses.add_loss(d_img_loss, 'd_img_gan_loss')

            self.reset_grad()
            d_img_losses.total_loss.backward()
            self.dis_img_optimizer.step()

        total_loss = d_obj_losses.total_loss + d_img_losses.total_loss
        return total_loss, d_obj_losses, d_img_losses

    def generator_step(self, step_vars, model_out):
        losses = {}

        imgs, objs, boxes, obj_to_img, predicates, masks = step_vars
        imgs_pred, boxes_pred, masks_pred, predicate_scores, layout = model_out

        total_loss = torch.zeros(1).to(imgs)
        skip_pixel_loss = (boxes is None)
        # Pixel Loss
        l1_pixel_weight = self.l1_pixel_loss_weight
        if skip_pixel_loss:
            l1_pixel_weight = 0

        l1_pixel_loss = F.l1_loss(imgs_pred, imgs)

        total_loss = self.add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                                   l1_pixel_weight)

        # Box Loss
        loss_bbox = F.mse_loss(boxes_pred, boxes)
        total_loss = self.add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                                   self.bbox_pred_loss_weight)

        if self.predicate_pred_loss_weight > 0:
            loss_predicate = F.cross_entropy(predicate_scores, predicates)
            total_loss = self.add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                                       self.predicate_pred_loss_weight)

        if self.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
            # Mask Loss
            mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
            total_loss = self.add_loss(total_loss, mask_loss, losses, 'mask_loss',
                                       self.mask_loss_weight)

        # GAN LOSSES
        if self.obj_discriminator is not None:
            # OBJ AC Loss: Classification of Objects
            scores_fake = self.obj_discriminator(
                imgs_pred, boxes, obj_to_img)

            # OBJ GAN Loss: Auxiliary Classification
            ac_loss = 0
            for i in range(self.num_D_obj):
                ac_score = scores_fake[i][self.ac_ind]
                ac_loss += F.cross_entropy(ac_score, objs)

            total_loss = self.add_loss(total_loss, ac_loss, losses, 'g_ac_loss',
                                       self.ac_loss_weight)

            # OBJ GAN Loss: Real vs Fake
            obj_loss = 0
            for i in range(self.num_D_obj):
                obj_score = scores_fake[i][self.obj_ind]
                obj_loss += self.gan_g_loss(obj_score)

            weight = self.discriminator_loss_weight * self.d_obj_weight
            total_loss = self.add_loss(total_loss, obj_loss, losses,
                                       'g_obj_gan_loss', weight)

            # Feat Matching Loss of Intermediate Layers
            if not self.no_objFeat_loss:
                scores_real = self.obj_discriminator(imgs, boxes, obj_to_img)
                feat_loss = 0
                for i in range(self.num_D_obj):
                    fake_feat = scores_fake[i][0]
                    real_feat = scores_real[i][0]
                    for j in range(len(fake_feat) - 1):
                        feat_loss += F.l1_loss(fake_feat[j], real_feat[j].detach())
                weight = self.obj_D_weights * self.obj_feat_weights * self.lambda_feat
                total_loss = self.add_loss(total_loss, feat_loss, losses,
                                           'g_obj_feat_loss', weight)

        if self.img_discriminator is not None:
            # IMG GAN Loss: Patches should be realistic
            scores_fake = self.img_discriminator(imgs_pred)
            scores_real = self.img_discriminator(imgs)

            weight = self.discriminator_loss_weight * self.d_img_weight
            img_loss = 0
            for i in range(self.num_D_img):
                img_scores = scores_fake[i]
                img_loss += self.gan_g_loss(img_scores[-1])

            total_loss = self.add_loss(total_loss, img_loss, losses,
                                       'g_img_gan_loss', weight)

            # Feat Matching Loss of Intermediate Layers
            if not self.no_imgFeat_loss:
                feat_loss = 0
                for i in range(self.num_D_img):
                    for j in range(len(scores_fake[i]) - 1):
                        feat_loss += F.l1_loss(
                            scores_fake[i][j], scores_real[i][j].detach())
                weight = self.img_D_weights * self.img_feat_weights * self.lambda_feat
                total_loss = self.add_loss(total_loss, feat_loss, losses,
                                           'g_img_feat_loss', weight)

        if not self.no_vgg_loss:
            g_vgg_loss = self.vgg_loss(imgs_pred, imgs)

            weight = self.lambda_feat
            total_loss = self.add_loss(total_loss, g_vgg_loss, losses,
                                       'g_loss_vgg', weight)

        losses['total_loss'] = total_loss.item()

        self.reset_grad()
        total_loss.backward()
        self.gen_optimizer.step()

        return total_loss, losses

    def sample_images(self, batch, iter_ctr=1):
        self.generator.eval()
        objs, triples, obj_to_img = batch

        with torch.no_grad():
            objs = to_var(objs)
            triples = to_var(triples)
            obj_to_img = to_var(obj_to_img)

            fake_images, _, _, _, _ = self.generator(
                objs, triples, obj_to_img)

        save_image(imagenet_deprocess_batch(fake_images, convert_range=False),
                   os.path.join(self.sample_path, '{}_fake.png'.format(iter_ctr + 1)), nrow=1, padding=0)
        print('Translated images and saved into {}..!'.format(self.sample_path))

    def build_sample_images(self, data_loader):
        batch = next(iter(data_loader))

        if len(batch) == 6:
            imgs, objs, boxes, triples, obj_to_img, _ = batch
        elif len(batch) == 7:
            imgs, objs, boxes, _, triples, obj_to_img, _ = batch

        return (objs, triples, obj_to_img)

    def test(self, num_batch=None):
        self.generator.eval()
        if num_batch != None:
            stop_test = num_batch
        else:
            stop_test = len(self.test_loader)

        img_num = 0
        graph_num = 0
        for i, batch in enumerate(self.test_loader):
            if len(batch) == 6:
                imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            elif len(batch) == 7:
                imgs, objs, boxes, _, triples, obj_to_img, triple_to_img = batch

            with torch.no_grad():
                objs = to_var(objs)
                triples = to_var(triples)
                obj_to_img = to_var(obj_to_img)

                fake_images, _, _, _ = self.generator(
                    objs, triples, obj_to_img)

            # Save Images
            fake_images = imagenet_deprocess_batch(
                fake_images, convert_range=True)
            for j in range(fake_images.shape[0]):
                img = fake_images[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(
                    self.test_path, "fake_{}.png".format(img_num))
                imwrite(img_path, img)
                img_num += 1

            graph_num = self.export_relationship_graph(
                objs, triples, triple_to_img, graph_num)
            if(i + 1 >= stop_test):
                break

        print('Translated images and saved into {}..!'.format(self.test_path))

    def export_relationship_graph(self, objs, triples, triple_to_img, graph_num):
        obj_names = self.vocab['object_idx_to_name']
        pred_names = self.vocab['pred_idx_to_name']

        # Do it per imag
        with open(os.path.join(self.test_path, 'test.txt'), 'a') as graph_file:
            for i in range(max(triple_to_img) + 1):
                print("===================================", file=graph_file)
                print("image: {}".format(graph_num), file=graph_file)
                triple_inds = (triple_to_img == i).nonzero()
                img_triples = triples[triple_inds].view(-1, 3)
                # print("triples: {}".format(img_triples))
                for s, p, o in img_triples:
                    if(pred_names[p] == '__in_image__'):
                        continue
                    s_label = obj_names[objs[s]]
                    p_label = pred_names[p]
                    o_label = obj_names[objs[o]]
                    print("{} --- {} ---> {}".format(s_label,
                                                     p_label, o_label), file=graph_file)
                graph_num += 1

        return graph_num
