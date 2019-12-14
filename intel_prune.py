# Implementation of Intelligent Pruning
import tensorflow as tf
from model.srgan import generator
from train import SrganGeneratorTrainer

from data import DIV2K

from tensorflow_model_optimization.sparsity import keras as sparsity

train_loader = DIV2K(scale=4,             
                    downgrade='bicubic', 
                    subset='train')      

train_ds = train_loader.dataset(batch_size=16,         
                                random_transform=True, 
                                repeat_count=None)     
valid_loader = DIV2K(scale=4,            
                    downgrade='bicubic', 
                    subset='valid')      

valid_ds = valid_loader.dataset(batch_size=1,           
                                random_transform=False,
                                repeat_count=1)         

pre_trainer = SrganGeneratorTrainer(model=generator(num_res_blocks=6), checkpoint_dir=f'.ckpt/pre_generator')

pre_trainer.train(train_ds, valid_ds.take(10), steps=1000000, evaluate_every=1000)

pre_trainer.model.save_weights('weights/srgan/pre_generator_i.h5')

from model.srgan import generator,discriminator
from train import SrganTrainer

generator = tf.unstack(tf.transpose(generator, (1, 0, 2)))

for i in range(tf.size(generator)):
    generator[i] = sparsity.prune_low_magnitude(generator[i])

generator = tf.stack(tf.transpose(generator, (1, 0, 2)))

gan_generator = sparsity.strip_pruning(generator)
gan_generator.load_weights('weights/srgan/pre_generator_i.h5')

gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

gan_trainer.train(train_ds, steps=200000)

# Save weights of generator and discriminator.
gan_trainer.generator.save_weights('weights/srgan/gan_generator_i.h5')
gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator_i.h5')
