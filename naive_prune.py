# Implementation of Naive Pruning

from model.srgan import generator
from train import SrganGeneratorTrainer

from data import DIV2K

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

pre_trainer.model.save_weights('weights/srgan/pre_generator_6.h5')

from model.srgan import generator, discriminator
from train import SrganTrainer

gan_generator = generator(num_res_blocks=8)
gan_generator.load_weights('weights/srgan/pre_generator_6.h5')

gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())

gan_trainer.train(train_ds, steps=200000)

# Save weights of generator and discriminator.
gan_trainer.generator.save_weights('weights/srgan/gan_generator_6.h5')
gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator_6.h5')
