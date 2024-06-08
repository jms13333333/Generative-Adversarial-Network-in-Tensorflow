!mkdir images

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

class MNISTGAN():
    def build_generator(self):
      input = Input(shape=(self.latent_dim,))
      x = Dense(64, activation='relu')(input)
      x = Dense(128, activation='relu')(x)
      x = Dense(784, activation='sigmoid')(x)

      generated_image = Reshape(self.shapes)(x)
      return Model(input,generated_image)

    def build_discriminator(self):
      input = Input(shape=self.shapes)
      x = Flatten()(input)
      x = Dense(128, activation='relu')(x)
      x = Dense(64, activation='relu')(x)
      validity = Dense(1, activation='sigmoid')(x)
      return Model(input, validity)

    def __init__(self, latent_dim, shape):
      self.rows = 28
      self.cols = 28
      self.channels = 1
      self.shapes = (self.rows, self.cols, self.channels)
      self.latent_dim = 100
      optimizer = Adam(.0002,.5)
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
      self.generator = self.build_generator()
      self.generator.compile(loss='binary_crossentropy', optimizer=optimizer,)

      z = Input(shape=(self.latent_dim,))
      generated_image = self.generator(z)

      validity = self.discriminator(generated_image)

      self.combined = Model(z, validity)
      optimizer.build(self.combined.trainable_variables)
      self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, epochs, batch_size, sample_interval):
      (X_train, _), (_, _) = mnist.load_data()
      X_train = X_train / 255
      X_train = np.expand_dims(X_train, axis=3)
      valid = np.ones((batch_size,1))
      fake = np.zeros((batch_size,1))
      allDloss = []
      allGloss = []
      for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        generated_images = self.generator.predict(noise)
        gen_imgs = self.generator.predict(noise)
        self.discriminator.trainable = True
        d_loss_real = self.discriminator.train_on_batch(real_images, valid)
        d_loss_fake = self.discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        self.discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        g_loss1 = self.combined.train_on_batch(noise, valid)
        g_loss2 = self.combined.train_on_batch(noise,valid)
        g_loss3 = self.combined.train_on_batch(noise,valid)

        g_loss = (g_loss2+g_loss1+g_loss3)/3

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        allDloss.append(d_loss)
        allGloss.append(g_loss)
        if epoch % sample_interval == 0:
          self.sample_images(epoch)
        if epoch == epochs-1:
          self.sample_images(epoch)
      np.save('d_loss.npy',np.array(allGloss))
      np.save('g_loss.npy',np.array(allDloss))

    def sample_images(self, epoch):
      r, c = 5, 5
      noise = np.random.normal(0, 1, (r * c, self.latent_dim))
      generated_images = self.generator.predict(noise)
      generated_images = 0.5 * generated_images + 0.5
      fig, axs = plt.subplots(r, c)
      cnt = 0
      for i in range(r):
        for j in range(c):
          axs[i,j].imshow(generated_images[cnt, :,:,0], cmap='gray')
          axs[i,j].axis('off')
          cnt += 1
      plt.savefig('images/mnist_%d.png' % epoch)
      plt.close()

if __name__== '__main__':
  mnist_gan = MNISTGAN(latent_dim=100, shape=(28,28,1))
  mnist_gan.generator.summary()
  mnist_gan.train(epochs=30000, batch_size=32, sample_interval=200)
