# TyreflawDetectionGAN
Due to the limited size of the dataset(flaw images), we enlarge the dataset with GAN to train CNN for tyre flaw detection.
## GAN
### Generator
- The generator uses U-net with skip connection. The input consists of a tag image and a noise image. An encoder and a decoder together build up the generator. Each layer in the encoder is connected to the corresponding layer in the decoder to enrich the information of the upsampling process.
- After training we will get a generator, which can generate artificial flaw images from tag images. With the artificial images we enlarge the dataset to train the CNN.
### Discriminator
- The discriminator uses Patch-GAN, whose output is a matrix. Each entry in the matrix represents the judging result(the discriminator here judges if the input image is artificial or not) of a certain part(sensing field) of the input. The tag image is also connected to the image that is ready to be discriminated as a part of the input.
## CNN
- The CNN uses a normal structure with 3 convolutive layers, 2 pool layers and 1 full connection layer.
