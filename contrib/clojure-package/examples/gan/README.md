# gan

This is an example of how to do a GAN with the MNIST data

## Usage

Do `lein run` and the images generated will be in the `results` directory. The gout* images are the ones generated, the diff* images are the visualization of the input gradient different fed to the generator

`lein run :gpu` will run on gpu

If you are running on AWS you will need to setup X11 for graphics
`sudo apt install xauth x11-apps`

then relogin in `ssh -X -i creds ubuntu@yourinstance`


