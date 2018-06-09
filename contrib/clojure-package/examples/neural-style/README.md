# neural-style

An example of neural style transfer

## Usage

use the `download.sh` script to get the params file and the input and output file

Then use `lein run`

The output images will be stored in the output directory. Please feel free to play with the params at the top of the file


This example only works on 1 device (cpu) right now

If you are running on AWS you will need to setup X11 for graphics
`sudo apt install xauth x11-apps`

then relogin in `ssh -X -i creds ubuntu@yourinstance`


_Note: This example is not working all the way - it needs some debugging help_


