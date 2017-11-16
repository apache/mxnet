# Convert all .flac files within this folder to .wav files

find . -iname "*.flac" | wc

for flacfile in `find . -iname "*.flac"`
do
    sox "${flacfile%.*}.flac" -e signed -b 16 -c 1 -r 16000 "${flacfile%.*}.wav"
done
