# Bash script to convert .md files to .ipynb for Sphinx. Must be run inside of tutorial/ subdirectory.
for file in *.md; do
    python ./scripts/md2ipynb4r.py $file "$(basename "$file" .md).ipynb"
    echo "Converted notebook $file"
done

# python ./scripts/md2ipynb4r.py $i .ipynb

# Usage: 
# cd ./source/tutorial/
# bash ./scripts/convertNotebooks.sh