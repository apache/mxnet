# Material Design HTML Theme for Sphinx

## How to use

- Install the theme by

```bash
pip install mxtheme
```

- Modify the `conf.py` for your sphinx project by

create a submodule of this repo on the same folder with `conf.py` for your sphinx project. then modify the following three lines in `conf.py`:

```python
html_theme = 'mxtheme'
```

In addition, to use the `card` directive in rst, you can and add the following two lines into your `def setup(app)` function:

```python
def setup(app):
    ...
    import mxtheme
    app.add_directive('card', mxtheme.CardDirective)
```

## How to build


Install `npm` first,

on ubuntu:

```
wget -qO- https://deb.nodesource.com/setup_8.x | sudo -E bash -
sudo apt-get install -y nodejs
```

on macos

```
brew install nodejs
```

Then install packages

```
npm install
```

Last, build css and js


```
npm run build
```

## Acknowledgment


This is fork of
[sphinx_materialdesign_theme](https://github.com/myyasuda/sphinx_materialdesign_theme). With
some CSS/JS modifications. Please refer to the original project for more
documents.
