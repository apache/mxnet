# MXNet.io v2

1. Install Jekyll https://jekyllrb.com/docs/installation/

This is for hosting the mxnet.io beta website

serve for test:
```
cd src && JEKYLL_ENV=development bundle exec jekyll serve
```

build for beta github pages:
```
cd src && JEKYLL_ENV=production bundle exec jekyll build --config _config_beta.yml -d ../docs && cd ..
```


build for release:
```
cd src && JEKYLL_ENV=production bundle exec jekyll build --config _config_prod.yml -d ../release && cd ..
```

test:

https://thomasdelteil.github.io/mxnet.io-v2/
