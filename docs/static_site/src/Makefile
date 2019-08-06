all: html

html:
	mkdir -p build
	cd src && bundle install && JEKYLL_ENV=production bundle exec jekyll build --config _config_prod.yml -d ../build/html && cd ..


clean:
	rm -rf build
