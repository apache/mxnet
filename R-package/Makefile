rcpplint:
	./3rdparty/dmlc-core/scripts/lint.py mxnet-rcpp all R-package/src

rpkg:
	mkdir -p R-package/inst/libs
	cp src/io/image_recordio.h R-package/src
	if [ -d "lib" ]; then \
		cp -rf lib/libmxnet.so R-package/inst/libs; \
		if [ -e "lib/libtvm_runtime.so" ]; then \
			cp -rf lib/libtvm_runtime.so R-package/inst/libs; \
		fi; \
	else \
		cp -rf build/libmxnet.so R-package/inst/libs; \
		if [ -e "build/libtvm_runtime.so" ]; then \
			cp -rf build/libtvm_runtime.so R-package/inst/libs; \
		fi; \
	fi

	mkdir -p R-package/inst/include
	cp -rl include/* R-package/inst/include
	Rscript -e "if(!require(devtools)){install.packages('devtools', repo = 'https://cloud.r-project.org/')}"
	Rscript -e "if(!require(roxygen2)||packageVersion('roxygen2') < '6.1.1'){install.packages('roxygen2', repo = 'https://cloud.r-project.org/')}"
	Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cloud.r-project.org/')); install_deps(pkg='R-package', dependencies = TRUE)"
	cp R-package/dummy.NAMESPACE R-package/NAMESPACE  # NAMESPACE will be replaced by devtools::document later
	echo "import(Rcpp)" >> R-package/NAMESPACE
	R CMD INSTALL R-package
	Rscript -e "require(mxnet); mxnet:::mxnet.export('R-package'); warnings()"
	Rscript -e "devtools::document('R-package');warnings()"
	R CMD INSTALL R-package

rpkgtest:
	Rscript -e 'require(testthat);res<-test_dir("R-package/tests/testthat");if(!testthat:::all_passed(res)){stop("Test failures", call. = FALSE)}'
	Rscript -e 'res<-covr:::package_coverage("R-package");fileConn<-file(paste("r-package_coverage_",toString(runif(1)),".json"));writeLines(covr:::to_codecov(res), fileConn);close(fileConn)'
