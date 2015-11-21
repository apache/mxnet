fnames <- list.files("R-package/vignettes/", pattern="*.Rmd")
sapply(fnames, function(x){
	knitr::purl(paste0("R-package/vignettes/", x))
	})