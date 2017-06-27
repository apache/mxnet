unzip_data <- function(root_path){
  stopifnot(file.exists("train.zip"))
  stopifnot(file.exists("validate.zip"))
  
  if (!dir.exists(root_path))
    dir.create(root_path)
  
  unzip("sample_submission_validate.csv.zip", exdir = root_path)
  unzip("train.csv.zip", exdir = root_path)
  
  if (!dir.exists(paste0(root_path, "/train")))
    unzip("train.zip", exdir = root_path)
  if (!dir.exists(paste0(root_path, "/validate"))) {
    unzip("validate.zip", exdir = root_path)
  }
}

get_frames <- function(root_path){
  list.files(root_path, full.names = TRUE) %>>% file.info %>>% (rownames(.)[.$isdir]) %>>% 
    lapply(function(subfolder){
      list.dirs(subfolder, TRUE, TRUE) %>>% `[`(str_detect(., "sax_\\d+$")) %>>%
        lapply(function(subfolder2){
          list.files(subfolder2, "-\\d{4}.dcm$", recursive = TRUE, full.names = TRUE)
        })
    }) %>>% do.call(what = c)
}

write_label_csv <- function(fname, frames, label_map_file = NULL) {
  
  index <- sapply(frames, function(x) sapply(str_split(x, "/"), `[`, 3)) %>>% as.integer
  if (is.null(label_map_file)) {
    fwrite(data.table(index, 0, 0), fname, col.names = FALSE)
  } else {
    fwrite(fread(label_map_file)[index, ], fname, col.names = FALSE)
  }
  invisible(TRUE)
}

encode_csv <- function(label_csv, systole_csv, diastole_csv) {
  labelData <- fread(label_csv)
  systole_encode <- sapply(labelData$V2, `<`, 1:600) %>>% (matrix(as.integer(.), nrow(.))) %>>% t
  diastole_encode <- sapply(labelData$V3, `<`, 1:600) %>>% (matrix(as.integer(.), nrow(.))) %>>% t
  fwrite(data.table(systole_encode), systole_csv, col.names = FALSE)
  fwrite(data.table(diastole_encode), diastole_csv, col.names = FALSE)
  invisible(TRUE)
}

resizeImage <- function(image, targetSize) {
  minDim <- min(dim(image))
  stPixel <- (dim(image) - minDim) / 2 + 1
  tmp <- image[stPixel[1]:(minDim + stPixel[1]-1) , stPixel[2]:(minDim + stPixel[2]-1)]
  
  outGird <- mapply(function(ts, os) (1:ts - 0.5) * (os / ts) - 0.5, 
                    targetSize, dim(tmp), SIMPLIFY = FALSE)
  resultImg <- interp2(0:(nrow(tmp)-1), 0:(ncol(tmp)-1), tmp, outGird[[1]], outGird[[2]]) %>>%
    `*`(255) %>>% as.integer %>>% matrix(targetSize[1])
  return(resultImg)
}

write_data_csv <- function(fname, frames, preproc) {
  clusterExport(cl, "preproc", environment())
  data <- parLapply(cl, frames, function(path){
    lapply(path, function(imgFile){
      img <- readDICOMFile(imgFile)$img %>>% `[`(rev(1:nrow(.)), 1:ncol(.))
      if (diff(dim(img)) < 0) img <- t(img)
      as.vector(preproc(img / max(img)))
    }) %>>% do.call(what = c)
  })
  fwrite(data.table(do.call(rbind, data)), fname, col.names = FALSE)
  invisible(TRUE)
}

library(pipeR)
library(stringr)
library(parallel)
library(data.table)

# unzip train.zip and validate.zip
unzip_data("data")

# Load the list of all the training frames, and shuffle them
# Shuffle the training frames
set.seed(10)
train_frames <- get_frames("data/train") %>>% `[`(sample.int(length(.), length(.)))
validate_frames <- get_frames("data/validate") %>>% `[`(sample.int(length(.), length(.)))

# Write the corresponding label information of each frame into file.
write_label_csv("train-label.csv", train_frames, "data/train.csv")
write_label_csv("validate-label.csv", validate_frames)

# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv("train-label.csv", "train-systole.csv", "train-diastole.csv")

# open cluster for parallel processing
cl <- makeCluster(detectCores() - 1L)
tmpOutput <- clusterEvalQ(cl, {
  library(oro.dicom)
  library(data.table)
  library(pipeR)
  library(Rcpp)
  library(RcppArmadillo)
  Sys.setenv(PKG_CPP_FLAG = "-Icpp")
  sourceCpp("cpp/exportFuncs.cpp")
})
clusterExport(cl, "resizeImage")

# Dump the data of each frame into a CSV file, apply crop to 64 preprocessor
write_data_csv("train-64x64-data.csv", train_frames, function(img) resizeImage(img, c(64, 64)))
write_data_csv("validate-64x64-data.csv", train_frames, function(img) resizeImage(img, c(64, 64)))

# stop cluster
stopCluster(cl)

