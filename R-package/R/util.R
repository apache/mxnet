# filter out null, keep the names
mx.util.filter.null <- function(lst) {
  Filter(Negate(is.null), lst)
}

#' Internal function to generate mxnet_generated.R
#' Users do not need to call this function.
#' @param path The path to the root of the package.
#'
#' @export
mxnet.export <- function(path) {
  mx.internal.export(path.expand(path))
}

#' Convert images into image recordio format
#' @param image_lst
#'   The image lst file
#' @param root
#'   The root folder for image files
#' @param output_rec
#'   The output rec file
#' @param label_width
#'   The label width in the list file. Default is 1.
#' @param pack_label
#'   Whether to also pack multi dimenional label in the record file. Default is 0.
#' @param new_size
#'   The shorter edge of image will be resized to the newsize. 
#'   Original images will be packed by default.
#' @param nsplit
#'   It is used for part generation, logically split the image.lst to NSPLIT parts by position.
#'   Default is 1.
#' @param partid
#'   It is used for part generation, pack the images from the specific part in image.lst.
#'   Default is 0.
#' @param center_crop
#'   Whether to crop the center image to make it square. Default is 0.
#' @param quality
#'   JPEG quality for encoding (1-100, default: 95) or PNG compression for encoding (1-9, default: 3).
#' @param color_mode
#'   Force color (1), gray image (0) or keep source unchanged (-1). Default is 1.
#' @param unchanged
#'   Keep the original image encoding, size and color. If set to 1, it will ignore the others parameters.
#' @param inter_method
#'   NN(0), BILINEAR(1), CUBIC(2), AREA(3), LANCZOS4(4), AUTO(9), RAND(10). Default is 1.
#' @param encoding
#'   The encoding type for images. It can be '.jpg' or '.png'. Default is '.jpg'.
#' @export
im2rec <- function(image_lst, root, output_rec, label_width = 1L,
                   pack_label = 0L, new_size = -1L, nsplit = 1L,
                   partid = 0L, center_crop = 0L, quality = 95L,
                   color_mode = 1L, unchanged = 0L, inter_method = 1L,
                   encoding = ".jpg") {
  image_lst <- path.expand(image_lst)
  root <- path.expand(root)
  output_rec <- path.expand(output_rec)
  mx.internal.im2rec(image_lst, root, output_rec, label_width,
                     pack_label, new_size,  nsplit, partid,
                     center_crop, quality, color_mode, unchanged,
                     inter_method, encoding)
}
