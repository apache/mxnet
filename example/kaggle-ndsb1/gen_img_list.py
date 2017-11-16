from __future__ import print_function
import csv
import os
import sys
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='generate train/test image list files form input directory. If training it will also split into tr and va sets.')
parser.add_argument('--image-folder', type=str, default="data/train/",
                    help='the input data directory')
parser.add_argument('--out-folder', type=str, default="data/",
                    help='the output folder')
parser.add_argument('--out-file', type=str, default="train.lst",
                    help='the output lst file')
parser.add_argument('--train', action='store_true',
                    help='if we are generating training list and hence we have to loop over subdirectories')
## These options are only used if we are doing training lst
parser.add_argument('--percent-val', type=float, default=0.25,
                    help='the percentage of training list to use as validation')
parser.add_argument('--stratified', action='store_true',
                    help='if True it will split train lst into tr and va sets using stratified sampling')
args = parser.parse_args()

random.seed(888)

fo_name=os.path.join(args.out_folder+args.out_file)
fo = csv.writer(open(fo_name, "w"), delimiter='\t', lineterminator='\n')
    
if args.train:
    tr_fo_name=os.path.join(args.out_folder+"tr.lst")
    va_fo_name=os.path.join(args.out_folder+"va.lst")
    tr_fo = csv.writer(open(tr_fo_name, "w"), delimiter='\t', lineterminator='\n')
    va_fo = csv.writer(open(va_fo_name, "w"), delimiter='\t', lineterminator='\n')

#check sampleSubmission.csv from kaggle website to view submission format
head = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')

# make image list
img_lst = []
cnt = 0
if args.train:
    for i in xrange(len(head)):
        path = args.image_folder + head[i]
        lst = os.listdir(args.image_folder + head[i])
        for img in lst:
            img_lst.append((cnt, i, path + '/' + img))
            cnt += 1
else:
    lst = os.listdir(args.image_folder)
    for img in lst:
        img_lst.append((cnt, 0, args.image_folder + img))
        cnt += 1

# shuffle
random.shuffle(img_lst)

#write
for item in img_lst:
    fo.writerow(item)
        


## If training, split into train and validation lists (tr.lst and va.lst)
## Optional stratified sampling

if args.train:
    img_lst=np.array(img_lst)
    if args.stratified:
        from sklearn.cross_validation import StratifiedShuffleSplit
        ## Stratified sampling to generate train and validation sets
        labels_train=img_lst[:,1]
        # unique_train, counts_train = np.unique(labels_train, return_counts=True) # To have a look at the frecuency distribution
        sss = StratifiedShuffleSplit(labels_train, 1, test_size=args.percent_val, random_state=0)
        for tr_idx, va_idx in sss:
            print("Train subset has ", len(tr_idx), " cases. Validation subset has ", len(va_idx), "cases")
    else:
        (nRows, nCols) = img_lst.shape
        splitat=int(round(nRows*(1-args.percent_val),0))
        tr_idx=range(0,splitat)
        va_idx=range(splitat,nRows)
        print("Train subset has ", len(tr_idx), " cases. Validation subset has ", len(va_idx), "cases")

    tr_lst=img_lst[tr_idx,:].tolist()
    va_lst=img_lst[va_idx,:].tolist()
    for item in tr_lst:
        tr_fo.writerow(item)
    for item in va_lst:
        va_fo.writerow(item)


