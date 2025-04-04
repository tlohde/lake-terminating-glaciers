#!/usr/bin/env bash

#SBATCH --ntasks=8
#SBATCH --mem=32G

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/s1759665/micromamba/envs/paper2
# python src/make_dirs.py --centrelines data/streams_v3.geojson

# printf "\nworking on: $1\n"
# printf "\ndownloading tiles\n"
# python src/dem_download_tiles.py --directory $1 --months 4 5 6 7 8 9 10 --buffer 5000

# printf "\ngetting masks\n"
# python src/dem_get_masks.py --directory $1

# printf "\ncoregistering\n"
# python src/dem_coregister.py --directory $1

# printf "\nstacking\n"
# python src/dem_stacking.py --directory $1

# printf "\ncomputing trends\n"
# python src/dem_trends.py --directory $1

# printf "\nsampling and exprting dataframe\n"
# python src/dem_make_sec_df.py --directory $1

# printf "\nall done\n"
# echo "tidying up"
# python src/dem_cleanup.py --directory $1


# if you're brave enough to trust the code to whip through everything in one hit:
# glacier=$(find data/ -type d -name "id*")

for g in $glacier; do

    echo "working on: $g"
#     echo "downloading tiles"
#     python src/dem_download_tiles.py --directory $g --months 6 7 8 9 --buffer 5000

#     echo "getting masks"
#     python src/dem_get_masks.py --directory $g

#     echo "coregistering"
#     python src/dem_coregister.py --directory $g

#     echo "stacking"
#     python src/dem_stacking.py --directory $g

#     echo "computing trends"
#     python src/dem_trends.py --directory $g

    echo "tidying up"
    python src/dem_cleanup.py --directory $g

done


