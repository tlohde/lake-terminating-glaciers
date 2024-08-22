#!/usr/bin/env bash

#SBATCH --ntasks=8
#SBATCH --mem=32G

source /opt/conda/etc/profile.d/conda.sh
conda activate /home/s1759665/micromamba/envs/paper2

printf "\nworking on id0_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id0_-138892x_-3197992y

printf "\nworking on id10\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id10_499410x_-1351353y

printf "\nworking on id11\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id11_497441x_-1338620y

printf "\nworking on id12\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id12_-581577x_-1263468y

printf "\nworking on id13\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id13_143649x_-865563y

printf "\nworking on id14\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id14_-239437x_-2838869y

printf "\nworking on id15\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id15_4515x_-3211902y

printf "\nworking on id1_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id1_6685x_-3188046y

printf "\nworking on id16\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id16_-9318x_-3152179y

printf "\nworking on id17\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id17_-167858x_-3142959y

printf "\nworking on id18\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id18_-280014x_-2902289y

printf "\nworking on id19\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id19_-263550x_-2709771y

printf "\nworking on id20\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id20_-225808x_-2493072y

printf "\nworking on id21\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id21_288157x_-930867y

printf "\nworking on id2_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id2_-194085x_-3077800y

printf "\nworking on id22\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id22_-281294x_-1899575y

printf "\nworking on id23\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id23_1378285x_-2468304y

printf "\nworking on id24\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id24_1380703x_-2459835y

printf "\nworking on id25\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id25_1383119x_-2447526y

printf "\nworking on id26\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id26_1365033x_-2493323y

printf "\nworking on id27\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id27_1364036x_-2507363y

printf "\nworking on id28\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id28_1365106x_-2513662y

printf "\nworking on id29\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id29_1368245x_-2521726y

printf "\nworking on id30\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id30_1386295x_-2438214y

printf "\nworking on id3_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id3_-254432x_-3020649y

printf "\nworking on id4_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id4_-229209x_-2972450y

printf "\nworking on id5_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id5_-226289x_-2968847y

printf "\nworking on id6_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id6_-246487x_-2871885y

printf "\nworking on id7_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id7_-289260x_-1889815y

printf "\nworking on id8_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id8_607229x_-1847482y

printf "\nworking on id9_\n"
/usr/bin/env time python src/dem_make_sec_df.py --directory data/id9_484789x_-1406074y