#!/bin/bash
# 5-fold CV by star: two folds at a time, 2 threads each (4 cores). Each fold skips itself if its json exists.
cd /home/user/tv/cv
export TV_THREADS=2
for k in 0 2 4; do
  python3 cv.py kepler $k > kepler_$k.log 2>&1 &
  [ $k -lt 4 ] && python3 cv.py kepler $((k+1)) > kepler_$((k+1)).log 2>&1 &
  wait
done
python3 cv.py tess 0 > tess_0.log 2>&1 & python3 cv.py tess 1 > tess_1.log 2>&1 & wait
python3 cv.py tess 2 > tess_2.log 2>&1 & python3 cv.py tess 3 > tess_3.log 2>&1 & wait
python3 cv.py tess 4 > tess_4.log 2>&1
python3 cv.py summary > summary.log 2>&1
echo ALLDONE >> summary.log
