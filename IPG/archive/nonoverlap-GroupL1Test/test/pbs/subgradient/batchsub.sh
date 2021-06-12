# files=`ls ./*.pbs`
# for file in $files
# do
#   echo "submit $file ..."
#   qsub $file
# done
dataset=$1
files=`ls ./*.pbs | grep $1`
total_jobs=`ls ./*.pbs | grep $1 | wc -l`
for file in $files
do
  echo "submit $file ..."
  qsub $file
done
echo "submit $total_jobs jobs"
