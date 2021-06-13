# files=`ls ./*.pbs`
# for file in $files
# do
#   echo "submit $file ..."
#   qsub $file
# done
for file in $(cat finished.txt)
do 
  # echo "search $file ..."
  if [ -e $file ]; then
    echo "rm $file ..."
    rm $file
  fi
done

dataset=$1
files=`ls ./*.pbs | grep $1`
total_jobs=`ls ./*.pbs | grep $1 | wc -l`
for file in $files
do
  echo "submit $file ..."
  qsub $file
done
echo "submit $total_jobs jobs"

