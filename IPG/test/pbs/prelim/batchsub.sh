files=`ls ./*.pbs`
for file in $files
do
  echo "submit $file ..."
  qsub $file
done