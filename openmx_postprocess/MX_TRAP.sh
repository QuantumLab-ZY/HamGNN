#! /bin/csh



echo -e "=========================Outputfile List==========================\n"
ls -c *out

echo -e "==================================================================\n"

echo -n  'Please enter the outputfile name ; Outputfile name = '
read outputfile            #input

if [ `echo $outputfile|grep .out` ] ; then
  

File_Name=$(echo -e "$outputfile" |  cut -d "." -f 1)   #file name gain

Path=$(pwd)                                             #
echo -n "$File_Name"                                    #Directly of file name is made
mkdir $File_Name                                        #copy OpenMXoutputfile to directly
cp $outputfile $File_Name                               #go to directly 
cd $Path/$File_Name

rm  $File_Name.energy
spinpolarization_swich=$(grep scf.SpinPolarization $outputfile | sed -e 's/^[ ]*//g' | sed 's/[\t ]\+/\t/g' | cut -f 2 | sed -e 's/\(^.*\)/\U\1/' | cat ) #
                                                                                                                        #
if [ $spinpolarization_swich = NC ] ; then    ##if 1                                                                          

 echo -e "scf.SpinPolarization is Non-colliner"

else ###else 1

 echo -e "scf.SpinPolarization is colliner"

fi  ###fi 1



if [ $spinpolarization_swich = NC ] ; then  ###Non-colliner or not### ##if 2

####Start:::Non-colliner######

echo -e  "Please wait. Generating .energy file ..."

kloopmax=$(grep kloop $outputfile | tail -n 1 | cut -c 10-20)

if [ `expr $kloopmax % 2` == 0 ] ; then   ##if 3

####Noncollier: Number of k-grid is odd######

kloopmaxpositive=`expr $kloopmax / 2`

####Noncollier: Number of k-grid is odd######
else ##3new

####Noncollier: Number of k-grid is devide######

kloopmaxpositive=`expr \( $kloopmax - 1 \) / 2`

####Noncollier: Number of k-grid is devide######

fi #3new

total_kpoints=`expr $kloopmaxpositive + 1`


rm  $File_Name.energyso
touch $File_Name.energyso
touch $File_Name.energysodammy

touch $File_Name.energysodammy_1
touch $File_Name.energysodammy_2
touch $File_Name.energysodammy_3
touch $File_Name.energysodammy_4
touch $File_Name.energysodammy_5
touch $File_Name.energysodammy_6
touch $File_Name.energysodammy_7
touch $File_Name.energysodammy_8
touch $File_Name.energyso_1
touch $File_Name.energyso_2
touch $File_Name.energyso_3
touch $File_Name.energyso_4
touch $File_Name.energyso_5
touch $File_Name.energyso_6
touch $File_Name.energyso_7
touch $File_Name.energyso_8

cp $outputfile $File_Name.out1
cp $outputfile $File_Name.out2
cp $outputfile $File_Name.out3
cp $outputfile $File_Name.out4
cp $outputfile $File_Name.out5
cp $outputfile $File_Name.out6
cp $outputfile $File_Name.out7
cp $outputfile $File_Name.out8



Amari=`expr $kloopmaxpositive % 8`
Amari_2=`expr $kloopmaxpositive - $Amari`

Parallel_1=`expr \( $Amari_2 / 8 \) \* 1`
Parallel_2=`expr \( $Amari_2 / 8 \) \* 2`
Parallel_3=`expr \( $Amari_2 / 8 \) \* 3`
Parallel_4=`expr \( $Amari_2 / 8 \) \* 4`
Parallel_5=`expr \( $Amari_2 / 8 \) \* 5`
Parallel_6=`expr \( $Amari_2 / 8 \) \* 6`
Parallel_7=`expr \( $Amari_2 / 8 \) \* 7`


Parallel_1_=`expr $Parallel_1 + 1`
Parallel_2_=`expr $Parallel_2 + 1`
Parallel_3_=`expr $Parallel_3 + 1`
Parallel_4_=`expr $Parallel_4 + 1`
Parallel_5_=`expr $Parallel_5 + 1`
Parallel_6_=`expr $Parallel_6 + 1`
Parallel_7_=`expr $Parallel_7 + 1`
wait
######################kloop = 1 ~ 1/8kloopmaxpositive#
for i in `seq 0 $Parallel_1` 
do
touch eigen_1 &
touch eigen2_1 &
touch TEST1_1 &
touch TEST2_1 &
touch TEST3_1 &
touch TEST4_1 &
touch test3_1 &
touch test4_1 &
touch test5_1 &
wait

echo "Non-colliner::kloop kx ky kz:$i/$Parallel_1" 


grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_1 &

wait
paste TEST1_1 TEST2_1 TEST3_1 > TEST4_1
wait
j=`expr $i + 1`
ei=$(grep "kloop="$j$ -B 2 $File_Name.out1 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k=`expr $ei + 1`

P=`expr $ei + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i$ -A $P $File_Name.out1| grep "kloop="$j$ -B $k | head -n $ei | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_1 &

wait

sort -n test3_1 > test4_1
cat test4_1 | wc > eigen_1
cat eigen_1 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_1
paste TEST4_1 eigen2_1 > TEST5_1
cat TEST5_1 test4_1 > test5_1

wait

cat test5_1 $File_Name.energysodammy_1 >> $File_Name.energyso_1
wait

rm test3_1 &
rm test4_1 &
rm test5_1 &
rm TEST1_1 &
rm TEST2_1 &
rm TEST3_1 &
rm TEST4_1 &
rm TEST5_1 &
rm eigen_1 &
rm eigen2_1 &
rm $File_Name.energysodammy_1  
touch $File_Name.energysodammy_1
wait

done &
######################kloop = 1 ~ 1/8kloopmaxpositive#######END#

######################kloop = 1/8kloopmaxpositive+1 ~ 2/8kloopmaxpositive#
for i2 in `seq $Parallel_1_ $Parallel_2` 
do
touch eigen_2 &
touch eigen2_2 &
wait


echo "Non-colliner::kloop kx ky kz:$i2/$Parallel_2" 


grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_2 &

wait
paste TEST1_2 TEST2_2 TEST3_2 > TEST4_2

j2=`expr $i2 + 1`
ei2=$(grep "kloop="$j2$ -B 2 $File_Name.out2 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k2=`expr $ei2 + 1`

P2=`expr $ei2 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i2$ -A $P2 $File_Name.out2| grep "kloop="$j2$ -B $k2 | head -n $ei2 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_2 &

wait

sort -n test3_2 > test4_2
cat test4_2 | wc > eigen_2
cat eigen_2 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_2
paste TEST4_2 eigen2_2 > TEST5_2
cat TEST5_2 test4_2 > test5_2

wait

cat test5_2 $File_Name.energysodammy_2 >> $File_Name.energyso_2

rm test3_2 &
rm test4_2 &
rm test5_2 &
rm TEST1_2 &
rm TEST2_2 &
rm TEST3_2 &
rm TEST4_2 &
rm eigen_2 &
rm eigen2_2 &
rm TEST5_2 &
rm $File_Name.energysodammy_2  
touch $File_Name.energysodammy_2
wait

done &

######################kloop = 1/8kloopmaxpositive+1 ~ 2/8kloopmaxpositive#######END#


######################kloop = 2/8kloopmaxpositive+1 ~ 3/8kloopmaxpositive#
for i3 in `seq $Parallel_2_ $Parallel_3` 
do
touch eigen_3 &
touch eigen2_3 &
wait


echo "Non-colliner::kloop kx ky kz:$i3/$Parallel_3" 


grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_3 &

wait
paste TEST1_3 TEST2_3 TEST3_3 > TEST4_3

j3=`expr $i3 + 1`
ei3=$(grep "kloop="$j3$ -B 2 $File_Name.out3 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k3=`expr $ei3 + 1`

P3=`expr $ei3 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i3$ -A $P3 $File_Name.out3| grep "kloop="$j3$ -B $k3 | head -n $ei3 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_3 &

wait

sort -n test3_3 > test4_3
cat test4_3 | wc > eigen_3
cat eigen_3 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_3
paste TEST4_3 eigen2_3 > TEST5_3
cat TEST5_3 test4_3 > test5_3

wait

cat test5_3 $File_Name.energysodammy_3 >> $File_Name.energyso_3

rm test3_3 &
rm test4_3 &
rm test5_3 &
rm TEST1_3 &
rm TEST2_3 &
rm TEST3_3 &
rm TEST4_3 &
rm TEST5_3 &
rm eigen_3 &
rm eigen2_3 &
rm $File_Name.energysodammy_3  
touch $File_Name.energysodammy_3
wait

done &

######################kloop = 2/8kloopmaxpositive+1 ~ 3/8kloopmaxpositive#######END#



######################kloop = 3/8kloopmaxpositive+1 ~ 4/8kloopmaxpositive#
for i4 in `seq $Parallel_3_ $Parallel_4` 
do
touch eigen_4 &
touch eigen2_4 &
wait


echo "Non-colliner::kloop kx ky kz:$i4/$Parallel_4" 


grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_4 &

wait
paste TEST1_4 TEST2_4 TEST3_4 > TEST4_4

j4=`expr $i4 + 1`
ei4=$(grep "kloop="$j4$ -B 2 $File_Name.out4 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k4=`expr $ei4 + 1`

P4=`expr $ei4 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i4$ -A $P4 $File_Name.out4| grep "kloop="$j4$ -B $k4 | head -n $ei4 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_4 &

wait

sort -n test3_4 > test4_4
cat test4_4 | wc > eigen_4
cat eigen_4 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_4
paste TEST4_4 eigen2_4 > TEST5_4
cat TEST5_4 test4_4 > test5_4

wait

cat test5_4 $File_Name.energysodammy_4 >> $File_Name.energyso_4

rm test3_4 &
rm test4_4 &
rm test5_4 &
rm TEST1_4 &
rm TEST2_4 &
rm TEST3_4 &
rm TEST4_4 &
rm TEST5_4 &
rm eigen_4 &
rm eigen2_4 &
rm $File_Name.energysodammy_4  
touch $File_Name.energysodammy_4
wait

done &

######################kloop = 3/8kloopmaxpositive+1 ~ 4/8kloopmaxpositive#######END#



######################kloop = 4/8kloopmaxpositive+1 ~ 5/8kloopmaxpositive#
for i5 in `seq $Parallel_4_ $Parallel_5` 
do
touch eigen_5 &
touch eigen2_5 &
wait


echo "Non-colliner::kloop kx ky kz:$i5/$Parallel_5" 


grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_5 &

wait
paste TEST1_5 TEST2_5 TEST3_5 > TEST4_5

j5=`expr $i5 + 1`
ei5=$(grep "kloop="$j5$ -B 2 $File_Name.out5 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k5=`expr $ei5 + 1`

P5=`expr $ei5 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i5$ -A $P5 $File_Name.out5| grep "kloop="$j5$ -B $k5 | head -n $ei5 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_5 &

wait

sort -n test3_5 > test4_5
cat test4_5 | wc > eigen_5
cat eigen_5 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_5
paste TEST4_5 eigen2_5 > TEST5_5
cat TEST5_5 test4_5 > test5_5

wait

cat test5_5 $File_Name.energysodammy_5 >> $File_Name.energyso_5

rm test3_5 &
rm test4_5 &
rm test5_5 &
rm TEST1_5 &
rm TEST2_5 &
rm TEST3_5 &
rm TEST4_5 &
rm TEST5_5 &
rm eigen_5 &
rm eigen2_5 &
rm $File_Name.energysodammy_5  
touch $File_Name.energysodammy_5
wait

done &

######################kloop = 4/8kloopmaxpositive+1 ~ 5/8kloopmaxpositive#######END#



######################kloop = 5/8kloopmaxpositive+1 ~ 6/8kloopmaxpositive#
for i6 in `seq $Parallel_5_ $Parallel_6` 
do
touch eigen_6 &
touch eigen2_6 &
wait


echo "Non-colliner::kloop kx ky kz:$i6/$Parallel_6" 


grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_6 &

wait
paste TEST1_6 TEST2_6 TEST3_6 > TEST4_6

j6=`expr $i6 + 1`
ei6=$(grep "kloop="$j6$ -B 2 $File_Name.out6 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k6=`expr $ei6 + 1`

P6=`expr $ei6 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i6$ -A $P6 $File_Name.out6| grep "kloop="$j6$ -B $k6 | head -n $ei6 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_6 &

wait

sort -n test3_6 > test4_6
cat test4_6 | wc > eigen_6
cat eigen_6 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_6
paste TEST4_6 eigen2_6 > TEST5_6
cat TEST5_6 test4_6 > test5_6

wait

cat test5_6 $File_Name.energysodammy_6 >> $File_Name.energyso_6

rm test3_6 &
rm test4_6 &
rm test5_6 &
rm TEST1_6 &
rm TEST2_6 &
rm TEST3_6 &
rm TEST4_6 &
rm TEST5_6 &
rm eigen_6 &
rm eigen2_6 &
rm $File_Name.energysodammy_6  
touch $File_Name.energysodammy_6
wait

done &

######################kloop = 5/8kloopmaxpositive+1 ~ 6/8kloopmaxpositive#######END#



######################kloop = 6/8kloopmaxpositive+1 ~ 7/8kloopmaxpositive#
for i7 in `seq $Parallel_6_ $Parallel_7` 
do
touch eigen_7 &
touch eigen2_7 &
wait


echo "Non-colliner::kloop kx ky kz:$i7/$Parallel_7" 


grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_7 &

wait
paste TEST1_7 TEST2_7 TEST3_7 > TEST4_7

j7=`expr $i7 + 1`
ei7=$(grep "kloop="$j7$ -B 2 $File_Name.out7 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k7=`expr $ei7 + 1`

P7=`expr $ei7 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i7$ -A $P7 $File_Name.out7| grep "kloop="$j7$ -B $k7 | head -n $ei7 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_7 &

wait

sort -n test3_7 > test4_7
cat test4_7 | wc > eigen_7
cat eigen_7 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_7
paste TEST4_7 eigen2_7 > TEST5_7
cat TEST5_7 test4_7 > test5_7

wait

cat test5_7 $File_Name.energysodammy_7 >> $File_Name.energyso_7

rm test3_7 &
rm test4_7 &
rm test5_7 &
rm TEST1_7 &
rm TEST2_7 &
rm TEST3_7 &
rm TEST4_7 &
rm TEST5_7 &
rm eigen_7 &
rm eigen2_7 &
rm $File_Name.energysodammy_7  
touch $File_Name.energysodammy_7
wait

done &

######################kloop = 6/8kloopmaxpositive+1 ~ 7/8kloopmaxpositive#######END#



######################kloop = 7/8kloopmaxpositive+1 ~ kloopmaxpositive#
for i8 in `seq $Parallel_7_ $kloopmaxpositive` 
do
touch eigen_8 &
touch eigen2_8 &
wait


echo "Non-colliner::kloop kx ky kz:$i8/$kloopmaxpositive" 


grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_8 &

wait
paste TEST1_8 TEST2_8 TEST3_8 > TEST4_8

j8=`expr $i8 + 1`
ei8=$(grep "kloop="$j8$ -B 2 $File_Name.out8 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k8=`expr $ei8 + 1`

P8=`expr $ei8 + 4`
wait
#kloop = i(=kloopmax/2 because kpoint is symmertry at the origin), #
grep  "kloop="$i8$ -A $P8 $File_Name.out8| grep "kloop="$j8$ -B $k8 | head -n $ei8 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_8 &

wait

sort -n test3_8 > test4_8
cat test4_8 | wc > eigen_8
cat eigen_8 | sed 's/[\t ]\+/\t/g' | cut -f2  > eigen2_8
paste TEST4_8 eigen2_8 > TEST5_8
cat TEST5_8 test4_8 > test5_8

wait

cat test5_8 $File_Name.energysodammy_8 >> $File_Name.energyso_8

rm test3_8 &
rm test4_8 &
rm test5_8 &
rm TEST1_8 &
rm TEST2_8 &
rm TEST3_8 &
rm TEST4_8 &
rm TEST5_8 &
rm eigen_8 &
rm eigen2_8 &
rm $File_Name.energysodammy_8  
touch $File_Name.energysodammy_8
wait

done &
wait
######################kloop = 7/8kloopmaxpositive+1 ~ kloopmaxpositive#######END#


cat  $File_Name.energyso_1 $File_Name.energyso_2 $File_Name.energyso_3 $File_Name.energyso_4 $File_Name.energyso_5 $File_Name.energyso_6 $File_Name.energyso_7 $File_Name.energyso_8 >  $File_Name.energyso


wait





sed -i "1s/^/$total_kpoints\n/" $File_Name.energyso
sed -i '1s/^/Energy file of BoltzTrap for OpenMX\n/' $File_Name.energyso

echo -e ".energy file for BoltzTraP has been generated.\n"

rm $File_Name.energysodammy_1
rm $File_Name.energysodammy_2
rm $File_Name.energysodammy_3
rm $File_Name.energysodammy_4
rm $File_Name.energysodammy_5
rm $File_Name.energysodammy_6
rm $File_Name.energysodammy_7
rm $File_Name.energysodammy_8

rm $File_Name.energyso_1
rm $File_Name.energyso_2
rm $File_Name.energyso_3
rm $File_Name.energyso_4
rm $File_Name.energyso_5
rm $File_Name.energyso_6
rm $File_Name.energyso_7
rm $File_Name.energyso_8

rm  $File_Name.out1
rm  $File_Name.out2
rm  $File_Name.out3
rm  $File_Name.out4
rm  $File_Name.out5
rm  $File_Name.out6
rm  $File_Name.out7
rm  $File_Name.out8

touch $File_Name.struct

LatticeUnit=$(grep Atoms.UnitVectors.Unit $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 2)

if grep 'a1 =' $outputfile >/dev/null; then

 grep "a1 = " -A 2 $outputfile | sed -e 's/^[ ]*//g' | sed 's/[\t ]\+/\t/g' | head -n 3 | awk '{ OFMT = "%.14f"}{print $3*1.889725989, $4*1.889725989, $5*1.889725989}' > $File_Name.struct

else

if [ $LatticeUnit = Ang -o $LatticeUnit = ang ] ; then #if 6

 awk '/<Atoms.UnitVectors/,/Atoms.UnitVectors>/' $outputfile | grep '\S' | tail -n 4 | head -n 3 | awk '{ OFMT = "%.14f"}{print $1*1.889725989, $2*1.889725989, $3*1.889725989}'  > $File_Name.struct

else #else 6
 awk '/<Atoms.UnitVectors/,/Atoms.UnitVectors>/' $outputfile | grep '\S' | tail -n 4 | head -n 3 | awk '{ OFMT = "%.14f"}{print $1, $2, $3}'  > $File_Name.struct

fi #fi 6


fi

echo -e "1" >> $File_Name.struct
echo -e "1 0 0 0 1 0 0 0 1" >> $File_Name.struct

sed -i '1s/^/Structure file of BoltzTrap for OpenMX\n/' $File_Name.struct

echo -e ".struct file for BoltzTraP has been generated.\n"


touch $File_Name.intrans
touch $File_Name.intrans_
Chemicalpotential=$(grep Chemical $outputfile | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 6| awk '{ OFMT = "%.14f"}{print $1*2}') 
####20170829_added####
grep scf.ElectronicTemperature $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 1 | grep "#" > damm
commentoutswitch=$(echo $?)
rm damm
if [ $commentoutswitch = 0 ] ; then

ElectronicTemperature=$(echo 300)

else

grep scf.ElectronicTemperature $File_Name.out > damm
tempswich=$(echo $?)
rm damm
if [ $tempswich = 0 ] ; then 

ElectronicTemperature=$(grep scf.ElectronicTemperature $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 2)

else 
ElectronicTemperature=$(echo 300)

fi

fi
####20170829_added####

Electron_number=$(grep "Number of States" $outputfile | sed 's/[\t ]\+/\t/g' | cut -f 6 | awk '{s=($0<0)?-1:1;print int($0*s*1000+0.5)/1000/s;}')

echo -e "GENE                      # Format of DOS\n" > $File_Name.intrans_
echo -e "0 0 0 0.0                 # iskip (not presently used) idebug setgap shiftgap\n" >> $File_Name.intrans_
echo -e "$Chemicalpotential 0.0005 0.4 $Electron_number   # Fermilevel (Ry), energygrid, energy span around Fermilevel, number of electrons\n" >> $File_Name.intrans_
echo -e "CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n" >> $File_Name.intrans_
echo -e "10                         # lpfac, number of latt-points per k-point\n" >> $File_Name.intrans_
echo -e "BOLTZ                     # run mode (only BOLTZ is supported)\n" >> $File_Name.intrans_
echo -e ".30                       # (efcut) energy range of chemical potential\n" >> $File_Name.intrans_
echo -e "$ElectronicTemperature $ElectronicTemperature                 # Tmax, temperature grid\n" >> $File_Name.intrans_
echo -e "-1.                       # energyrange of bands given individual DOS output sig_xxx and dos_xxx (xxx is band number)\n" >> $File_Name.intrans_
echo -e "HISTO\n" >> $File_Name.intrans_
####20170829_added####

echo -e "0 0 0 0 0\n" >> $File_Name.intrans_
echo -e "1\n" >> $File_Name.intrans_
echo -e "0\n" >> $File_Name.intrans_

####20170829_added####

 grep -v '^\s*$' $File_Name.intrans_ > $File_Name.intrans
rm $File_Name.intrans_
echo -e ".intrans file for BoltzTraP has been generated\n"
echo -e "Conversion has been finished.\n"
echo -e "Directory is $File_Name\n" 
#####END:::Non-colliner ######
rm $File_Name.energysodammy


else ##else 2



####Start::Colliner #####

if [ $spinpolarization_swich = on -o $spinpolarization_swich = ON -o $spinpolarization_swich = On ] ; then   #if 4
##Start::  spinpolarization  on###
###Up spin####
 echo -e  "Please wait. Generating .energyup file ..."

kloopmax=$(grep kloop $outputfile | tail -n 1 | cut -c 10-20)
kloooopmax=`expr $kloopmax - 1`
kloooooopmax=`expr $kloopmax + 1`
rm  $File_Name.energyup &
touch $File_Name.energyup &
touch $File_Name.energydammy_1 &
touch $File_Name.energydammy_2 &
touch $File_Name.energydammy_3 &
touch $File_Name.energydammy_4 &
touch $File_Name.energydammy_5 &
touch $File_Name.energydammy_6 &
touch $File_Name.energydammy_7 &
touch $File_Name.energydammy_8 &
touch $File_Name.energy_1 &
touch $File_Name.energy_2 &
touch $File_Name.energy_3 &
touch $File_Name.energy_4 &
touch $File_Name.energy_5 &
touch $File_Name.energy_6 &
touch $File_Name.energy_7 &
touch $File_Name.energy_8 &

cp $outputfile $File_Name.out1 &
cp $outputfile $File_Name.out2 &
cp $outputfile $File_Name.out3 &
cp $outputfile $File_Name.out4 &
cp $outputfile $File_Name.out5 &
cp $outputfile $File_Name.out6 &
cp $outputfile $File_Name.out7 &
cp $outputfile $File_Name.out8 &



Amari=`expr $kloopmax % 8`
Amari_2=`expr $kloopmax - $Amari`

Parallel_1=`expr \( $Amari_2 / 8 \) \* 1`
Parallel_2=`expr \( $Amari_2 / 8 \) \* 2`
Parallel_3=`expr \( $Amari_2 / 8 \) \* 3`
Parallel_4=`expr \( $Amari_2 / 8 \) \* 4`
Parallel_5=`expr \( $Amari_2 / 8 \) \* 5`
Parallel_6=`expr \( $Amari_2 / 8 \) \* 6`
Parallel_7=`expr \( $Amari_2 / 8 \) \* 7`
Parallel_8=`expr \( $Amari_2 / 8 \) \* 8`



Parallel_1_=`expr $Parallel_1 + 1`
Parallel_2_=`expr $Parallel_2 + 1`
Parallel_3_=`expr $Parallel_3 + 1`
Parallel_4_=`expr $Parallel_4 + 1`
Parallel_5_=`expr $Parallel_5 + 1`
Parallel_6_=`expr $Parallel_6 + 1`
Parallel_7_=`expr $Parallel_7 + 1`
wait

########################################################################################################1

###kloop = i ~ 1/8(max-Amari)####
for i in `seq 0 $Parallel_1` 
do

touch eigen_1 &
touch eigen2_1 &
touch TEST1_1 &
touch TEST2_1 &
touch TEST3_1 &
touch TEST4_1 &
touch test3_1 &
touch test4_1 &
touch test5_1 &
wait 
 
echo "Colliner_upspin::kloop kx ky kz:$i/$Parallel_1" 


grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_1 &

wait
paste TEST1_1 TEST2_1 TEST3_1 > TEST4_1

j=`expr $i + 1`
ei=$(grep "kloop="$j$ -B 2 $File_Name.out1 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k=`expr $ei + 1`
P=`expr $ei + 4`
grep  "kloop="$i$ -A $P $File_Name.out1| grep "kloop="$j$ -B $k | head -n $ei | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_1  &
wait

sort -n test3_1 > test4_1
cat test4_1 | wc > eigen_1
cat eigen_1 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_1
paste TEST4_1 eigen2_1 > TEST5_1
cat TEST5_1 test4_1 > test5_1



cat test5_1 $File_Name.energydammy_1 >> $File_Name.energy_1
wait 
rm test3_1 &
rm test4_1 &
rm test5_1 &
rm TEST1_1 &
rm TEST2_1 &
rm TEST3_1 &
rm TEST4_1 &
rm TEST5_1 &
rm eigen2_1 &
rm eigen_1 &

rm $File_Name.energydammy_1  
touch $File_Name.energydammy_1
wait
done &

#########################################################################################1

########################################################################################################2

###kloop = i ~ 1/8(max-Amari)####
for i2 in `seq $Parallel_1_ $Parallel_2` 
do 

touch eigen_2 &
touch eigen2_2 &
touch TEST1_2 &
touch TEST2_2 &
touch TEST3_2 &
touch TEST4_2 &
touch test3_2 &
touch test4_2 &
touch test5_2 &
wait

echo "Colliner_upspin::kloop kx ky kz:$i2/$Parallel_2" 


grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_2 &

wait
paste TEST1_2 TEST2_2 TEST3_2 > TEST4_2

j2=`expr $i2 + 1`
ei2=$(grep "kloop="$j2$ -B 2 $File_Name.out2 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k2=`expr $ei2 + 1`
P2=`expr $ei2 + 4`
grep  "kloop="$i2$ -A $P2 $File_Name.out2 | grep "kloop="$j2$ -B $k2 | head -n $ei2 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_2  &
wait

sort -n test3_2 > test4_2
more test4_2 | wc > eigen_2
more eigen_2 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_2
paste TEST4_2 eigen2_2 > TEST5_2
cat TEST5_2 test4_2 > test5_2



cat test5_2 $File_Name.energydammy_2 >> $File_Name.energy_2
wait 
rm test3_2 &
rm test4_2 &
rm test5_2 &
rm TEST1_2 &
rm TEST2_2 &
rm TEST3_2 &
rm TEST4_2 &
rm TEST5_2 &
rm eigen2_2 &
rm eigen_2 &
rm $File_Name.energydammy_2  
touch $File_Name.energydammy_2
wait
done &

#########################################################################################2

########################################################################################################3

###kloop = i ~ 1/8(max-Amari)####
for i3 in `seq $Parallel_2_ $Parallel_3` 
do

touch eigen_3 &
touch eigen2_3 &
touch TEST1_3 &
touch TEST2_3 &
touch TEST3_3 &
touch TEST4_3 &
touch test3_3 &
touch test4_3 &
touch test5_3 &
wait

echo "Colliner_upspin::kloop kx ky kz:$i3/$Parallel_3" 


grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_3 &

wait
paste TEST1_3 TEST2_3 TEST3_3 > TEST4_3

j3=`expr $i3 + 1`
ei3=$(grep "kloop="$j3$ -B 2 $File_Name.out3 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k3=`expr $ei3 + 1`
P3=`expr $ei3 + 4`
grep  "kloop="$i3$ -A $P3 $File_Name.out3| grep "kloop="$j3$ -B $k3 | head -n $ei3 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_3  &
wait

sort -n test3_3 > test4_3
more test4_3 | wc > eigen_3
more eigen_3 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_3
paste TEST4_3 eigen2_3 > TEST5_3
cat TEST5_3 test4_3 > test5_3



cat test5_3 $File_Name.energydammy_3 >> $File_Name.energy_3
wait 
rm test3_3 &
rm test4_3 &
rm test5_3 &
rm TEST1_3 &
rm TEST2_3 &
rm TEST3_3 &
rm TEST4_3 &
rm TEST5_3 &
rm eigen2_3 &
rm eigen_3 &
rm $File_Name.energydammy_3  
touch $File_Name.energydammy_3
wait
done &

#########################################################################################3

########################################################################################################4

###kloop = i ~ 1/8(max-Amari)####
for i4 in `seq $Parallel_3_ $Parallel_4` 
do
touch eigen_4 &
touch eigen2_4 &
touch TEST1_4 &
touch TEST2_4 &
touch TEST3_4 &
touch TEST4_4 &
touch test3_4 &
touch test4_4 &
touch test5_4 &
wait

echo "Colliner_upspin::kloop kx ky kz:$i4/$Parallel_4" 


grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_4 &

wait
paste TEST1_4 TEST2_4 TEST3_4 > TEST4_4

j4=`expr $i4 + 1`
ei4=$(grep "kloop="$j4$ -B 2 $File_Name.out4 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k4=`expr $ei4 + 1`
P4=`expr $ei4 + 4`
grep  "kloop="$i4$ -A $P4 $File_Name.out4| grep "kloop="$j4$ -B $k4 | head -n $ei4 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_4  &
wait

sort -n test3_4 > test4_4
more test4_4 | wc > eigen_4
more eigen_4 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_4
paste TEST4_4 eigen2_4 > TEST5_4
cat TEST5_4 test4_4 > test5_4



cat test5_4 $File_Name.energydammy_4 >> $File_Name.energy_4
wait 
rm test3_4 &
rm test4_4 &
rm test5_4 &
rm TEST1_4 &
rm TEST2_4 &
rm TEST3_4 &
rm TEST4_4 &
rm TEST5_4 &
rm eigen2_4 &
rm eigen_4 &
rm $File_Name.energydammy_4  
touch $File_Name.energydammy_4
wait
done &

#########################################################################################4

########################################################################################################5

###kloop = i ~ 1/8(max-Amari)####
for i5 in `seq $Parallel_4_ $Parallel_5` 
do
touch eigen_5 &
touch eigen2_5 &
touch TEST1_5 &
touch TEST2_5 &
touch TEST3_5 &
touch TEST4_5 &
touch test3_5 &
touch test4_5 &
touch test5_5 &
wait

echo "Colliner_upspin::kloop kx ky kz:$i5/$Parallel_5" 


grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_5 &

wait
paste TEST1_5 TEST2_5 TEST3_5 > TEST4_5

j5=`expr $i5 + 1`
ei5=$(grep "kloop="$j5$ -B 2 $File_Name.out5 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k5=`expr $ei5 + 1`
P5=`expr $ei5 + 4`
grep  "kloop="$i5$ -A $P5 $File_Name.out5| grep "kloop="$j5$ -B $k5 | head -n $ei5 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_5  &
wait

sort -n test3_5 > test4_5
more test4_5 | wc > eigen_5
more eigen_5 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_5
paste TEST4_5 eigen2_5 > TEST5_5
cat TEST5_5 test4_5 > test5_5



cat test5_5 $File_Name.energydammy_5 >> $File_Name.energy_5
wait 
rm test3_5 &
rm test4_5 &
rm test5_5 &
rm TEST1_5 &
rm TEST2_5 &
rm TEST3_5 &
rm TEST4_5 &
rm TEST5_5 &
rm eigen2_5 &
rm eigen_5 &
rm $File_Name.energydammy_5  
touch $File_Name.energydammy_5
wait
done &

#########################################################################################5

########################################################################################################6

###kloop = i ~ 1/8(max-Amari)####
for i6 in `seq $Parallel_5_ $Parallel_6` 
do
touch eigen_6 &
touch eigen2_6 &
touch TEST1_6 &
touch TEST2_6 &
touch TEST3_6 &
touch TEST4_6 &
touch test3_6 &
touch test4_6 &
touch test5_6 &

wait

echo "Colliner_upspin::kloop kx ky kz:$i6/$Parallel_6" 


grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_6 &

wait
paste TEST1_6 TEST2_6 TEST3_6 > TEST4_6

j6=`expr $i6 + 1`
ei6=$(grep "kloop="$j6$ -B 2 $File_Name.out6 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k6=`expr $ei6 + 1`
P6=`expr $ei6 + 4`
grep  "kloop="$i6$ -A $P6 $File_Name.out6| grep "kloop="$j6$ -B $k6 | head -n $ei6 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_6  &
wait

sort -n test3_6 > test4_6
more test4_6 | wc > eigen_6
more eigen_6 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_6
paste TEST4_6 eigen2_6 > TEST5_6
cat TEST5_6 test4_6 > test5_6



cat test5_6 $File_Name.energydammy_6 >> $File_Name.energy_6
wait 
rm test3_6 &
rm test4_6 &
rm test5_6 &
rm TEST1_6 &
rm TEST2_6 &
rm TEST3_6 &
rm TEST4_6 &
rm TEST5_6 &
rm eigen2_6 &
rm eigen_6 &
rm $File_Name.energydammy_6  
touch $File_Name.energydammy_6
wait
done &

#########################################################################################6

########################################################################################################7

###kloop = i ~ 1/8(max-Amari)####
for i7 in `seq $Parallel_6_ $Parallel_7` 
do
touch eigen_7 &
touch eigen2_7 &
touch TEST1_7 &
touch TEST2_7 &
touch TEST3_7 &
touch TEST4_7 &
touch test3_7 &
touch test4_7 &
touch test5_7 &

wait

echo "Colliner_upspin::kloop kx ky kz:$i7/$Parallel_7" 


grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_7 &

wait
paste TEST1_7 TEST2_7 TEST3_7 > TEST4_7

j7=`expr $i7 + 1`
ei7=$(grep "kloop="$j7$ -B 2 $File_Name.out7 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k7=`expr $ei7 + 1`
P7=`expr $ei7 + 4`
grep  "kloop="$i7$ -A $P7 $File_Name.out7| grep "kloop="$j7$ -B $k7 | head -n $ei7 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_7  &
wait

sort -n test3_7 > test4_7
more test4_7 | wc > eigen_7
more eigen_7 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_7
paste TEST4_7 eigen2_7 > TEST5_7
cat TEST5_7 test4_7 > test5_7



cat test5_7 $File_Name.energydammy_7 >> $File_Name.energy_7
wait 
rm test3_7 &
rm test4_7 &
rm test5_7 &
rm TEST1_7 &
rm TEST2_7 &
rm TEST3_7 &
rm TEST4_7 &
rm eigen2_7 &
rm eigen_7 &
rm TEST5_7 &
rm $File_Name.energydammy_7  
touch $File_Name.energydammy_7
wait
done &

#########################################################################################7

########################################################################################################8

###kloop = i ~ 1/8(max-Amari)####
for i8 in `seq $Parallel_7_ $kloooopmax` 
do 
touch eigen_8 &
touch eigen2_8 &
touch TEST1_8 &
touch TEST2_8 &
touch TEST3_8 &
touch TEST4_8 &
touch test3_8 &
touch test4_8 &
touch test5_8 &

wait 
echo "Colliner_upspin::kloop kx ky kz:$i8/$kloooopmax" 


grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_8 &

wait
paste TEST1_8 TEST2_8 TEST3_8 > TEST4_8

j8=`expr $i8 + 1`
ei8=$(grep "kloop="$j8$ -B 2 $File_Name.out8 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k8=`expr $ei8 + 1`
P8=`expr $ei8 + 4`
grep  "kloop="$i8$ -A $P8 $File_Name.out8| grep "kloop="$j8$ -B $k8 | head -n $ei8 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_8  &
wait

sort -n test3_8 > test4_8
more test4_8 | wc > eigen_8
more eigen_8 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_8
paste TEST4_8 eigen2_8 > TEST5_8
cat TEST5_8 test4_8 > test5_8



cat test5_8 $File_Name.energydammy_8 >> $File_Name.energy_8
wait 
rm test3_8 &
rm test4_8 &
rm test5_8 &
rm TEST1_8 &
rm TEST2_8 &
rm TEST3_8 &
rm TEST4_8 &
rm TEST5_8 &
rm eigen2_8 &
rm eigen_8 &
rm $File_Name.energydammy_8  
touch $File_Name.energydammy_8
wait
done &
wait
#########################################################################################8

cat $File_Name.energy_1 $File_Name.energy_2 $File_Name.energy_3 $File_Name.energy_4 $File_Name.energy_5 $File_Name.energy_6 $File_Name.energy_7 $File_Name.energy_8 > $File_Name.energyup

wait 
#####kloop = max####
touch $File_Name.energydammy
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1 &
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2 &
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3 &
wait
paste TEST1 TEST2 TEST3 > TEST4
ei=$(grep "kloop="$kloooopmax$ -B 2 $outputfile | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)
U=`expr $ei + 2`
grep  "kloop="$kloopmax$ -A $U $outputfile| tail -n $ei | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}' > test3  &
wait

sort -n test3 > test4
more test4 | wc > eigen
more eigen | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2
paste TEST4 eigen2 > TEST5
cat TEST5 test4 > test5



cat test5 $File_Name.energydammy >> $File_Name.energyup

rm test3 &
rm test4 &
rm test5 &

rm TEST1 &
rm TEST2 &
rm TEST3 &
rm TEST4 &
rm TEST5 &
rm $File_Name.energydammy  
touch $File_Name.energydammy



sed -i "1s/^/$kloooooopmax\n/" $File_Name.energyup
sed -i '1s/^/Energy file of BoltzTrap for OpenMX\n/' $File_Name.energyup

echo -e ".energyup file for BoltzTraP has been generated.\n"

rm  $File_Name.out1
rm  $File_Name.out2
rm  $File_Name.out3
rm  $File_Name.out4
rm  $File_Name.out5
rm  $File_Name.out6
rm  $File_Name.out7
rm  $File_Name.out8

rm  $File_Name.energy_1
rm  $File_Name.energy_2
rm  $File_Name.energy_3
rm  $File_Name.energy_4
rm  $File_Name.energy_5
rm  $File_Name.energy_6
rm  $File_Name.energy_7
rm  $File_Name.energy_8

rm $File_Name.energydammy_1
rm $File_Name.energydammy_2
rm $File_Name.energydammy_3
rm $File_Name.energydammy_4
rm $File_Name.energydammy_5
rm $File_Name.energydammy_6
rm $File_Name.energydammy_7
rm $File_Name.energydammy_8
rm eigen &

rm eigen2 &


###Down spin####

 echo -e  "Please wait. Generating .energydn file ..."

kloopmax=$(grep kloop $outputfile | tail -n 1 | cut -c 10-20)
kloooopmax=`expr $kloopmax - 1`
kloooooopmax=`expr $kloopmax + 1`
rm  $File_Name.energydn &
touch $File_Name.energydn &
touch $File_Name.energydammy_1 &
touch $File_Name.energydammy_2 &
touch $File_Name.energydammy_3 &
touch $File_Name.energydammy_4 &
touch $File_Name.energydammy_5 &
touch $File_Name.energydammy_6 &
touch $File_Name.energydammy_7 &
touch $File_Name.energydammy_8 &
touch $File_Name.energy_1 &
touch $File_Name.energy_2 &
touch $File_Name.energy_3 &
touch $File_Name.energy_4 &
touch $File_Name.energy_5 &
touch $File_Name.energy_6 &
touch $File_Name.energy_7 &
touch $File_Name.energy_8 &

cp $outputfile $File_Name.out1 &
cp $outputfile $File_Name.out2 &
cp $outputfile $File_Name.out3 &
cp $outputfile $File_Name.out4 &
cp $outputfile $File_Name.out5 &
cp $outputfile $File_Name.out6 &
cp $outputfile $File_Name.out7 &
cp $outputfile $File_Name.out8 &



Amari=`expr $kloopmax % 8`
Amari_2=`expr $kloopmax - $Amari`

Parallel_1=`expr \( $Amari_2 / 8 \) \* 1`
Parallel_2=`expr \( $Amari_2 / 8 \) \* 2`
Parallel_3=`expr \( $Amari_2 / 8 \) \* 3`
Parallel_4=`expr \( $Amari_2 / 8 \) \* 4`
Parallel_5=`expr \( $Amari_2 / 8 \) \* 5`
Parallel_6=`expr \( $Amari_2 / 8 \) \* 6`
Parallel_7=`expr \( $Amari_2 / 8 \) \* 7`
Parallel_8=`expr \( $Amari_2 / 8 \) \* 8`



Parallel_1_=`expr $Parallel_1 + 1`
Parallel_2_=`expr $Parallel_2 + 1`
Parallel_3_=`expr $Parallel_3 + 1`
Parallel_4_=`expr $Parallel_4 + 1`
Parallel_5_=`expr $Parallel_5 + 1`
Parallel_6_=`expr $Parallel_6 + 1`
Parallel_7_=`expr $Parallel_7 + 1`

wait
########################################################################################################1

###kloop = i ~ 1/8(max-Amari)####
for i in `seq 0 $Parallel_1` 
do

touch eigen_1 &
touch eigen2_1 &
touch TEST1_1 &
touch TEST2_1 &
touch TEST3_1 &
touch TEST4_1 &
touch test3_1 &
touch test4_1 &
touch test5_1 &
wait 
 
echo "Colliner_donwspin::kloop kx ky kz:$i/$Parallel_1" 


grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_1 &

wait
paste TEST1_1 TEST2_1 TEST3_1 > TEST4_1

j=`expr $i + 1`
ei=$(grep "kloop="$j$ -B 2 $File_Name.out1 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k=`expr $ei + 1`
P=`expr $ei + 4`
grep  "kloop="$i$ -A $P $File_Name.out1| grep "kloop="$j$ -B $k | head -n $ei | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_1  &
wait

sort -n test3_1 > test4_1
cat test4_1 | wc > eigen_1
cat eigen_1 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_1
paste TEST4_1 eigen2_1 > TEST5_1
cat TEST5_1 test4_1 > test5_1



cat test5_1 $File_Name.energydammy_1 >> $File_Name.energy_1
wait 
rm test3_1 &
rm test4_1 &
rm test5_1 &
rm TEST1_1 &
rm TEST2_1 &
rm TEST3_1 &
rm TEST4_1 &
rm TEST5_1 &
rm eigen2_1 &
rm eigen_1 &

rm $File_Name.energydammy_1  
touch $File_Name.energydammy_1
wait
done &

#########################################################################################1

########################################################################################################2

###kloop = i ~ 1/8(max-Amari)####
for i2 in `seq $Parallel_1_ $Parallel_2` 
do 

touch eigen_2 &
touch eigen2_2 &
touch TEST1_2 &
touch TEST2_2 &
touch TEST3_2 &
touch TEST4_2 &
touch test3_2 &
touch test4_2 &
touch test5_2 &
wait

echo "Colliner_donwspin::kloop kx ky kz:$i2/$Parallel_2" 


grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_2 &

wait
paste TEST1_2 TEST2_2 TEST3_2 > TEST4_2

j2=`expr $i2 + 1`
ei2=$(grep "kloop="$j2$ -B 2 $File_Name.out2 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k2=`expr $ei2 + 1`
P2=`expr $ei2 + 4`
grep  "kloop="$i2$ -A $P2 $File_Name.out2 | grep "kloop="$j2$ -B $k2 | head -n $ei2 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_2  &
wait

sort -n test3_2 > test4_2
more test4_2 | wc > eigen_2
more eigen_2 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_2
paste TEST4_2 eigen2_2 > TEST5_2
cat TEST5_2 test4_2 > test5_2



cat test5_2 $File_Name.energydammy_2 >> $File_Name.energy_2
wait 
rm test3_2 &
rm test4_2 &
rm test5_2 &
rm TEST1_2 &
rm TEST2_2 &
rm TEST3_2 &
rm TEST4_2 &
rm TEST5_2 &
rm eigen2_2 &
rm eigen_2 &
rm $File_Name.energydammy_2  
touch $File_Name.energydammy_2
wait
done &

#########################################################################################2

########################################################################################################3

###kloop = i ~ 1/8(max-Amari)####
for i3 in `seq $Parallel_2_ $Parallel_3` 
do

touch eigen_3 &
touch eigen2_3 &
touch TEST1_3 &
touch TEST2_3 &
touch TEST3_3 &
touch TEST4_3 &
touch test3_3 &
touch test4_3 &
touch test5_3 &
wait

echo "Colliner_donwspin::kloop kx ky kz:$i3/$Parallel_3" 


grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_3 &

wait
paste TEST1_3 TEST2_3 TEST3_3 > TEST4_3

j3=`expr $i3 + 1`
ei3=$(grep "kloop="$j3$ -B 2 $File_Name.out3 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k3=`expr $ei3 + 1`
P3=`expr $ei3 + 4`
grep  "kloop="$i3$ -A $P3 $File_Name.out3| grep "kloop="$j3$ -B $k3 | head -n $ei3 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_3  &
wait

sort -n test3_3 > test4_3
more test4_3 | wc > eigen_3
more eigen_3 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_3
paste TEST4_3 eigen2_3 > TEST5_3
cat TEST5_3 test4_3 > test5_3



cat test5_3 $File_Name.energydammy_3 >> $File_Name.energy_3
wait 
rm test3_3 &
rm test4_3 &
rm test5_3 &
rm TEST1_3 &
rm TEST2_3 &
rm TEST3_3 &
rm TEST4_3 &
rm TEST5_3 &
rm eigen2_3 &
rm eigen_3 &
rm $File_Name.energydammy_3  
touch $File_Name.energydammy_3
wait
done &

#########################################################################################3

########################################################################################################4

###kloop = i ~ 1/8(max-Amari)####
for i4 in `seq $Parallel_3_ $Parallel_4` 
do
touch eigen_4 &
touch eigen2_4 &
touch TEST1_4 &
touch TEST2_4 &
touch TEST3_4 &
touch TEST4_4 &
touch test3_4 &
touch test4_4 &
touch test5_4 &
wait

echo "Colliner_donwspin::kloop kx ky kz:$i4/$Parallel_4" 


grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_4 &

wait
paste TEST1_4 TEST2_4 TEST3_4 > TEST4_4

j4=`expr $i4 + 1`
ei4=$(grep "kloop="$j4$ -B 2 $File_Name.out4 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k4=`expr $ei4 + 1`
P4=`expr $ei4 + 4`
grep  "kloop="$i4$ -A $P4 $File_Name.out4| grep "kloop="$j4$ -B $k4 | head -n $ei4 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_4  &
wait

sort -n test3_4 > test4_4
more test4_4 | wc > eigen_4
more eigen_4 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_4
paste TEST4_4 eigen2_4 > TEST5_4
cat TEST5_4 test4_4 > test5_4



cat test5_4 $File_Name.energydammy_4 >> $File_Name.energy_4
wait 
rm test3_4 &
rm test4_4 &
rm test5_4 &
rm TEST1_4 &
rm TEST2_4 &
rm TEST3_4 &
rm TEST4_4 &
rm TEST5_4 &
rm eigen2_4 &
rm eigen_4 &
rm $File_Name.energydammy_4  
touch $File_Name.energydammy_4
wait
done &

#########################################################################################4

########################################################################################################5

###kloop = i ~ 1/8(max-Amari)####
for i5 in `seq $Parallel_4_ $Parallel_5` 
do
touch eigen_5 &
touch eigen2_5 &
touch TEST1_5 &
touch TEST2_5 &
touch TEST3_5 &
touch TEST4_5 &
touch test3_5 &
touch test4_5 &
touch test5_5 &
wait

echo "Colliner_donwspin::kloop kx ky kz:$i5/$Parallel_5" 


grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_5 &

wait
paste TEST1_5 TEST2_5 TEST3_5 > TEST4_5

j5=`expr $i5 + 1`
ei5=$(grep "kloop="$j5$ -B 2 $File_Name.out5 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k5=`expr $ei5 + 1`
P5=`expr $ei5 + 4`
grep  "kloop="$i5$ -A $P5 $File_Name.out5| grep "kloop="$j5$ -B $k5 | head -n $ei5 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_5  &
wait

sort -n test3_5 > test4_5
more test4_5 | wc > eigen_5
more eigen_5 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_5
paste TEST4_5 eigen2_5 > TEST5_5
cat TEST5_5 test4_5 > test5_5



cat test5_5 $File_Name.energydammy_5 >> $File_Name.energy_5
wait 
rm test3_5 &
rm test4_5 &
rm test5_5 &
rm TEST1_5 &
rm TEST2_5 &
rm TEST3_5 &
rm TEST4_5 &
rm TEST5_5 &
rm eigen2_5 &
rm eigen_5 &
rm $File_Name.energydammy_5  
touch $File_Name.energydammy_5
wait
done &

#########################################################################################5

########################################################################################################6

###kloop = i ~ 1/8(max-Amari)####
for i6 in `seq $Parallel_5_ $Parallel_6` 
do
touch eigen_6 &
touch eigen2_6 &
touch TEST1_6 &
touch TEST2_6 &
touch TEST3_6 &
touch TEST4_6 &
touch test3_6 &
touch test4_6 &
touch test5_6 &

wait

echo "Colliner_donwspin::kloop kx ky kz:$i6/$Parallel_6" 


grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_6 &

wait
paste TEST1_6 TEST2_6 TEST3_6 > TEST4_6

j6=`expr $i6 + 1`
ei6=$(grep "kloop="$j6$ -B 2 $File_Name.out6 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k6=`expr $ei6 + 1`
P6=`expr $ei6 + 4`
grep  "kloop="$i6$ -A $P6 $File_Name.out6| grep "kloop="$j6$ -B $k6 | head -n $ei6 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_6  &
wait

sort -n test3_6 > test4_6
more test4_6 | wc > eigen_6
more eigen_6 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_6
paste TEST4_6 eigen2_6 > TEST5_6
cat TEST5_6 test4_6 > test5_6



cat test5_6 $File_Name.energydammy_6 >> $File_Name.energy_6
wait 
rm test3_6 &
rm test4_6 &
rm test5_6 &
rm TEST1_6 &
rm TEST2_6 &
rm TEST3_6 &
rm TEST4_6 &
rm TEST5_6 &
rm eigen2_6 &
rm eigen_6 &
rm $File_Name.energydammy_6  
touch $File_Name.energydammy_6
wait
done &

#########################################################################################6

########################################################################################################7

###kloop = i ~ 1/8(max-Amari)####
for i7 in `seq $Parallel_6_ $Parallel_7` 
do
touch eigen_7 &
touch eigen2_7 &
touch TEST1_7 &
touch TEST2_7 &
touch TEST3_7 &
touch TEST4_7 &
touch test3_7 &
touch test4_7 &
touch test5_7 &

wait

echo "Colliner_donwspin::kloop kx ky kz:$i7/$Parallel_7" 


grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_7 &

wait
paste TEST1_7 TEST2_7 TEST3_7 > TEST4_7

j7=`expr $i7 + 1`
ei7=$(grep "kloop="$j7$ -B 2 $File_Name.out7 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k7=`expr $ei7 + 1`
P7=`expr $ei7 + 4`
grep  "kloop="$i7$ -A $P7 $File_Name.out7| grep "kloop="$j7$ -B $k7 | head -n $ei7 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_7  &
wait

sort -n test3_7 > test4_7
more test4_7 | wc > eigen_7
more eigen_7 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_7
paste TEST4_7 eigen2_7 > TEST5_7
cat TEST5_7 test4_7 > test5_7



cat test5_7 $File_Name.energydammy_7 >> $File_Name.energy_7
wait 
rm test3_7 &
rm test4_7 &
rm test5_7 &
rm TEST1_7 &
rm TEST2_7 &
rm TEST3_7 &
rm TEST4_7 &
rm eigen2_7 &
rm eigen_7 &
rm TEST5_7 &
rm $File_Name.energydammy_7  
touch $File_Name.energydammy_7
wait
done &

#########################################################################################7

########################################################################################################8

###kloop = i ~ 1/8(max-Amari)####
for i8 in `seq $Parallel_7_ $kloooopmax` 
do 
touch eigen_8 &
touch eigen2_8 &
touch TEST1_8 &
touch TEST2_8 &
touch TEST3_8 &
touch TEST4_8 &
touch test3_8 &
touch test4_8 &
touch test5_8 &

wait 
echo "Colliner_donwspin::kloop kx ky kz:$i8/$kloooopmax" 


grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_8 &

wait
paste TEST1_8 TEST2_8 TEST3_8 > TEST4_8

j8=`expr $i8 + 1`
ei8=$(grep "kloop="$j8$ -B 2 $File_Name.out8 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k8=`expr $ei8 + 1`
P8=`expr $ei8 + 4`
grep  "kloop="$i8$ -A $P8 $File_Name.out8| grep "kloop="$j8$ -B $k8 | head -n $ei8 | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_8  &
wait

sort -n test3_8 > test4_8
more test4_8 | wc > eigen_8
more eigen_8 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_8
paste TEST4_8 eigen2_8 > TEST5_8
cat TEST5_8 test4_8 > test5_8



cat test5_8 $File_Name.energydammy_8 >> $File_Name.energy_8
wait 
rm test3_8 &
rm test4_8 &
rm test5_8 &
rm TEST1_8 &
rm TEST2_8 &
rm TEST3_8 &
rm TEST4_8 &
rm TEST5_8 &
rm eigen2_8 &
rm eigen_8 &
rm $File_Name.energydammy_8  
touch $File_Name.energydammy_8
wait
done &
wait
#########################################################################################8

cat $File_Name.energy_1 $File_Name.energy_2 $File_Name.energy_3 $File_Name.energy_4 $File_Name.energy_5 $File_Name.energy_6 $File_Name.energy_7 $File_Name.energy_8 > $File_Name.energydn

wait 
#####kloop = max####
touch $File_Name.energydammy
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1 &
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2 &
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3 &
wait
paste TEST1 TEST2 TEST3 > TEST4
ei=$(grep "kloop="$kloooopmax$ -B 2 $outputfile | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)
U=`expr $ei + 2`
grep  "kloop="$kloopmax$ -A $U $outputfile| tail -n $ei | sed 's/[\t ]\+/\t/g' | cut -f4 | awk '{ OFMT = "%.14f"}{print $1*2}' > test3  &
wait

sort -n test3 > test4
more test4 | wc > eigen
more eigen | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2
paste TEST4 eigen2 > TEST5
cat TEST5 test4 > test5



cat test5 $File_Name.energydammy >> $File_Name.energydn

rm test3 &
rm test4 &
rm test5 &

rm TEST1 &
rm TEST2 &
rm TEST3 &
rm TEST4 &
rm TEST5 &
rm $File_Name.energydammy  









sed -i "1s/^/$kloooooopmax\n/" $File_Name.energydn
sed -i '1s/^/Energy file of BoltzTrap for OpenMX\n/' $File_Name.energydn

echo -e ".energydn file for BoltzTraP has been generated.\n"

touch $File_Name.struct
LatticeUnit=$(grep Atoms.UnitVectors.Unit $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 2)

if grep 'a1 =' $outputfile >/dev/null; then

 grep "a1 = " -A 2 $outputfile | sed -e 's/^[ ]*//g' | sed 's/[\t ]\+/\t/g' | head -n 3 | awk '{ OFMT = "%.14f"}{print $3*1.889725989, $4*1.889725989, $5*1.889725989}' > $File_Name.struct

else

if [ $LatticeUnit = Ang -o $LatticeUnit = ang ] ; then #if 6

 awk '/<Atoms.UnitVectors/,/Atoms.UnitVectors>/' $outputfile | grep '\S' | tail -n 4 | head -n 3 | awk '{ OFMT = "%.14f"}{print $1*1.889725989, $2*1.889725989, $3*1.889725989}'  > $File_Name.struct

else #else 6
 awk '/<Atoms.UnitVectors/,/Atoms.UnitVectors>/' $outputfile | grep '\S' | tail -n 4 | head -n 3 | awk '{ OFMT = "%.14f"}{print $1, $2, $3}'  > $File_Name.struct

fi #fi 6


fi



echo -e "1" >> $File_Name.struct
echo -e "1 0 0 0 1 0 0 0 1" >> $File_Name.struct

sed -i '1s/^/Structure file of BoltzTrap for OpenMX\n/' $File_Name.struct

echo -e ".struct file for BoltzTraP has been generated.\n"

touch $File_Name.intrans
touch $File_Name.intrans_
Chemicalpotential=$(grep Chemical $outputfile | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 6| awk '{ OFMT = "%.14f"}{print $1*2}') 
####20170829_added####
grep scf.ElectronicTemperature $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 1 | grep "#" > damm
commentoutswitch=$(echo $?)
rm damm
if [ $commentoutswitch = 0 ] ; then

ElectronicTemperature=$(echo 300)

else

grep scf.ElectronicTemperature $File_Name.out > damm
tempswich=$(echo $?)
rm damm
if [ $tempswich = 0 ] ; then 

ElectronicTemperature=$(grep scf.ElectronicTemperature $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 2)

else 
ElectronicTemperature=$(echo 300)

fi

fi
####20170829_added####

Electron_number=$(grep "Number of States" $outputfile | sed 's/[\t ]\+/\t/g' | cut -f 6 | awk '{s=($0<0)?-1:1;print int($0*s*1000+0.5)/1000/s;}')


echo -e "GENE                      # Format of DOS\n" > $File_Name.intrans_
echo -e "0 0 0 0.0                 # iskip (not presently used) idebug setgap shiftgap\n" >> $File_Name.intrans_
echo -e "$Chemicalpotential 0.0005 0.4 $Electron_number   # Fermilevel (Ry), energygrid, energy span around Fermilevel, number of electrons\n" >> $File_Name.intrans_
echo -e "CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n" >> $File_Name.intrans_
echo -e "10                         # lpfac, number of latt-points per k-point\n" >> $File_Name.intrans_
echo -e "BOLTZ                     # run mode (only BOLTZ is supported)\n" >> $File_Name.intrans_
echo -e ".30                       # (efcut) energy range of chemical potential\n" >> $File_Name.intrans_
echo -e "$ElectronicTemperature $ElectronicTemperature              # Tmax, temperature grid\n" >> $File_Name.intrans_
echo -e "-1.                       # energyrange of bands given individual DOS output sig_xxx and dos_xxx (xxx is band number)\n" >> $File_Name.intrans_
echo -e "HISTO\n" >> $File_Name.intrans_
####20170829_added####

echo -e "0 0 0 0 0\n" >> $File_Name.intrans_
echo -e "1\n" >> $File_Name.intrans_
echo -e "0\n" >> $File_Name.intrans_

####20170829_added####

 grep -v '^\s*$' $File_Name.intrans_ > $File_Name.intrans
rm $File_Name.intrans_
echo -e ".intrans file for BoltzTraP has been generated.\n"

echo -e "Conversion has been finished.\n"
echo -e "Directory is $File_Name\n" 
rm  $File_Name.out1 &
rm  $File_Name.out2 &
rm  $File_Name.out3 &
rm  $File_Name.out4 &
rm  $File_Name.out5 &
rm  $File_Name.out6 &
rm  $File_Name.out7 &
rm  $File_Name.out8 &


rm  $File_Name.energy_1 &
rm  $File_Name.energy_2 &
rm  $File_Name.energy_3 &
rm  $File_Name.energy_4 &
rm  $File_Name.energy_5 &
rm  $File_Name.energy_6 &
rm  $File_Name.energy_7 &
rm  $File_Name.energy_8 &

rm $File_Name.energydammy_1 &
rm $File_Name.energydammy_2 &
rm $File_Name.energydammy_3 &
rm $File_Name.energydammy_4 &
rm $File_Name.energydammy_5 &
rm $File_Name.energydammy_6 &
rm $File_Name.energydammy_7 &
rm $File_Name.energydammy_8 &
rm eigen &

rm eigen2 &
wait

else #4


##Start::  Average band (up down spin) ###

 echo -e  "Please wait. Generating .energy file ..."

kloopmax=$(grep kloop $outputfile | tail -n 1 | cut -c 10-20)
kloooopmax=`expr $kloopmax - 1`
kloooooopmax=`expr $kloopmax + 1`
rm  $File_Name.energy
touch $File_Name.energy
touch $File_Name.energydammy_1
touch $File_Name.energydammy_2
touch $File_Name.energydammy_3
touch $File_Name.energydammy_4
touch $File_Name.energydammy_5
touch $File_Name.energydammy_6
touch $File_Name.energydammy_7
touch $File_Name.energydammy_8
touch $File_Name.energy_1
touch $File_Name.energy_2
touch $File_Name.energy_3
touch $File_Name.energy_4
touch $File_Name.energy_5
touch $File_Name.energy_6
touch $File_Name.energy_7
touch $File_Name.energy_8

cp $outputfile $File_Name.out1
cp $outputfile $File_Name.out2
cp $outputfile $File_Name.out3
cp $outputfile $File_Name.out4
cp $outputfile $File_Name.out5
cp $outputfile $File_Name.out6
cp $outputfile $File_Name.out7
cp $outputfile $File_Name.out8



Amari=`expr $kloopmax % 8`
Amari_2=`expr $kloopmax - $Amari`

Parallel_1=`expr \( $Amari_2 / 8 \) \* 1`
Parallel_2=`expr \( $Amari_2 / 8 \) \* 2`
Parallel_3=`expr \( $Amari_2 / 8 \) \* 3`
Parallel_4=`expr \( $Amari_2 / 8 \) \* 4`
Parallel_5=`expr \( $Amari_2 / 8 \) \* 5`
Parallel_6=`expr \( $Amari_2 / 8 \) \* 6`
Parallel_7=`expr \( $Amari_2 / 8 \) \* 7`
Parallel_8=`expr \( $Amari_2 / 8 \) \* 8`



Parallel_1_=`expr $Parallel_1 + 1`
Parallel_2_=`expr $Parallel_2 + 1`
Parallel_3_=`expr $Parallel_3 + 1`
Parallel_4_=`expr $Parallel_4 + 1`
Parallel_5_=`expr $Parallel_5 + 1`
Parallel_6_=`expr $Parallel_6 + 1`
Parallel_7_=`expr $Parallel_7 + 1`

wait
########################################################################################################1

###kloop = i ~ 1/8(max-Amari)####
for i in `seq 0 $Parallel_1` 
do

touch eigen_1 &
touch eigen2_1 &
touch TEST1_1 &
touch TEST2_1 &
touch TEST3_1 &
touch TEST4_1 &
touch test3_1 &
touch test4_1 &
touch test5_1 &
wait 
 
echo "Nospin::kloop kx ky kz:$i/$Parallel_1" 


grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_1 &
grep "kloop="$i$ -A 1 $File_Name.out1   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_1 &

wait
paste TEST1_1 TEST2_1 TEST3_1 > TEST4_1

j=`expr $i + 1`
ei=$(grep "kloop="$j$ -B 2 $File_Name.out1 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k=`expr $ei + 1`
P=`expr $ei + 4`
grep  "kloop="$i$ -A $P $File_Name.out1| grep "kloop="$j$ -B $k | head -n $ei | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_1  &
wait

sort -n test3_1 > test4_1
cat test4_1 | wc > eigen_1
cat eigen_1 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_1
paste TEST4_1 eigen2_1 > TEST5_1
cat TEST5_1 test4_1 > test5_1



cat test5_1 $File_Name.energydammy_1 >> $File_Name.energy_1
wait 
rm test3_1 &
rm test4_1 &
rm test5_1 &
rm TEST1_1 &
rm TEST2_1 &
rm TEST3_1 &
rm TEST4_1 &
rm TEST5_1 &
rm eigen2_1 &
rm eigen_1 &

rm $File_Name.energydammy_1  
touch $File_Name.energydammy_1
wait
done &

#########################################################################################1

########################################################################################################2

###kloop = i ~ 1/8(max-Amari)####
for i2 in `seq $Parallel_1_ $Parallel_2` 
do 

touch eigen_2 &
touch eigen2_2 &
touch TEST1_2 &
touch TEST2_2 &
touch TEST3_2 &
touch TEST4_2 &
touch test3_2 &
touch test4_2 &
touch test5_2 &
wait

echo "Nospin::kloop kx ky kz:$i2/$Parallel_2" 


grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_2 &
grep "kloop="$i2$ -A 1 $File_Name.out2   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_2 &

wait
paste TEST1_2 TEST2_2 TEST3_2 > TEST4_2

j2=`expr $i2 + 1`
ei2=$(grep "kloop="$j2$ -B 2 $File_Name.out2 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k2=`expr $ei2 + 1`
P2=`expr $ei2 + 4`
grep  "kloop="$i2$ -A $P2 $File_Name.out2 | grep "kloop="$j2$ -B $k2 | head -n $ei2 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_2  &
wait

sort -n test3_2 > test4_2
more test4_2 | wc > eigen_2
more eigen_2 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_2
paste TEST4_2 eigen2_2 > TEST5_2
cat TEST5_2 test4_2 > test5_2



cat test5_2 $File_Name.energydammy_2 >> $File_Name.energy_2
wait 
rm test3_2 &
rm test4_2 &
rm test5_2 &
rm TEST1_2 &
rm TEST2_2 &
rm TEST3_2 &
rm TEST4_2 &
rm TEST5_2 &
rm eigen2_2 &
rm eigen_2 &
rm $File_Name.energydammy_2  
touch $File_Name.energydammy_2
wait
done &

#########################################################################################2

########################################################################################################3

###kloop = i ~ 1/8(max-Amari)####
for i3 in `seq $Parallel_2_ $Parallel_3` 
do

touch eigen_3 &
touch eigen2_3 &
touch TEST1_3 &
touch TEST2_3 &
touch TEST3_3 &
touch TEST4_3 &
touch test3_3 &
touch test4_3 &
touch test5_3 &
wait

echo "Nospin::kloop kx ky kz:$i3/$Parallel_3" 


grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_3 &
grep "kloop="$i3$ -A 1 $File_Name.out3   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_3 &

wait
paste TEST1_3 TEST2_3 TEST3_3 > TEST4_3

j3=`expr $i3 + 1`
ei3=$(grep "kloop="$j3$ -B 2 $File_Name.out3 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k3=`expr $ei3 + 1`
P3=`expr $ei3 + 4`
grep  "kloop="$i3$ -A $P3 $File_Name.out3| grep "kloop="$j3$ -B $k3 | head -n $ei3 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_3  &
wait

sort -n test3_3 > test4_3
more test4_3 | wc > eigen_3
more eigen_3 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_3
paste TEST4_3 eigen2_3 > TEST5_3
cat TEST5_3 test4_3 > test5_3



cat test5_3 $File_Name.energydammy_3 >> $File_Name.energy_3
wait 
rm test3_3 &
rm test4_3 &
rm test5_3 &
rm TEST1_3 &
rm TEST2_3 &
rm TEST3_3 &
rm TEST4_3 &
rm TEST5_3 &
rm eigen2_3 &
rm eigen_3 &
rm $File_Name.energydammy_3  
touch $File_Name.energydammy_3
wait
done &

#########################################################################################3

########################################################################################################4

###kloop = i ~ 1/8(max-Amari)####
for i4 in `seq $Parallel_3_ $Parallel_4` 
do
touch eigen_4 &
touch eigen2_4 &
touch TEST1_4 &
touch TEST2_4 &
touch TEST3_4 &
touch TEST4_4 &
touch test3_4 &
touch test4_4 &
touch test5_4 &
wait

echo "Nospin::kloop kx ky kz:$i4/$Parallel_4" 


grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_4 &
grep "kloop="$i4$ -A 1 $File_Name.out4   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_4 &

wait
paste TEST1_4 TEST2_4 TEST3_4 > TEST4_4

j4=`expr $i4 + 1`
ei4=$(grep "kloop="$j4$ -B 2 $File_Name.out4 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k4=`expr $ei4 + 1`
P4=`expr $ei4 + 4`
grep  "kloop="$i4$ -A $P4 $File_Name.out4| grep "kloop="$j4$ -B $k4 | head -n $ei4 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_4  &
wait

sort -n test3_4 > test4_4
more test4_4 | wc > eigen_4
more eigen_4 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_4
paste TEST4_4 eigen2_4 > TEST5_4
cat TEST5_4 test4_4 > test5_4



cat test5_4 $File_Name.energydammy_4 >> $File_Name.energy_4
wait 
rm test3_4 &
rm test4_4 &
rm test5_4 &
rm TEST1_4 &
rm TEST2_4 &
rm TEST3_4 &
rm TEST4_4 &
rm TEST5_4 &
rm eigen2_4 &
rm eigen_4 &
rm $File_Name.energydammy_4  
touch $File_Name.energydammy_4
wait
done &

#########################################################################################4

########################################################################################################5

###kloop = i ~ 1/8(max-Amari)####
for i5 in `seq $Parallel_4_ $Parallel_5` 
do
touch eigen_5 &
touch eigen2_5 &
touch TEST1_5 &
touch TEST2_5 &
touch TEST3_5 &
touch TEST4_5 &
touch test3_5 &
touch test4_5 &
touch test5_5 &
wait

echo "Nospin::kloop kx ky kz:$i5/$Parallel_5" 


grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_5 &
grep "kloop="$i5$ -A 1 $File_Name.out5   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_5 &

wait
paste TEST1_5 TEST2_5 TEST3_5 > TEST4_5

j5=`expr $i5 + 1`
ei5=$(grep "kloop="$j5$ -B 2 $File_Name.out5 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k5=`expr $ei5 + 1`
P5=`expr $ei5 + 4`
grep  "kloop="$i5$ -A $P5 $File_Name.out5| grep "kloop="$j5$ -B $k5 | head -n $ei5 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_5  &
wait

sort -n test3_5 > test4_5
more test4_5 | wc > eigen_5
more eigen_5 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_5
paste TEST4_5 eigen2_5 > TEST5_5
cat TEST5_5 test4_5 > test5_5



cat test5_5 $File_Name.energydammy_5 >> $File_Name.energy_5
wait 
rm test3_5 &
rm test4_5 &
rm test5_5 &
rm TEST1_5 &
rm TEST2_5 &
rm TEST3_5 &
rm TEST4_5 &
rm TEST5_5 &
rm eigen2_5 &
rm eigen_5 &
rm $File_Name.energydammy_5  
touch $File_Name.energydammy_5
wait
done &

#########################################################################################5

########################################################################################################6

###kloop = i ~ 1/8(max-Amari)####
for i6 in `seq $Parallel_5_ $Parallel_6` 
do
touch eigen_6 &
touch eigen2_6 &
touch TEST1_6 &
touch TEST2_6 &
touch TEST3_6 &
touch TEST4_6 &
touch test3_6 &
touch test4_6 &
touch test5_6 &

wait

echo "Nospin::kloop kx ky kz:$i6/$Parallel_6" 


grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_6 &
grep "kloop="$i6$ -A 1 $File_Name.out6   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_6 &

wait
paste TEST1_6 TEST2_6 TEST3_6 > TEST4_6

j6=`expr $i6 + 1`
ei6=$(grep "kloop="$j6$ -B 2 $File_Name.out6 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k6=`expr $ei6 + 1`
P6=`expr $ei6 + 4`
grep  "kloop="$i6$ -A $P6 $File_Name.out6| grep "kloop="$j6$ -B $k6 | head -n $ei6 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_6  &
wait

sort -n test3_6 > test4_6
more test4_6 | wc > eigen_6
more eigen_6 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_6
paste TEST4_6 eigen2_6 > TEST5_6
cat TEST5_6 test4_6 > test5_6



cat test5_6 $File_Name.energydammy_6 >> $File_Name.energy_6
wait 
rm test3_6 &
rm test4_6 &
rm test5_6 &
rm TEST1_6 &
rm TEST2_6 &
rm TEST3_6 &
rm TEST4_6 &
rm TEST5_6 &
rm eigen2_6 &
rm eigen_6 &
rm $File_Name.energydammy_6  
touch $File_Name.energydammy_6
wait
done &

#########################################################################################6

########################################################################################################7

###kloop = i ~ 1/8(max-Amari)####
for i7 in `seq $Parallel_6_ $Parallel_7` 
do
touch eigen_7 &
touch eigen2_7 &
touch TEST1_7 &
touch TEST2_7 &
touch TEST3_7 &
touch TEST4_7 &
touch test3_7 &
touch test4_7 &
touch test5_7 &

wait

echo "Nospin::kloop kx ky kz:$i7/$Parallel_7" 


grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_7 &
grep "kloop="$i7$ -A 1 $File_Name.out7   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_7 &

wait
paste TEST1_7 TEST2_7 TEST3_7 > TEST4_7

j7=`expr $i7 + 1`
ei7=$(grep "kloop="$j7$ -B 2 $File_Name.out7 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k7=`expr $ei7 + 1`
P7=`expr $ei7 + 4`
grep  "kloop="$i7$ -A $P7 $File_Name.out7| grep "kloop="$j7$ -B $k7 | head -n $ei7 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_7  &
wait

sort -n test3_7 > test4_7
more test4_7 | wc > eigen_7
more eigen_7 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_7
paste TEST4_7 eigen2_7 > TEST5_7
cat TEST5_7 test4_7 > test5_7



cat test5_7 $File_Name.energydammy_7 >> $File_Name.energy_7
wait 
rm test3_7 &
rm test4_7 &
rm test5_7 &
rm TEST1_7 &
rm TEST2_7 &
rm TEST3_7 &
rm TEST4_7 &
rm eigen2_7 &
rm eigen_7 &
rm TEST5_7 &
rm $File_Name.energydammy_7  
touch $File_Name.energydammy_7
wait
done &

#########################################################################################7

########################################################################################################8

###kloop = i ~ 1/8(max-Amari)####
for i8 in `seq $Parallel_7_ $kloooopmax` 
do 
touch eigen_8 &
touch eigen2_8 &
touch TEST1_8 &
touch TEST2_8 &
touch TEST3_8 &
touch TEST4_8 &
touch test3_8 &
touch test4_8 &
touch test5_8 &

wait 
echo "Nospin::kloop kx ky kz:$i8/$kloooopmax" 


grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2_8 &
grep "kloop="$i8$ -A 1 $File_Name.out8   | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3_8 &

wait
paste TEST1_8 TEST2_8 TEST3_8 > TEST4_8

j8=`expr $i8 + 1`
ei8=$(grep "kloop="$j8$ -B 2 $File_Name.out8 | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)


k8=`expr $ei8 + 1`
P8=`expr $ei8 + 4`
grep  "kloop="$i8$ -A $P8 $File_Name.out8| grep "kloop="$j8$ -B $k8 | head -n $ei8 | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}'  > test3_8  &
wait

sort -n test3_8 > test4_8
more test4_8 | wc > eigen_8
more eigen_8 | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2_8
paste TEST4_8 eigen2_8 > TEST5_8
cat TEST5_8 test4_8 > test5_8



cat test5_8 $File_Name.energydammy_8 >> $File_Name.energy_8
wait 
rm test3_8 &
rm test4_8 &
rm test5_8 &
rm TEST1_8 &
rm TEST2_8 &
rm TEST3_8 &
rm TEST4_8 &
rm TEST5_8 &
rm eigen2_8 &
rm eigen_8 &
rm $File_Name.energydammy_8  
touch $File_Name.energydammy_8
wait
done &
wait
#########################################################################################8

cat $File_Name.energy_1 $File_Name.energy_2 $File_Name.energy_3 $File_Name.energy_4 $File_Name.energy_5 $File_Name.energy_6 $File_Name.energy_7 $File_Name.energy_8 > $File_Name.energy

wait 
#####kloop = max####
touch $File_Name.energydammy
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 3 > TEST1 &
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 5 > TEST2 &
grep "kloop="$kloopmax$ -A 1 $outputfile  | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 7 > TEST3 &
wait
paste TEST1 TEST2 TEST3 > TEST4
ei=$(grep "kloop="$kloooopmax$ -B 2 $outputfile | head -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 2)
U=`expr $ei + 2`
grep  "kloop="$kloopmax$ -A $U $outputfile| tail -n $ei | sed 's/[\t ]\+/\t/g' | cut -f3 | awk '{ OFMT = "%.14f"}{print $1*2}' > test3  &
wait

sort -n test3 > test4
more test4 | wc > eigen
more eigen | sed 's/[\t ]\+/\t/g' | cut -f2 > eigen2
paste TEST4 eigen2 > TEST5
cat TEST5 test4 > test5



cat test5 $File_Name.energydammy >> $File_Name.energy

rm test3 &
rm test4 &
rm test5 &

rm TEST1 &
rm TEST2 &
rm TEST3 &
rm TEST4 &
rm TEST5 &
rm $File_Name.energydammy  
touch $File_Name.energydammy



sed -i "1s/^/$kloooooopmax\n/" $File_Name.energy
sed -i '1s/^/Energy file of BoltzTrap for OpenMX\n/' $File_Name.energy

echo -e ".energy file for BoltzTraP has been generated.\n"




touch $File_Name.struct
LatticeUnit=$(grep Atoms.UnitVectors.Unit $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 2 | sed -e 's/\(^.*\)/\U\1/' | cat)

if grep 'a1 =' $outputfile >/dev/null; then

 grep "a1 = " -A 2 $outputfile | sed -e 's/^[ ]*//g' | sed 's/[\t ]\+/\t/g' | head -n 3 | awk '{ OFMT = "%.14f"}{print $3*1.889725989, $4*1.889725989, $5*1.889725989}' > $File_Name.struct

else

if [ $LatticeUnit = ANG -o $LatticeUnit = ang ] ; then #if 6

 awk '/<Atoms.UnitVectors/,/Atoms.UnitVectors>/' $outputfile | grep '\S' | tail -n 4 | head -n 3 | awk '{ OFMT = "%.14f"}{print $1*1.889725989, $2*1.889725989, $3*1.889725989}'  > $File_Name.struct

else #else 6
 awk '/<Atoms.UnitVectors/,/Atoms.UnitVectors>/' $outputfile | grep '\S' | tail -n 4 | head -n 3 | awk '{ OFMT = "%.14f"}{print $1, $2, $3}'  > $File_Name.struct

fi #fi 6


fi



echo -e "1" >> $File_Name.struct
echo -e "1 0 0 0 1 0 0 0 1" >> $File_Name.struct

sed -i '1s/^/Structure file of BoltzTrap for OpenMX\n/' $File_Name.struct

echo -e ".struct file for BoltzTraP has been generated.\n"

touch $File_Name.intrans
touch $File_Name.intrans_
Chemicalpotential=$(grep Chemical $outputfile | tail -n 1 | sed 's/[\t ]\+/\t/g' | cut -f 6| awk '{ OFMT = "%.14f"}{print $1*2}') 

Electron_number=$(grep "Number of States" $outputfile | sed 's/[\t ]\+/\t/g' | cut -f 6 | awk '{s=($0<0)?-1:1;print int($0*s*1000+0.5)/1000/s;}')

####20170829_added####
grep scf.ElectronicTemperature $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 1 | grep "#" > damm
commentoutswitch=$(echo $?)
rm damm
if [ $commentoutswitch = 0 ] ; then

ElectronicTemperature=$(echo 300)

else

grep scf.ElectronicTemperature $File_Name.out > damm
tempswich=$(echo $?)
rm damm
if [ $tempswich = 0 ] ; then 

ElectronicTemperature=$(grep scf.ElectronicTemperature $File_Name.out | sed 's/[\t ]\+/\t/g' | sed 's/^[ \t]*//' | cut -f 2)

else 
ElectronicTemperature=$(echo 300)

fi

fi
####20170829_added####


echo -e "GENE                      # Format of DOS\n" > $File_Name.intrans_
echo -e "0 0 0 0.0                 # iskip (not presently used) idebug setgap shiftgap\n" >> $File_Name.intrans_
echo -e "$Chemicalpotential 0.0005 0.4 $Electron_number   # Fermilevel (Ry), energygrid, energy span around Fermilevel, number of electrons\n" >> $File_Name.intrans_
echo -e "CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n" >> $File_Name.intrans_
echo -e "10                         # lpfac, number of latt-points per k-point\n" >> $File_Name.intrans_
echo -e "BOLTZ                     # run mode (only BOLTZ is supported)\n" >> $File_Name.intrans_
echo -e ".30                       # (efcut) energy range of chemical potential\n" >> $File_Name.intrans_
echo -e "$ElectronicTemperature $ElectronicTemperature                  # Tmax, temperature grid\n" >> $File_Name.intrans_
echo -e "-1.                       # energyrange of bands given individual DOS output sig_xxx and dos_xxx (xxx is band number)\n" >> $File_Name.intrans_
echo -e "HISTO\n" >> $File_Name.intrans_
####20170829_added####

echo -e "0 0 0 0 0\n" >> $File_Name.intrans_
echo -e "1\n" >> $File_Name.intrans_
echo -e "0\n" >> $File_Name.intrans_

####20170829_added####

 grep -v '^\s*$' $File_Name.intrans_ > $File_Name.intrans
rm $File_Name.intrans_
echo -e ".intrans file for BoltzTraP has been generated.\n"

echo -e "Conversion has been finished.\n"
echo -e "Directory is $File_Name\n" 
rm  $File_Name.out1 &
rm  $File_Name.out2 &
rm  $File_Name.out3 &
rm  $File_Name.out4 &
rm  $File_Name.out5 &
rm  $File_Name.out6 &
rm  $File_Name.out7 &
rm  $File_Name.out8 &


rm  $File_Name.energy_1 &
rm  $File_Name.energy_2 &
rm  $File_Name.energy_3 &
rm  $File_Name.energy_4 &
rm  $File_Name.energy_5 &
rm  $File_Name.energy_6 &
rm  $File_Name.energy_7 &
rm  $File_Name.energy_8 &
rm $File_Name.energydammy &
rm $File_Name.energydammy_1 &
rm $File_Name.energydammy_2 &
rm $File_Name.energydammy_3 &
rm $File_Name.energydammy_4 &
rm $File_Name.energydammy_5 &
rm $File_Name.energydammy_6 &
rm $File_Name.energydammy_7 &
rm $File_Name.energydammy_8 &

rm eigen &

rm eigen2 &
wait





##END::  spinpolarization  on###





fi  #fi 4

fi  #fi 2

exit 0

else

echo "Enter the .out file!!"

exit 1

fi
