awk 'BEGIN  {srand()} 
     !/^$/  { if (rand() <= .3 || FNR==1) print > "data.txt"}' data.raw