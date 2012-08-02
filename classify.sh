echo "Enter text to be classified, hit return to run classification."
read text

if [ `echo "$text" | sed -r 's/ +/ /g' | bin/stupidfilter data/c_rbf` = "1.000000" ]
 then
  echo "Text is not likely to be stupid."
fi

if [ `echo "$text" | sed -r 's/ +/ /g' | bin/stupidfilter data/c_rbf` = "0.000000" ]
 then
  echo "Text is likely to be stupid."
fi

